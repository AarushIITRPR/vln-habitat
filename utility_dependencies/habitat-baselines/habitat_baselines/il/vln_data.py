#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

import habitat
from habitat import logger
from habitat.config import read_write
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.il.models.vln_policy import (
    ACTION_NAMES,
    START_PREV_ACTION,
)

STOP_ACTION = int(HabitatSimActions.stop)


def _config_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class VLNOracleTrajectoryDataset(Dataset):
    r"""Collects teacher-forced VLN trajectories with Habitat's oracle follower.

    Each sample is one R2R episode converted to a sequence of RGB frames,
    previous actions, and oracle action labels. The collector follows the
    episode reference path waypoints and finally issues stop at the goal.
    """

    def __init__(
        self,
        config: Any,
        split: Optional[str] = None,
        max_episodes: int = -1,
        cache_dir: str = "",
        refresh_cache: bool = False,
    ) -> None:
        self.config = config
        self.vln_config = config.habitat_baselines.il.vln
        self.split = split or config.habitat.dataset.split
        self.max_episodes = max_episodes
        self.cache_dir = cache_dir
        self.refresh_cache = refresh_cache
        self.max_trajectory_steps = int(
            _config_get(self.vln_config, "max_trajectory_steps", 200)
        )
        self.use_depth = bool(_config_get(self.vln_config, "use_depth", False))
        self.goal_radius = float(
            _config_get(self.vln_config, "follower_goal_radius", 0.5)
        )
        self.follower_mode = str(
            _config_get(self.vln_config, "follower_mode", "geodesic_path")
        )

        self.samples: List[Dict[str, Any]] = self._load_or_collect()
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No VLN trajectories were collected for split '{self.split}'. "
                "Check that the R2R dataset and MP3D scenes are installed."
            )

    def _cache_path(self) -> str:
        episode_tag = self.max_episodes if self.max_episodes > 0 else "all"
        file_name = (
            f"{self.split}_episodes_{episode_tag}_"
            f"steps_{self.max_trajectory_steps}_"
            f"{'rgbd' if self.use_depth else 'rgb'}.pt"
        )
        return os.path.join(self.cache_dir, file_name)

    def _load_or_collect(self) -> List[Dict[str, Any]]:
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = self._cache_path()
            if os.path.exists(cache_path) and not self.refresh_cache:
                logger.info(f"Loading cached VLN trajectories: {cache_path}")
                return _torch_load(cache_path)

        samples = self._collect_samples()
        if self.cache_dir:
            cache_path = self._cache_path()
            logger.info(f"Saving cached VLN trajectories: {cache_path}")
            torch.save(samples, cache_path)
        return samples

    def _habitat_config_for_split(self) -> Any:
        habitat_config = copy.deepcopy(self.config.habitat)
        with read_write(habitat_config):
            habitat_config.dataset.split = self.split
        return habitat_config

    def _collect_samples(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        habitat_config = self._habitat_config_for_split()

        with habitat.Env(config=habitat_config) as env:
            follower = ShortestPathFollower(
                env.sim, goal_radius=self.goal_radius, return_one_hot=False
            )
            follower.mode = self.follower_mode

            num_episodes = len(env.episodes)
            if self.max_episodes > 0:
                num_episodes = min(num_episodes, self.max_episodes)

            iterator = tqdm(
                range(num_episodes),
                desc=f"Collecting VLN oracle trajectories ({self.split})",
            )
            for _ in iterator:
                observations = env.reset()
                sample = self._rollout_episode(env, follower, observations)
                if sample is not None:
                    samples.append(sample)

        logger.info(
            "Collected %s VLN trajectories for split '%s'",
            len(samples),
            self.split,
        )
        return samples

    def _rollout_episode(
        self,
        env: habitat.Env,
        follower: ShortestPathFollower,
        observations: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        episode = env.current_episode
        instruction = episode.instruction.instruction_text
        targets = list(episode.reference_path)
        if episode.goals:
            targets.append(episode.goals[0].position)

        if len(targets) == 0:
            return None

        rgb_frames: List[Tensor] = []
        depth_frames: List[Tensor] = []
        actions: List[int] = []
        prev_actions: List[int] = []
        prev_action = START_PREV_ACTION

        for target_idx, target in enumerate(targets):
            is_final_target = target_idx == len(targets) - 1
            while not env.episode_over:
                if len(actions) >= self.max_trajectory_steps:
                    break

                best_action = follower.get_next_action(target)
                if (
                    best_action is None
                    or best_action == HabitatSimActions.stop
                ):
                    if is_final_target:
                        prev_action = self._append_step(
                            observations,
                            HabitatSimActions.stop,
                            prev_action,
                            rgb_frames,
                            depth_frames,
                            actions,
                            prev_actions,
                        )
                        env.step(HabitatSimActions.stop)
                    break

                if int(best_action) >= len(ACTION_NAMES):
                    logger.warning(
                        "Skipping unsupported VLN action id %s in episode %s",
                        best_action,
                        episode.episode_id,
                    )
                    break

                prev_action = self._append_step(
                    observations,
                    int(best_action),
                    prev_action,
                    rgb_frames,
                    depth_frames,
                    actions,
                    prev_actions,
                )
                observations = env.step(int(best_action))

            if env.episode_over or len(actions) >= self.max_trajectory_steps:
                break

        if len(actions) == 0:
            return None

        sample = {
            "episode_id": str(episode.episode_id),
            "trajectory_id": getattr(episode, "trajectory_id", -1),
            "instruction": instruction,
            "rgb": torch.stack(rgb_frames, dim=0),
            "actions": torch.tensor(actions, dtype=torch.long),
            "prev_actions": torch.tensor(prev_actions, dtype=torch.long),
        }
        if self.use_depth:
            sample["depth"] = torch.stack(depth_frames, dim=0)
        return sample

    def _append_step(
        self,
        observations: Dict[str, Any],
        action: int,
        prev_action: int,
        rgb_frames: List[Tensor],
        depth_frames: List[Tensor],
        actions: List[int],
        prev_actions: List[int],
    ) -> int:
        rgb_frames.append(self._rgb_to_tensor(observations["rgb"]))
        if self.use_depth:
            depth_frames.append(self._depth_to_tensor(observations["depth"]))
        actions.append(int(action))
        prev_actions.append(int(prev_action))
        return int(action) + 1

    @staticmethod
    def _rgb_to_tensor(rgb: np.ndarray) -> Tensor:
        if rgb.ndim != 3:
            raise ValueError(f"Expected RGB observation HxWxC, got {rgb.shape}")
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        rgb = np.ascontiguousarray(rgb)
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    @staticmethod
    def _depth_to_tensor(depth: np.ndarray) -> Tensor:
        if depth.ndim == 2:
            depth = depth[..., None]
        if depth.ndim != 3:
            raise ValueError(f"Expected depth observation HxWxC, got {depth.shape}")
        if depth.shape[-1] > 1:
            depth = depth[..., :1]
        depth = np.ascontiguousarray(depth.astype(np.float32))
        return torch.from_numpy(depth).permute(2, 0, 1).contiguous()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


class VLNBatchCollator:
    r"""Pads variable-length VLN trajectories and tokenizes instructions."""

    def __init__(
        self,
        tokenizer: Any,
        max_instruction_length: int = 80,
        stop_action_repeat: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_instruction_length = max_instruction_length
        self.stop_action_repeat = max(1, int(stop_action_repeat))

    def _expanded_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if (
            self.stop_action_repeat <= 1
            or sample["actions"].numel() == 0
            or int(sample["actions"][-1].item()) != STOP_ACTION
        ):
            return sample

        repeat = self.stop_action_repeat - 1
        final_rgb = sample["rgb"][-1:].repeat(repeat, 1, 1, 1)
        final_action = sample["actions"][-1:].repeat(repeat)
        # Keep the same previous action context that originally led to STOP.
        final_prev_action = sample["prev_actions"][-1:].repeat(repeat)
        out = dict(sample)
        out["rgb"] = torch.cat([sample["rgb"], final_rgb], dim=0)
        if "depth" in sample:
            final_depth = sample["depth"][-1:].repeat(repeat, 1, 1, 1)
            out["depth"] = torch.cat([sample["depth"], final_depth], dim=0)
        out["actions"] = torch.cat([sample["actions"], final_action], dim=0)
        out["prev_actions"] = torch.cat(
            [sample["prev_actions"], final_prev_action], dim=0
        )
        return out

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [self._expanded_sample(sample) for sample in batch]
        lengths = torch.tensor(
            [sample["actions"].numel() for sample in batch], dtype=torch.long
        )
        max_steps = int(lengths.max().item())
        channels, height, width = batch[0]["rgb"].shape[1:]
        has_depth = "depth" in batch[0]

        rgb = torch.zeros(
            len(batch),
            max_steps,
            channels,
            height,
            width,
            dtype=batch[0]["rgb"].dtype,
        )
        depth = None
        if has_depth:
            depth_channels, depth_height, depth_width = batch[0]["depth"].shape[1:]
            depth = torch.zeros(
                len(batch),
                max_steps,
                depth_channels,
                depth_height,
                depth_width,
                dtype=batch[0]["depth"].dtype,
            )
        actions = torch.full(
            (len(batch), max_steps), -100, dtype=torch.long
        )
        prev_actions = torch.zeros(
            len(batch), max_steps, dtype=torch.long
        )
        action_mask = torch.zeros(
            len(batch), max_steps, dtype=torch.bool
        )

        instructions: List[str] = []
        episode_ids: List[str] = []
        for idx, sample in enumerate(batch):
            steps = sample["actions"].numel()
            rgb[idx, :steps] = sample["rgb"]
            if has_depth and depth is not None:
                depth[idx, :steps] = sample["depth"]
            actions[idx, :steps] = sample["actions"]
            prev_actions[idx, :steps] = sample["prev_actions"]
            action_mask[idx, :steps] = True
            instructions.append(sample["instruction"])
            episode_ids.append(sample["episode_id"])

        tokenized = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.max_instruction_length,
            return_tensors="pt",
        )

        out = {
            "rgb": rgb,
            "actions": actions,
            "prev_actions": prev_actions,
            "action_mask": action_mask,
            "lengths": lengths,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "instructions": instructions,
            "episode_ids": episode_ids,
        }
        if depth is not None:
            out["depth"] = depth
        return out

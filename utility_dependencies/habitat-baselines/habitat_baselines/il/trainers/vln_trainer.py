#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader

import habitat
from habitat import logger
from habitat.config import read_write
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.il.vln_data import (
    VLNBatchCollator,
    VLNOracleTrajectoryDataset,
)
from habitat_baselines.il.models.vln_policy import (
    ACTION_NAMES,
    START_PREV_ACTION,
    VLNPolicyNet,
)


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


@baseline_registry.register_trainer(name="vln")
class VLNILTrainer(BaseILTrainer):
    r"""Imitation-learning trainer for Habitat R2R VLN.

    The trainer builds oracle demonstrations from Habitat reference paths,
    trains a vision-language policy with cross-entropy action supervision, and
    can optionally roll the policy out in Habitat to report SR/SPL.
    """

    supported_tasks = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.device = (
            torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if config is not None:
            logger.info(f"config: {OmegaConf.to_yaml(config)}")

    @property
    def vln_config(self) -> Any:
        return self.config.habitat_baselines.il.vln

    def _make_model(self) -> VLNPolicyNet:
        cfg = self.vln_config
        return VLNPolicyNet(
            visual_backbone=_cfg_get(cfg, "visual_backbone", "resnet18"),
            visual_pretrained=bool(_cfg_get(cfg, "visual_pretrained", True)),
            freeze_visual_encoder=bool(
                _cfg_get(cfg, "freeze_visual_encoder", True)
            ),
            text_model_name=str(
                _cfg_get(cfg, "text_model_name", "distilbert-base-uncased")
            ),
            text_pretrained=bool(_cfg_get(cfg, "text_pretrained", True)),
            freeze_text_encoder=bool(_cfg_get(cfg, "freeze_text_encoder", True)),
            transformers_local_files_only=bool(
                _cfg_get(cfg, "transformers_local_files_only", False)
            ),
            hidden_size=int(_cfg_get(cfg, "hidden_size", 256)),
            recurrent_hidden_size=int(
                _cfg_get(cfg, "recurrent_hidden_size", 512)
            ),
            prev_action_embedding_dim=int(
                _cfg_get(cfg, "prev_action_embedding_dim", 32)
            ),
            dropout=float(_cfg_get(cfg, "dropout", 0.1)),
            image_size=int(_cfg_get(cfg, "image_size", 224)),
            simple_text_vocab_size=int(
                _cfg_get(cfg, "simple_text_vocab_size", 30522)
            ),
            max_instruction_length=int(
                _cfg_get(cfg, "max_instruction_length", 80)
            ),
            fusion_type=str(_cfg_get(cfg, "fusion_type", "gated")),
            cma_num_heads=int(_cfg_get(cfg, "cma_num_heads", 4)),
            use_depth=bool(_cfg_get(cfg, "use_depth", False)),
            separate_stop_head=bool(_cfg_get(cfg, "separate_stop_head", False)),
        )

    def _make_dataset(self, split: str, max_episodes: int):
        splits = [value.strip() for value in split.split(",") if value.strip()]
        if len(splits) > 1:
            datasets = [
                VLNOracleTrajectoryDataset(
                    self.config,
                    split=value,
                    max_episodes=max_episodes,
                    cache_dir=str(_cfg_get(self.vln_config, "cache_dir", "")),
                    refresh_cache=bool(
                        _cfg_get(self.vln_config, "refresh_cache", False)
                    ),
                )
                for value in splits
            ]
            return ConcatDataset(datasets)

        return VLNOracleTrajectoryDataset(
            self.config,
            split=split,
            max_episodes=max_episodes,
            cache_dir=str(_cfg_get(self.vln_config, "cache_dir", "")),
            refresh_cache=bool(_cfg_get(self.vln_config, "refresh_cache", False)),
        )

    def _make_loader(self, dataset, model: VLNPolicyNet, shuffle: bool):
        collator = VLNBatchCollator(
            model.tokenizer,
            max_instruction_length=int(
                _cfg_get(self.vln_config, "max_instruction_length", 80)
            ),
            stop_action_repeat=int(_cfg_get(self.vln_config, "stop_action_repeat", 1)),
        )
        return DataLoader(
            dataset,
            batch_size=int(_cfg_get(self.vln_config, "batch_size", 4)),
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collator,
        )

    def train(self) -> None:
        cfg = self.vln_config
        train_split = str(
            _cfg_get(cfg, "train_split", self.config.habitat.dataset.split)
        )
        val_split = str(_cfg_get(cfg, "val_split", "val_seen"))

        model = self._make_model().to(self.device)
        train_dataset = self._make_dataset(
            train_split, int(_cfg_get(cfg, "max_train_episodes", -1))
        )
        val_dataset = self._make_dataset(
            val_split, int(_cfg_get(cfg, "max_val_episodes", -1))
        )
        train_loader = self._make_loader(train_dataset, model, shuffle=True)
        val_loader = self._make_loader(val_dataset, model, shuffle=False)

        optimizer = torch.optim.AdamW(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=float(_cfg_get(cfg, "lr", 3e-5)),
            weight_decay=float(_cfg_get(cfg, "weight_decay", 0.01)),
        )

        logger.info(model)
        logger.info(
            "Training VLN policy on %s trajectories; validating on %s",
            len(train_dataset),
            len(val_dataset),
        )

        with TensorboardWriter(
            self.config.habitat_baselines.tensorboard_dir,
            flush_secs=self.flush_secs,
        ) as writer:
            max_epochs = int(_cfg_get(cfg, "max_epochs", 10))
            for epoch in range(1, max_epochs + 1):
                train_stats = self._run_epoch(
                    model,
                    train_loader,
                    optimizer=optimizer,
                    epoch=epoch,
                    max_epochs=max_epochs,
                )
                val_stats = self._run_epoch(model, val_loader, optimizer=None)
                rollout_stats = {}
                if bool(_cfg_get(cfg, "rollout_eval", True)):
                    rollout_stats = self._rollout_eval(model, val_split)

                logger.info(
                    "VLN epoch %d/%d | train loss %.4f acc %.3f | "
                    "val loss %.4f acc %.3f | stop acc train %.3f val %.3f",
                    epoch,
                    max_epochs,
                    train_stats["loss"],
                    train_stats["action_accuracy"],
                    val_stats["loss"],
                    val_stats["action_accuracy"],
                    train_stats.get("stop_accuracy", 0.0),
                    val_stats.get("stop_accuracy", 0.0),
                )
                if rollout_stats:
                    logger.info(
                        "VLN epoch %d/%d rollout | SR %.3f SPL %.3f "
                        "distance_to_goal %.3f",
                        epoch,
                        max_epochs,
                        rollout_stats.get("success", 0.0),
                        rollout_stats.get("spl", 0.0),
                        rollout_stats.get("distance_to_goal", 0.0),
                    )
                self._write_epoch_stats(writer, "train", train_stats, epoch)
                self._write_epoch_stats(writer, "val", val_stats, epoch)
                if rollout_stats:
                    self._write_epoch_stats(
                        writer, "train_rollout", rollout_stats, epoch
                    )

                interval = int(
                    self.config.habitat_baselines.checkpoint_interval
                )
                if interval <= 0:
                    interval = 1
                if epoch % interval == 0 or epoch == max_epochs:
                    self.save_checkpoint(
                        OrderedDict(
                            {
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "config": OmegaConf.to_container(
                                    self.config, resolve=True
                                ),
                                "action_names": ACTION_NAMES,
                            }
                        ),
                        f"epoch_{epoch}.ckpt",
                    )

    def _run_epoch(
        self,
        model: VLNPolicyNet,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int = 1,
        max_epochs: int = 1,
    ) -> Dict[str, float]:
        is_train = optimizer is not None
        model.train(is_train)

        total_loss = 0.0
        total_correct = 0.0
        total_stop_correct = 0.0
        total_actions = 0

        for batch in loader:
            batch = self._batch_to_device(batch)

            output, _ = model(
                batch["rgb"],
                batch["input_ids"],
                batch["attention_mask"],
                batch["prev_actions"],
                lengths=batch["lengths"],
                depth=batch.get("depth"),
            )
            loss, correct, stop_correct, count = self._masked_action_stats(
                model,
                output,
                batch["actions"],
                batch["action_mask"],
                weight=self._action_loss_weights(
                    output[0].device if isinstance(output, tuple) else output.device
                ),
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                max_grad_norm = float(
                    _cfg_get(self.vln_config, "max_grad_norm", 1.0)
                )
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                optimizer.step()

            total_loss += float(loss.item()) * count
            total_correct += correct
            total_stop_correct += stop_correct
            total_actions += count

        if total_actions == 0:
            return {"loss": 0.0, "action_accuracy": 0.0, "stop_accuracy": 0.0}
        return {
            "loss": total_loss / total_actions,
            "action_accuracy": total_correct / total_actions,
            "stop_accuracy": total_stop_correct / total_actions,
        }

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in batch.items():
            out[key] = value.to(self.device) if torch.is_tensor(value) else value
        return out

    def _masked_action_stats(
        self,
        model: VLNPolicyNet,
        output,
        actions: Tensor,
        action_mask: Tensor,
        weight: Optional[Tensor] = None,
    ) -> tuple:
        valid = action_mask.view(-1)
        if isinstance(output, tuple):
            action_logits, stop_logits = output
        else:
            action_logits, stop_logits = output, None
        flat_stop_logits = None

        flat_logits = action_logits.view(-1, action_logits.size(-1))[valid]
        flat_actions = actions.view(-1)[valid]
        loss = F.cross_entropy(flat_logits, flat_actions, weight=weight)
        if stop_logits is not None:
            flat_stop_logits = stop_logits.reshape(-1)[valid]
            stop_targets = (flat_actions == HabitatSimActions.stop).float()
            stop_loss_weight = float(
                _cfg_get(self.vln_config, "stop_head_loss_weight", 1.0)
            )
            loss = loss + stop_loss_weight * F.binary_cross_entropy_with_logits(
                flat_stop_logits, stop_targets
            )

        predictions, stop_prob = model.select_action(
            (flat_logits, flat_stop_logits)
            if flat_stop_logits is not None
            else flat_logits,
            stop_threshold=float(_cfg_get(self.vln_config, "stop_threshold", 0.5)),
        )
        correct = float((predictions == flat_actions).sum().item())
        if stop_prob is None:
            stop_correct = 0.0
        else:
            stop_predictions = (
                stop_prob >= float(_cfg_get(self.vln_config, "stop_threshold", 0.5))
            )
            stop_correct = float(
                (stop_predictions == (flat_actions == HabitatSimActions.stop))
                .sum()
                .item()
            )
        return loss, correct, stop_correct, int(flat_actions.numel())

    def _action_loss_weights(self, device: torch.device) -> Optional[Tensor]:
        weights = _cfg_get(self.vln_config, "action_loss_weights", None)
        if weights is None:
            stop_weight = float(_cfg_get(self.vln_config, "stop_loss_weight", 1.0))
            if stop_weight == 1.0:
                return None
            values = [1.0] * len(ACTION_NAMES)
            values[HabitatSimActions.stop] = stop_weight
        else:
            values = [float(value) for value in weights]
            if len(values) != len(ACTION_NAMES):
                raise ValueError(
                    "action_loss_weights must contain one weight per action: "
                    f"{ACTION_NAMES}"
                )
        return torch.tensor(values, dtype=torch.float32, device=device)

    @staticmethod
    def _write_epoch_stats(
        writer: TensorboardWriter,
        prefix: str,
        stats: Dict[str, float],
        epoch: int,
    ) -> None:
        for key, value in stats.items():
            writer.add_scalar(f"{prefix}/{key}", value, epoch)

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        model = self._make_model().to(self.device)
        if self.config.habitat_baselines.eval.should_load_ckpt:
            checkpoint = self.load_checkpoint(checkpoint_path)
            model.load_state_dict(checkpoint["state_dict"])

        val_split = str(_cfg_get(self.vln_config, "val_split", "val_seen"))
        val_dataset = self._make_dataset(
            val_split, int(_cfg_get(self.vln_config, "max_val_episodes", -1))
        )
        val_loader = self._make_loader(val_dataset, model, shuffle=False)
        val_stats = self._run_epoch(model, val_loader, optimizer=None)
        logger.info(
            "VLN eval checkpoint %s | val loss %.4f acc %.3f",
            checkpoint_path,
            val_stats["loss"],
            val_stats["action_accuracy"],
        )
        self._write_epoch_stats(
            writer, "eval_teacher_forced", val_stats, checkpoint_index
        )

        if bool(_cfg_get(self.vln_config, "rollout_eval", True)):
            rollout_stats = self._rollout_eval(model, val_split)
            logger.info("VLN rollout eval: %s", rollout_stats)
            self._write_epoch_stats(
                writer, "eval_rollout", rollout_stats, checkpoint_index
            )

    def _rollout_eval(self, model: VLNPolicyNet, split: str) -> Dict[str, float]:
        model.eval()
        habitat_config = copy.deepcopy(self.config.habitat)
        with read_write(habitat_config):
            habitat_config.dataset.split = split

        max_episodes = int(_cfg_get(self.vln_config, "max_rollout_episodes", 16))
        max_steps = int(
            _cfg_get(
                self.vln_config,
                "max_rollout_steps",
                self.config.habitat.environment.max_episode_steps,
            )
        )
        metric_totals: Dict[str, float] = defaultdict(float)
        action_totals: Dict[str, float] = defaultdict(float)
        evaluated = 0

        with habitat.Env(config=habitat_config) as env:
            if max_episodes <= 0:
                max_episodes = len(env.episodes)
            max_episodes = min(max_episodes, len(env.episodes))

            for _ in range(max_episodes):
                observations = env.reset()
                instruction = env.current_episode.instruction.instruction_text
                tokenized = model.tokenize([instruction], device=self.device)
                with torch.no_grad():
                    instruction_features = model.encode_instruction(
                        tokenized["input_ids"],
                        tokenized["attention_mask"],
                    )
                hidden_state = None
                prev_action = torch.tensor(
                    [START_PREV_ACTION], dtype=torch.long, device=self.device
                )

                for _step in range(max_steps):
                    rgb = self._obs_rgb_to_tensor(observations["rgb"])
                    with torch.no_grad():
                        depth = (
                            self._obs_depth_to_tensor(observations["depth"])
                            if bool(_cfg_get(self.vln_config, "use_depth", False))
                            else None
                        )
                        output, hidden_state = model.step(
                            rgb,
                            tokenized["input_ids"],
                            tokenized["attention_mask"],
                            prev_action,
                            hidden_state=hidden_state,
                            instruction_features=instruction_features,
                            depth=depth,
                        )
                    selected_action, _ = model.select_action(
                        output,
                        stop_threshold=float(
                            _cfg_get(self.vln_config, "stop_threshold", 0.5)
                        ),
                    )
                    action = int(selected_action.item())
                    if 0 <= action < len(ACTION_NAMES):
                        action_totals[f"action_{ACTION_NAMES[action]}"] += 1.0
                    observations = env.step(action)
                    prev_action.fill_(action + 1)
                    if env.episode_over or action == HabitatSimActions.stop:
                        break

                for key, value in env.get_metrics().items():
                    if isinstance(value, (int, float, np.floating)):
                        metric_totals[key] += float(value)
                evaluated += 1

        if evaluated == 0:
            return {}
        stats = {
            key: value / evaluated for key, value in sorted(metric_totals.items())
        }
        total_actions = sum(action_totals.values())
        if total_actions > 0:
            for key, value in sorted(action_totals.items()):
                stats[key] = value / total_actions
        return stats

    def _obs_rgb_to_tensor(self, rgb: np.ndarray) -> Tensor:
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        rgb = np.ascontiguousarray(rgb)
        return (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .to(self.device)
        )

    def _obs_depth_to_tensor(self, depth: np.ndarray) -> Tensor:
        if depth.ndim == 2:
            depth = depth[..., None]
        if depth.shape[-1] > 1:
            depth = depth[..., :1]
        depth = np.ascontiguousarray(depth.astype(np.float32))
        return (
            torch.from_numpy(depth)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .to(self.device)
        )

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        try:
            return torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            return torch.load(checkpoint_path, map_location="cpu")

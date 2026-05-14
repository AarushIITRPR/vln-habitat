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
from torch.utils.data import DataLoader

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
        )

    def _make_dataset(self, split: str, max_episodes: int):
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
                    model, train_loader, optimizer=optimizer
                )
                val_stats = self._run_epoch(model, val_loader, optimizer=None)

                logger.info(
                    "VLN epoch %d/%d | train loss %.4f acc %.3f | "
                    "val loss %.4f acc %.3f",
                    epoch,
                    max_epochs,
                    train_stats["loss"],
                    train_stats["action_accuracy"],
                    val_stats["loss"],
                    val_stats["action_accuracy"],
                )
                self._write_epoch_stats(writer, "train", train_stats, epoch)
                self._write_epoch_stats(writer, "val", val_stats, epoch)

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
    ) -> Dict[str, float]:
        is_train = optimizer is not None
        model.train(is_train)

        total_loss = 0.0
        total_correct = 0.0
        total_actions = 0

        for batch in loader:
            batch = self._batch_to_device(batch)
            logits, _ = model(
                batch["rgb"],
                batch["input_ids"],
                batch["attention_mask"],
                batch["prev_actions"],
                lengths=batch["lengths"],
            )
            loss, correct, count = self._masked_action_stats(
                logits, batch["actions"], batch["action_mask"]
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
            total_actions += count

        if total_actions == 0:
            return {"loss": 0.0, "action_accuracy": 0.0}
        return {
            "loss": total_loss / total_actions,
            "action_accuracy": total_correct / total_actions,
        }

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in batch.items():
            out[key] = value.to(self.device) if torch.is_tensor(value) else value
        return out

    @staticmethod
    def _masked_action_stats(
        logits: Tensor, actions: Tensor, action_mask: Tensor
    ) -> tuple:
        valid = action_mask.view(-1)
        flat_logits = logits.view(-1, logits.size(-1))[valid]
        flat_actions = actions.view(-1)[valid]
        loss = F.cross_entropy(flat_logits, flat_actions)
        predictions = flat_logits.argmax(dim=-1)
        correct = float((predictions == flat_actions).sum().item())
        return loss, correct, int(flat_actions.numel())

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
        evaluated = 0

        with habitat.Env(config=habitat_config) as env:
            if max_episodes <= 0:
                max_episodes = len(env.episodes)
            max_episodes = min(max_episodes, len(env.episodes))

            for _ in range(max_episodes):
                observations = env.reset()
                instruction = env.current_episode.instruction.instruction_text
                tokenized = model.tokenize([instruction], device=self.device)
                hidden_state = None
                prev_action = torch.tensor(
                    [START_PREV_ACTION], dtype=torch.long, device=self.device
                )

                for _step in range(max_steps):
                    rgb = self._obs_rgb_to_tensor(observations["rgb"])
                    with torch.no_grad():
                        logits, hidden_state = model.step(
                            rgb,
                            tokenized["input_ids"],
                            tokenized["attention_mask"],
                            prev_action,
                            hidden_state=hidden_state,
                        )
                    action = int(logits.argmax(dim=-1).item())
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
        return {
            key: value / evaluated for key, value in sorted(metric_totals.items())
        }

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

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        try:
            return torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            return torch.load(checkpoint_path, map_location="cpu")

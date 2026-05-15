#!/usr/bin/env python3

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
HAB_BASELINES = PARENT / "habitat-baselines" / "habitat_baselines" / "config"

sys.path.insert(0, str(PARENT / "habitat-baselines"))
sys.path.insert(0, str(PARENT / "habitat-lab"))

import habitat  # noqa: E402
from habitat.config import read_write  # noqa: E402
from habitat.config.default import patch_config  # noqa: E402
from habitat.config.default_structured_configs import register_hydra_plugin  # noqa: E402
from habitat.sims.habitat_simulator.actions import HabitatSimActions  # noqa: E402
from habitat_baselines.config.default_structured_configs import (  # noqa: E402
    HabitatBaselinesConfigPlugin,
)
from habitat_baselines.il.models.vln_policy import (  # noqa: E402
    ACTION_NAMES,
    START_PREV_ACTION,
)
from habitat_baselines.il.trainers.vln_trainer import VLNILTrainer  # noqa: E402


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _apply_checkpoint_options(cfg, checkpoint):
    saved = checkpoint.get("config", {})
    saved_vln = saved.get("habitat_baselines", {}).get("il", {}).get("vln", {})
    state_dict = checkpoint.get("state_dict", {})
    for key in (
        "fusion_type",
        "cma_num_heads",
        "use_depth",
        "separate_stop_head",
        "stop_threshold",
    ):
        if key in saved_vln:
            setattr(cfg.habitat_baselines.il.vln, key, saved_vln[key])
    if "use_depth" not in saved_vln:
        cfg.habitat_baselines.il.vln.use_depth = any(
            key.startswith("depth_encoder.") for key in state_dict
        )
    if "separate_stop_head" not in saved_vln:
        cfg.habitat_baselines.il.vln.separate_stop_head = any(
            key.startswith("stop_head.") for key in state_dict
        )


def _paraphrase(text: str) -> str:
    replacements = [
        (r"\bgo\b", "proceed"),
        (r"\bwalk\b", "move"),
        (r"\bhead\b", "continue"),
        (r"\bcontinue\b", "keep going"),
        (r"\bturn left\b", "make a left turn"),
        (r"\bturn right\b", "make a right turn"),
        (r"\bleft\b", "left-hand side"),
        (r"\bright\b", "right-hand side"),
        (r"\bstraight\b", "forward"),
        (r"\bhallway\b", "corridor"),
        (r"\broom\b", "area"),
        (r"\bdoorway\b", "door opening"),
        (r"\buntil\b", "up to the point where"),
        (r"\bthen\b", "after that"),
        (r"\bstop\b", "come to a stop"),
    ]
    out = text
    for pattern, value in replacements:
        out = re.sub(pattern, value, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def _obs_rgb_to_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    rgb = np.ascontiguousarray(rgb)
    return (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
        .to(device)
    )


def _obs_depth_to_tensor(depth: np.ndarray, device: torch.device) -> torch.Tensor:
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
        .to(device)
    )


def _balanced_episode_subset(episodes, count: int):
    by_scene = defaultdict(list)
    scene_order = []
    for episode in episodes:
        if episode.scene_id not in by_scene:
            scene_order.append(episode.scene_id)
        by_scene[episode.scene_id].append(episode)

    selected = []
    for scene_round in zip_longest(*(by_scene[scene] for scene in scene_order)):
        for episode in scene_round:
            if episode is None:
                continue
            selected.append(episode)
            if len(selected) >= count:
                return selected
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--instruction-mode", choices=["original", "paraphrase"], default="original")
    parser.add_argument(
        "--episode-selection",
        choices=["balanced", "sequential"],
        default="balanced",
        help="balanced spreads a small evaluation subset across scenes",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoint = _torch_load(args.checkpoint)
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    with initialize_config_dir(version_base=None, config_dir=str(HAB_BASELINES)):
        cfg = compose(config_name="vln/il_vln_cma.yaml")
    cfg = patch_config(cfg)
    with read_write(cfg):
        cfg.habitat.dataset.data_path = str(
            ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"
        )
        cfg.habitat.dataset.scenes_dir = str(ROOT / "data/scene_datasets/")
        cfg.habitat.dataset.split = args.split
        cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        cfg.habitat.simulator.concur_render = False
        cfg.habitat.task.measurements.success.success_distance = 3.0
        cfg.habitat.environment.iterator_options.group_by_scene = False
        cfg.habitat.environment.iterator_options.shuffle = False
        _apply_checkpoint_options(cfg, checkpoint)

    trainer = VLNILTrainer(cfg)
    model = trainer._make_model().to(trainer.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    totals = defaultdict(float)
    action_counts = Counter()
    per_episode = []
    with habitat.Env(config=cfg.habitat) as env:
        count = min(args.episodes, len(env.episodes)) if args.episodes > 0 else len(env.episodes)
        if args.episode_selection == "balanced" and args.episodes > 0:
            env.episodes = _balanced_episode_subset(env.episodes, count)
        elif args.episodes > 0:
            env.episodes = env.episodes[:count]
        for _ in range(count):
            observations = env.reset()
            raw_instruction = env.current_episode.instruction.instruction_text
            instruction = (
                _paraphrase(raw_instruction)
                if args.instruction_mode == "paraphrase"
                else raw_instruction
            )
            tokenized = model.tokenize([instruction], device=trainer.device)
            with torch.no_grad():
                instruction_features = model.encode_instruction(
                    tokenized["input_ids"], tokenized["attention_mask"]
                )
            hidden_state = None
            prev_action = torch.tensor(
                [START_PREV_ACTION], dtype=torch.long, device=trainer.device
            )
            steps = 0
            final_action = None
            for _step in range(args.max_steps):
                rgb = _obs_rgb_to_tensor(observations["rgb"], trainer.device)
                depth = (
                    _obs_depth_to_tensor(observations["depth"], trainer.device)
                    if bool(trainer.vln_config.get("use_depth", False))
                    else None
                )
                with torch.no_grad():
                    output, hidden_state = model.step(
                        rgb,
                        tokenized["input_ids"],
                        tokenized["attention_mask"],
                        prev_action,
                        hidden_state=hidden_state,
                        instruction_features=instruction_features,
                        depth=depth,
                    )
                selected, _ = model.select_action(
                    output,
                    stop_threshold=float(trainer.vln_config.get("stop_threshold", 0.5)),
                )
                action = int(selected.item())
                if 0 <= action < len(ACTION_NAMES):
                    action_counts[ACTION_NAMES[action]] += 1
                    final_action = ACTION_NAMES[action]
                else:
                    final_action = str(action)
                observations = env.step(action)
                prev_action.fill_(action + 1)
                steps += 1
                if env.episode_over or action == HabitatSimActions.stop:
                    break
            metrics = {
                k: float(v)
                for k, v in env.get_metrics().items()
                if isinstance(v, (int, float, np.floating))
            }
            for key, value in metrics.items():
                totals[key] += value
            per_episode.append(
                {
                    "episode_id": str(env.current_episode.episode_id),
                    "scene_id": env.current_episode.scene_id,
                    "steps": steps,
                    "final_action": final_action,
                    "instruction": raw_instruction,
                    "evaluated_instruction": instruction,
                    **metrics,
                }
            )

    evaluated = len(per_episode)
    summary = {key: value / evaluated for key, value in sorted(totals.items())}
    total_actions = sum(action_counts.values())
    if total_actions:
        summary["action_distribution"] = {
            key: value / total_actions for key, value in sorted(action_counts.items())
        }
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "instruction_mode": args.instruction_mode,
        "episodes": evaluated,
        "max_steps": args.max_steps,
        "summary": summary,
        "per_episode": per_episode,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

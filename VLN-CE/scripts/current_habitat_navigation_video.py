#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
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
from habitat_baselines.il.models.vln_policy import START_PREV_ACTION  # noqa: E402
from habitat_baselines.il.trainers.vln_trainer import VLNILTrainer  # noqa: E402


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _apply_checkpoint_vln_options(cfg, checkpoint_path):
    checkpoint = _torch_load(checkpoint_path)
    saved_config = checkpoint.get("config", {})
    saved_vln = (
        saved_config.get("habitat_baselines", {})
        .get("il", {})
        .get("vln", {})
    )
    for key in (
        "use_depth",
        "separate_stop_head",
        "stop_threshold",
        "stop_head_loss_weight",
        "fusion_type",
        "cma_num_heads",
    ):
        if key in saved_vln:
            setattr(cfg.habitat_baselines.il.vln, key, saved_vln[key])
    state_dict = checkpoint.get("state_dict", {})
    if "use_depth" not in saved_vln:
        cfg.habitat_baselines.il.vln.use_depth = any(
            key.startswith("depth_encoder.") for key in state_dict
        )
    if "separate_stop_head" not in saved_vln:
        cfg.habitat_baselines.il.vln.separate_stop_head = any(
            key.startswith("stop_head.") for key in state_dict
        )
    return checkpoint


def _position_xz(position):
    if position is None:
        return None
    arr = np.asarray(position, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    return np.array([arr[0], arr[2]], dtype=np.float32)


def _agent_position(env):
    try:
        return _position_xz(env.sim.get_agent_state().position)
    except Exception:
        return None


def _goal_position(episode):
    if not getattr(episode, "goals", None):
        return None
    return _position_xz(episode.goals[0].position)


def _reference_path(episode):
    points = []
    for point in getattr(episode, "reference_path", []) or []:
        xz = _position_xz(point)
        if xz is not None:
            points.append(xz)
    goal = _goal_position(episode)
    if goal is not None:
        points.append(goal)
    return points


def _project_points(points, width, height, pad=24):
    valid = [point for point in points if point is not None]
    if not valid:
        return {}

    stacked = np.stack(valid)
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    usable_w = max(width - 2 * pad, 1)
    usable_h = max(height - 2 * pad, 1)
    scale = min(usable_w / span[0], usable_h / span[1])
    center = (lo + hi) * 0.5

    def project(point):
        rel = (point - center) * scale
        x = int(round(width * 0.5 + rel[0]))
        y = int(round(height * 0.5 - rel[1]))
        return (
            int(np.clip(x, pad // 2, width - pad // 2)),
            int(np.clip(y, pad // 2, height - pad // 2)),
        )

    return {id(point): project(point) for point in valid}


def _draw_polyline(canvas, points, projected, color, thickness=2, dashed=False):
    pts = [projected.get(id(point)) for point in points if point is not None]
    pts = [point for point in pts if point is not None]
    if len(pts) < 2:
        return
    for idx in range(1, len(pts)):
        if dashed and idx % 2 == 0:
            continue
        cv2.line(canvas, pts[idx - 1], pts[idx], color, thickness, cv2.LINE_AA)


def _draw_trajectory_panel(
    height,
    start_pos,
    goal_pos,
    final_pos,
    reference_points,
    actual_points,
    upto_index,
):
    width = 256
    panel = np.full((height, width, 3), 246, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (215, 215, 215), 1)
    cv2.putText(
        panel,
        "path",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    shown_actual = actual_points[: max(1, upto_index + 1)]
    all_points = (
        [start_pos, goal_pos, final_pos]
        + list(reference_points)
        + list(shown_actual)
    )
    projected = _project_points(all_points, width, height, pad=34)

    _draw_polyline(panel, reference_points, projected, (175, 175, 175), 1, dashed=True)
    _draw_polyline(panel, shown_actual, projected, (40, 120, 220), 3)

    def marker(point, color, label, radius=5, filled=True):
        xy = projected.get(id(point)) if point is not None else None
        if xy is None:
            return
        thickness = -1 if filled else 2
        cv2.circle(panel, xy, radius, color, thickness, cv2.LINE_AA)
        cv2.putText(
            panel,
            label,
            (xy[0] + 7, xy[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    current_pos = shown_actual[-1] if shown_actual else start_pos
    marker(start_pos, (40, 160, 70), "start", radius=5)
    marker(goal_pos, (40, 60, 220), "goal", radius=6)
    marker(final_pos, (90, 45, 45), "final", radius=6, filled=False)
    marker(current_pos, (220, 120, 40), "now", radius=5)

    cv2.putText(
        panel,
        "gray: ref  blue: agent",
        (10, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    return panel


def _frame(
    rgb,
    instruction,
    metrics,
    action_name,
    start_pos=None,
    goal_pos=None,
    final_pos=None,
    reference_points=None,
    actual_points=None,
    frame_index=0,
    show_path=True,
):
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    frame = np.ascontiguousarray(rgb)
    panel_h = 96
    panel = np.full((panel_h, frame.shape[1], 3), 255, dtype=np.uint8)
    text = f"{action_name} | dist={metrics.get('distance_to_goal', 0):.2f} | SR={metrics.get('success', 0):.0f} SPL={metrics.get('spl', 0):.2f}"
    cv2.putText(panel, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    words = instruction.split()
    line = ""
    lines = []
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > 82:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for idx, line in enumerate(lines[:2]):
        cv2.putText(panel, line, (8, 56 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    composed = np.concatenate([frame, panel], axis=0)
    if not show_path:
        return composed

    path_panel = _draw_trajectory_panel(
        composed.shape[0],
        start_pos,
        goal_pos,
        final_pos,
        reference_points or [],
        actual_points or [],
        frame_index,
    )
    return np.concatenate([composed, path_panel], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "data/checkpoints/current_cma_da/epoch_1.ckpt"))
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--episode-ids",
        default="",
        help="Comma-separated episode ids to render from the selected split.",
    )
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--out-dir", default=str(ROOT / "data/videos/cma_da_navigation"))
    parser.add_argument(
        "--no-path-overlay",
        action="store_true",
        help="Disable the start/current/goal/final trajectory panel.",
    )
    args = parser.parse_args()

    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    with initialize_config_dir(version_base=None, config_dir=str(HAB_BASELINES)):
        cfg = compose(config_name="vln/il_vln_cma.yaml")

    cfg = patch_config(cfg)
    with read_write(cfg):
        cfg.habitat.dataset.data_path = str(
            ROOT
            / "data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"
        )
        cfg.habitat.dataset.scenes_dir = str(ROOT / "data/scene_datasets/")
        cfg.habitat.dataset.split = args.split
        cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        cfg.habitat.simulator.concur_render = False
        cfg.habitat.task.measurements.success.success_distance = 3.0
        cfg.habitat_baselines.il.vln.fusion_type = "cma"
        checkpoint = _apply_checkpoint_vln_options(cfg, args.checkpoint)

    trainer = VLNILTrainer(cfg)
    model = trainer._make_model().to(trainer.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    action_names = ["stop", "move_forward", "turn_left", "turn_right"]

    with habitat.Env(config=cfg) as env:
        selected_ids = {
            value.strip()
            for value in args.episode_ids.split(",")
            if value.strip()
        }
        total = len(env.episodes) if selected_ids else min(args.episodes, len(env.episodes))
        rendered = 0
        for _ep_idx in range(total):
            observations = env.reset()
            episode = env.current_episode
            instruction = episode.instruction.instruction_text
            episode_id = episode.episode_id
            if selected_ids and str(episode_id) not in selected_ids:
                continue
            start_pos = _agent_position(env)
            goal_pos = _goal_position(episode)
            reference_points = _reference_path(episode)
            tokenized = model.tokenize([instruction], device=trainer.device)
            with torch.no_grad():
                instruction_features = model.encode_instruction(
                    tokenized["input_ids"], tokenized["attention_mask"]
                )
            hidden_state = None
            prev_action = torch.tensor(
                [START_PREV_ACTION], dtype=torch.long, device=trainer.device
            )
            rollout = []
            positions = [start_pos] if start_pos is not None else []

            for _step in range(args.max_steps):
                rgb = trainer._obs_rgb_to_tensor(observations["rgb"])
                with torch.no_grad():
                    depth = (
                        trainer._obs_depth_to_tensor(observations["depth"])
                        if bool(trainer.vln_config.get("use_depth", False))
                        else None
                    )
                    logits, hidden_state = model.step(
                        rgb,
                        tokenized["input_ids"],
                        tokenized["attention_mask"],
                        prev_action,
                        hidden_state=hidden_state,
                        instruction_features=instruction_features,
                        depth=depth,
                    )
                selected_action, _ = model.select_action(
                    logits,
                    stop_threshold=float(trainer.vln_config.get("stop_threshold", 0.5)),
                )
                action = int(selected_action.item())
                observations = env.step(action)
                prev_action.fill_(action + 1)
                position = _agent_position(env)
                if position is not None:
                    positions.append(position)
                rollout.append(
                    {
                        "rgb": observations["rgb"],
                        "metrics": env.get_metrics(),
                        "action_name": action_names[action],
                        "position": position,
                    }
                )
                if env.episode_over or action == HabitatSimActions.stop:
                    break
            final_pos = positions[-1] if positions else None
            frames = [
                _frame(
                    step["rgb"],
                    instruction,
                    step["metrics"],
                    step["action_name"],
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    final_pos=final_pos,
                    reference_points=reference_points,
                    actual_points=positions,
                    frame_index=min(idx + 1, len(positions) - 1),
                    show_path=not args.no_path_overlay,
                )
                for idx, step in enumerate(rollout)
            ]

            video_path = out_dir / f"navigation_episode_{episode_id}.mp4"
            imageio.mimsave(
                video_path,
                frames,
                fps=10,
                codec="libx264",
                quality=8,
                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            print(video_path)
            rendered += 1
            if selected_ids and rendered >= len(selected_ids):
                break


if __name__ == "__main__":
    main()

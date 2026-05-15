#!/usr/bin/env python3

import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
HAB_BASELINES = PARENT / "habitat-baselines" / "habitat_baselines" / "config"

sys.path.insert(0, str(PARENT / "habitat-baselines"))
sys.path.insert(0, str(PARENT / "habitat-lab"))

from habitat.config import read_write  # noqa: E402
from habitat.config.default import patch_config  # noqa: E402
from habitat.config.default_structured_configs import register_hydra_plugin  # noqa: E402
from habitat_baselines.config.default_structured_configs import (  # noqa: E402
    HabitatBaselinesConfigPlugin,
)
from habitat_baselines.run import execute_exp  # noqa: E402


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_action_weights(name: str):
    raw = os.environ.get(name, "1.0,1.0,1.0,1.0").strip()
    if raw.lower() in {"", "none", "null"}:
        return None
    values = [float(part.strip()) for part in raw.split(",")]
    if len(values) != 4:
        raise ValueError(f"{name} must contain four comma-separated values")
    return values


def main() -> None:
    run_name = os.environ.get("RUN_NAME", "current_cma_da")

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
        cfg.habitat.dataset.split = "train"
        cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        cfg.habitat.simulator.concur_render = False
        cfg.habitat.task.measurements.success.success_distance = 3.0

        cfg.habitat_baselines.checkpoint_folder = str(
            ROOT / f"data/checkpoints/{run_name}"
        )
        cfg.habitat_baselines.eval_ckpt_path_dir = str(
            ROOT / f"data/checkpoints/{run_name}"
        )
        cfg.habitat_baselines.tensorboard_dir = str(
            ROOT / f"data/tensorboard_dirs/{run_name}"
        )
        cfg.habitat_baselines.il.results_dir = str(
            ROOT / f"data/checkpoints/{run_name}/evals/{{split}}"
        )
        cfg.habitat_baselines.il.output_log_dir = str(
            ROOT / f"data/logs/{run_name}"
        )

        vln = cfg.habitat_baselines.il.vln
        vln.fusion_type = os.environ.get("FUSION_TYPE", "cma")
        vln.train_split = os.environ.get("TRAIN_SPLIT", "train")
        vln.val_split = os.environ.get("VAL_SPLIT", "val_seen")
        vln.max_epochs = _env_int("MAX_EPOCHS", 5)
        vln.max_train_episodes = _env_int("MAX_TRAIN_EPISODES", 32)
        vln.max_val_episodes = _env_int("MAX_VAL_EPISODES", 8)
        vln.max_rollout_episodes = _env_int("MAX_ROLLOUT_EPISODES", 8)
        vln.max_trajectory_steps = _env_int("MAX_TRAJECTORY_STEPS", 160)
        vln.max_rollout_steps = _env_int("MAX_ROLLOUT_STEPS", 160)
        vln.batch_size = _env_int("BATCH_SIZE", 4)
        vln.lr = _env_float("LR", 1e-4)
        vln.action_loss_weights = _env_action_weights("ACTION_LOSS_WEIGHTS")
        vln.stop_action_repeat = _env_int("STOP_ACTION_REPEAT", 4)
        vln.use_depth = os.environ.get("USE_DEPTH", "1") != "0"
        vln.separate_stop_head = os.environ.get("SEPARATE_STOP_HEAD", "1") != "0"
        vln.stop_threshold = _env_float("STOP_THRESHOLD", 0.55)
        vln.stop_head_loss_weight = _env_float("STOP_HEAD_LOSS_WEIGHT", 1.0)
        vln.cache_dir = str(ROOT / f"data/trajectories_dirs/{run_name}/cache")
        vln.rollout_eval = os.environ.get("ROLLOUT_EVAL", "1") != "0"

    execute_exp(cfg, "train")


if __name__ == "__main__":
    main()

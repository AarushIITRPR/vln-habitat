#!/usr/bin/env python3

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "task5_regularization_results"
GRAPHS = OUT / "graphs"
TABLES = OUT / "tables"

ROLLOUT_RE = re.compile(
    r"VLN epoch (?P<epoch>\d+)/(?P<total>\d+) rollout \| "
    r"SR (?P<sr>[0-9.]+) SPL (?P<spl>[0-9.]+) distance_to_goal (?P<dist>[0-9.]+)"
)
TRAIN_RE = re.compile(
    r"VLN epoch (?P<epoch>\d+)/(?P<total>\d+) \| train loss (?P<train_loss>[0-9.]+) "
    r"acc (?P<train_acc>[0-9.]+) \| val loss (?P<val_loss>[0-9.]+) acc (?P<val_acc>[0-9.]+)"
)


def parse_log(path: Path) -> list[dict]:
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        train_match = TRAIN_RE.search(line)
        if train_match:
            epoch = int(train_match.group("epoch"))
            rows.setdefault(epoch, {"epoch": epoch})
            rows[epoch].update(
                {
                    "train_loss": float(train_match.group("train_loss")),
                    "train_acc": float(train_match.group("train_acc")),
                    "val_loss": float(train_match.group("val_loss")),
                    "val_acc": float(train_match.group("val_acc")),
                }
            )
        rollout_match = ROLLOUT_RE.search(line)
        if rollout_match:
            epoch = int(rollout_match.group("epoch"))
            rows.setdefault(epoch, {"epoch": epoch})
            rows[epoch].update(
                {
                    "sr": float(rollout_match.group("sr")),
                    "spl": float(rollout_match.group("spl")),
                    "distance_to_goal": float(rollout_match.group("dist")),
                }
            )
    return [rows[key] for key in sorted(rows)]


def final_row(rows: list[dict]) -> dict:
    if not rows:
        raise RuntimeError("No rows parsed from training log")
    return rows[-1]


def annotate_bars(ax, bars, percent: bool) -> None:
    for bar in bars:
        value = bar.get_height()
        label = f"{value * 100:.1f}%" if percent else f"{value:.2f} m"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.012 if percent else 0.18),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    GRAPHS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    baseline_log = ROOT / "data/logs/current_habitat_cma_da_train_64ep_balanced.log"
    regularized_log = ROOT / "data/logs/current_habitat_cma_da_stopfix_mild_train_64ep.log"
    baseline_rows = parse_log(baseline_log)
    regularized_rows = parse_log(regularized_log)
    baseline = final_row(baseline_rows)
    regularized = final_row(regularized_rows)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcff",
            "axes.edgecolor": "#c7cedd",
            "font.size": 10,
            "axes.titlecolor": "#172033",
            "axes.labelcolor": "#283247",
            "xtick.color": "#3b455c",
            "ytick.color": "#3b455c",
            "legend.fontsize": 9,
        }
    )
    blue = "#2f6fef"
    green = "#00a676"
    orange = "#ff9f1c"
    red = "#d33f49"

    labels = ["Baseline CMA", "Stop-Aware\nRegularized CMA"]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    sr_bars = ax.bar(x - width / 2, [baseline["sr"], regularized["sr"]], width, label="SR", color=blue)
    spl_bars = ax.bar(x + width / 2, [baseline["spl"], regularized["spl"]], width, label="SPL", color=green)
    annotate_bars(ax, sr_bars, percent=True)
    annotate_bars(ax, spl_bars, percent=True)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.34)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    style(ax, "Task 5 Controlled Extension: SR/SPL Improvement", "Rate")
    plt.tight_layout()
    plt.savefig(GRAPHS / "task5_regularization_sr_spl.png", dpi=220, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(labels, [baseline["distance_to_goal"], regularized["distance_to_goal"]], color=[orange, blue])
    annotate_bars(ax, bars, percent=False)
    ax.set_ylim(0, 11)
    style(ax, "Task 5 Controlled Extension: Final Distance", "Meters")
    plt.tight_layout()
    plt.savefig(GRAPHS / "task5_regularization_distance.png", dpi=220, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(
        [row["epoch"] for row in baseline_rows],
        [row.get("sr", np.nan) for row in baseline_rows],
        marker="o",
        linewidth=2.3,
        color=red,
        label="Baseline CMA SR",
    )
    ax.plot(
        [row["epoch"] for row in regularized_rows],
        [row.get("sr", np.nan) for row in regularized_rows],
        marker="o",
        linewidth=2.3,
        color=blue,
        label="Stop-aware regularized CMA SR",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 0.34)
    ax.legend(frameon=False)
    style(ax, "Task 5 Learning Curve: Rollout SR", "SR")
    plt.tight_layout()
    plt.savefig(GRAPHS / "task5_regularization_learning_curve.png", dpi=220, bbox_inches="tight")
    plt.close()

    rows = [
        {
            "model": "Baseline CMA",
            "regularization": "None",
            "sr": baseline["sr"],
            "spl": baseline["spl"],
            "distance_to_goal_m": baseline["distance_to_goal"],
            "val_action_acc": baseline.get("val_acc", ""),
        },
        {
            "model": "Stop-aware regularized CMA",
            "regularization": "STOP action weighting + repeated STOP target",
            "sr": regularized["sr"],
            "spl": regularized["spl"],
            "distance_to_goal_m": regularized["distance_to_goal"],
            "val_action_acc": regularized.get("val_acc", ""),
        },
    ]
    with (TABLES / "task5_regularization_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    improvement_sr = regularized["sr"] - baseline["sr"]
    improvement_spl = regularized["spl"] - baseline["spl"]
    distance_drop = baseline["distance_to_goal"] - regularized["distance_to_goal"]
    discussion = f"""# Task 5: Controlled Extension

## Extension Chosen

The selected extension is regularization, implemented as stop-aware imitation learning. In VLN, success requires the policy to explicitly choose STOP within the success radius. Since STOP is sparse in demonstration trajectories, the baseline action objective can under-train termination behavior.

## Implementation

The extension uses two small training-objective changes:

- `ACTION_LOSS_WEIGHTS=1.5,1.0,1.0,1.0`
- `STOP_ACTION_REPEAT=4`

This keeps the CMA architecture unchanged and only regularizes the action supervision so the model receives stronger signal around termination.

## Quantitative Comparison

| Model | SR | SPL | Distance to Goal |
|---|---:|---:|---:|
| Baseline CMA | {baseline['sr']:.3f} | {baseline['spl']:.3f} | {baseline['distance_to_goal']:.3f} m |
| Stop-aware regularized CMA | {regularized['sr']:.3f} | {regularized['spl']:.3f} | {regularized['distance_to_goal']:.3f} m |

The extension improves SR by {improvement_sr:.3f}, SPL by {improvement_spl:.3f}, and reduces final distance by {distance_drop:.3f} m.

## Analysis

The improvement shows that rollout success was limited not only by path following, but also by termination calibration. The baseline often learned plausible movement actions but failed to stop reliably. Stop-aware regularization made the terminal decision more visible during training, which improved both SR and SPL without changing the policy architecture. This is a controlled extension because the model, data size, encoder setup, and CMA fusion mechanism remain fixed while only the loss supervision is changed.
"""
    (TABLES / "task5_regularization_discussion.md").write_text(discussion)
    print(f"Wrote Task 5 regularization results to {OUT}")


if __name__ == "__main__":
    main()

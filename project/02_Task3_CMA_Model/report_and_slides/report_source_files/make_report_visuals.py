#!/usr/bin/env python3

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs/vln_acm_overleaf/figures"
LOCAL_DOCS = ROOT / "docs"
VLN = ROOT / "VLN-CE"

EPOCH_RE = re.compile(
    r"VLN epoch (?P<epoch>\d+)/(?P<total>\d+) \| train loss (?P<train_loss>[0-9.]+) "
    r"acc (?P<train_acc>[0-9.]+) \| val loss (?P<val_loss>[0-9.]+) acc (?P<val_acc>[0-9.]+)"
)
ROLLOUT_RE = re.compile(
    r"VLN epoch (?P<epoch>\d+)/(?P<total>\d+) rollout \| SR (?P<sr>[0-9.]+) "
    r"SPL (?P<spl>[0-9.]+) distance_to_goal (?P<dist>[0-9.]+)"
)


def parse_log(path: Path) -> list[dict]:
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        match = EPOCH_RE.search(line)
        if match:
            epoch = int(match.group("epoch"))
            rows.setdefault(epoch, {"epoch": epoch})
            rows[epoch].update(
                train_acc=float(match.group("train_acc")),
                val_acc=float(match.group("val_acc")),
                train_loss=float(match.group("train_loss")),
                val_loss=float(match.group("val_loss")),
            )
        match = ROLLOUT_RE.search(line)
        if match:
            epoch = int(match.group("epoch"))
            rows.setdefault(epoch, {"epoch": epoch})
            rows[epoch].update(
                sr=float(match.group("sr")),
                spl=float(match.group("spl")),
                distance_to_goal=float(match.group("dist")),
            )
    return [rows[key] for key in sorted(rows)]


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcff",
            "axes.edgecolor": "#c7cedd",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": "#152033",
            "axes.labelcolor": "#283247",
            "xtick.color": "#3b455c",
            "ytick.color": "#3b455c",
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )


def polish(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, pad=10, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_percent(ax, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def annotate_value(ax, bars, suffix: str = "") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.2f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save(fig, name: str) -> None:
    for out_dir in (FIG_DIR, LOCAL_DOCS):
        fig.savefig(out_dir / name, dpi=260, bbox_inches="tight")
    plt.close(fig)


def learning_curve() -> None:
    rows = parse_log(VLN / "data/logs/current_habitat_cma_da_stopfix_mild_train_64ep.log")
    epochs = [row["epoch"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(epochs, [row["train_acc"] for row in rows], marker="o", linewidth=2.2, color="#2f6fef", label="Train action accuracy")
    ax.plot(epochs, [row["val_acc"] for row in rows], marker="o", linewidth=2.2, color="#00a676", label="Validation action accuracy")
    ax.plot(epochs, [row["sr"] for row in rows], marker="o", linewidth=2.2, color="#d33f49", label="Rollout SR")
    ax.plot(epochs, [row["spl"] for row in rows], marker="o", linewidth=2.2, color="#ff9f1c", label="Rollout SPL")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 0.96)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#d8deea",
        framealpha=0.92,
        ncols=2,
        loc="lower right",
    )
    polish(ax, "Learning Curve for Stop-Aware CMA", "Rate")
    fig.tight_layout()
    save(fig, "09_learning_curve_accuracy_sr.png")


def fusion_ablation() -> None:
    cma = parse_log(VLN / "task45_results/raw/train_task45_cma_fusion_ablation_16ep.log")
    gated = parse_log(VLN / "task45_results/raw/train_task45_gated_fusion_ablation_16ep.log")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot([row["epoch"] for row in gated], [row["distance_to_goal"] for row in gated], marker="o", linewidth=2.4, color="#d33f49", label="Gated fusion")
    ax.plot([row["epoch"] for row in cma], [row["distance_to_goal"] for row in cma], marker="o", linewidth=2.4, color="#2f6fef", label="CMA cross-attention")
    ax.set_xlabel("Epoch")
    ax.set_ylim(4.5, 13.5)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#d8deea",
        framealpha=0.92,
        loc="upper right",
    )
    polish(ax, "Fusion Ablation: Final Distance to Goal", "Distance (m)")
    fig.tight_layout()
    save(fig, "06_ablation_fusion_distance_curve.png")


def task5_regularization() -> None:
    labels = ["Baseline\nCMA", "Stop-aware\nCMA"]
    sr = [0.125, 0.25]
    spl = [0.086, 0.25]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sr_bars = ax.bar(x - width / 2, sr, width, color="#2f6fef", label="SR")
    spl_bars = ax.bar(x + width / 2, spl, width, color="#00a676", label="SPL")
    annotate_percent(ax, sr_bars)
    annotate_percent(ax, spl_bars)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.40)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#d8deea",
        framealpha=0.92,
        loc="upper left",
    )
    polish(ax, "Task 5: Stop-Aware Regularization", "Rate")
    fig.tight_layout()
    save(fig, "task5_regularization_sr_spl.png")


def results_overview() -> None:
    labels = ["Baseline\nCMA", "Stop-aware\nCMA", "Controlled\nvalidation", "Paraphrase\ntest"]
    sr = [0.125, 0.25, 0.375, 0.0625]
    spl = [0.086, 0.25, 0.375, 0.0625]
    distance_labels = ["Reduced\ndata", "Gated\nfusion", "CMA\nfusion", "Stop-aware\nCMA"]
    distance = [12.915, 12.106, 5.919, 7.198]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={"width_ratios": [1.2, 1.0]})
    x = np.arange(len(labels))
    width = 0.36
    bars1 = ax1.bar(x - width / 2, sr, width, color="#2f6fef", label="SR")
    bars2 = ax1.bar(x + width / 2, spl, width, color="#00a676", label="SPL")
    annotate_percent(ax1, bars1)
    annotate_percent(ax1, bars2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, 0.54)
    ax1.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#d8deea",
        framealpha=0.92,
        loc="upper left",
    )
    polish(ax1, "Success and Efficiency", "Rate")

    bars = ax2.bar(distance_labels, distance, color=["#ff9f1c", "#d33f49", "#2f6fef", "#00a676"])
    annotate_value(ax2, bars, " m")
    ax2.set_ylim(0, 15)
    polish(ax2, "Distance Diagnostics", "Distance (m)")
    ax2.tick_params(axis="x", labelsize=8)
    fig.suptitle("Summary of Reported VLN Results", fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "report_results_overview.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    learning_curve()
    fusion_ablation()
    task5_regularization()
    results_overview()
    print(f"Wrote report visuals to {FIG_DIR}")


if __name__ == "__main__":
    main()

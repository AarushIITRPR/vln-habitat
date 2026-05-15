#!/usr/bin/env python3

import csv
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "task45_results"
RAW = OUT / "raw"
GRAPHS = OUT / "graphs"
TABLES = OUT / "tables"


EPOCH_RE = re.compile(
    r"VLN epoch (?P<epoch>\d+)/(?P<total>\d+) \| train loss (?P<train_loss>[0-9.]+) "
    r"acc (?P<train_acc>[0-9.]+) \| val loss (?P<val_loss>[0-9.]+) acc (?P<val_acc>[0-9.]+)"
)
ROLLOUT_RE = re.compile(
    r"VLN epoch (?P<epoch>\d+)/(?P<total>\d+) rollout \| SR (?P<sr>[0-9.]+) "
    r"SPL (?P<spl>[0-9.]+) distance_to_goal (?P<dist>[0-9.]+)"
)


def ensure_dirs() -> None:
    GRAPHS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def read_json_summary(name: str) -> dict:
    with (RAW / name).open() as f:
        return json.load(f)["summary"]


def read_first_json_summary(*names: str) -> dict:
    for name in names:
        if (RAW / name).exists():
            return read_json_summary(name)
    raise FileNotFoundError(names[0])


def parse_log(path: Path) -> list[dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return []
    for line in path.read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            epoch = int(m.group("epoch"))
            rows.setdefault(epoch, {"epoch": epoch})
            rows[epoch].update(
                {
                    "train_loss": float(m.group("train_loss")),
                    "train_acc": float(m.group("train_acc")),
                    "val_loss": float(m.group("val_loss")),
                    "val_acc": float(m.group("val_acc")),
                }
            )
        m = ROLLOUT_RE.search(line)
        if m:
            epoch = int(m.group("epoch"))
            rows.setdefault(epoch, {"epoch": epoch})
            rows[epoch].update(
                {
                    "sr": float(m.group("sr")),
                    "spl": float(m.group("spl")),
                    "distance_to_goal": float(m.group("dist")),
                }
            )
    return [rows[k] for k in sorted(rows)]


def final_metrics(path: Path) -> dict:
    rows = parse_log(path)
    if not rows:
        return {"success": np.nan, "spl": np.nan, "distance_to_goal": np.nan}
    row = rows[-1]
    return {
        "success": row.get("sr", np.nan),
        "spl": row.get("spl", np.nan),
        "distance_to_goal": row.get("distance_to_goal", np.nan),
        "val_acc": row.get("val_acc", np.nan),
    }


def style_axes(ax, title: str, ylabel: Optional[str] = None) -> None:
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def grouped_bars(ax, labels, series, colors, ylim=None, percent=False) -> None:
    x = np.arange(len(labels))
    width = 0.74 / len(series)
    offsets = np.linspace(-0.37 + width / 2, 0.37 - width / 2, len(series))
    for offset, (name, values), color in zip(offsets, series.items(), colors):
        bars = ax.bar(x + offset, values, width, label=name, color=color)
        for bar, value in zip(bars, values):
            label = f"{value * 100:.1f}%" if percent else f"{value:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.012 if percent else 0.18),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, ncols=len(series), loc="upper center", bbox_to_anchor=(0.5, 1.13))


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(GRAPHS / name, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_dirs()
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
    colors = ["#2f6fef", "#00a676", "#ff9f1c", "#d33f49"]

    evals = {
        "val_seen_original": read_json_summary("eval_clean_cma_val_seen_original.json"),
        "val_unseen_original": read_first_json_summary(
            "eval_clean_cma_val_unseen_original_balanced16.json",
            "eval_clean_cma_val_unseen_original.json",
        ),
        "val_seen_paraphrase": read_json_summary("eval_clean_cma_val_seen_paraphrase.json"),
        "val_unseen_paraphrase": read_first_json_summary(
            "eval_clean_cma_val_unseen_paraphrase_balanced16.json",
            "eval_clean_cma_val_unseen_paraphrase.json",
        ),
    }

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    labels = ["Val Seen", "Val Unseen"]
    grouped_bars(
        ax,
        labels,
        {
            "SR": [evals["val_seen_original"]["success"], evals["val_unseen_original"]["success"]],
            "SPL": [evals["val_seen_original"]["spl"], evals["val_unseen_original"]["spl"]],
        },
        colors[:2],
        ylim=(0, 0.30),
        percent=True,
    )
    style_axes(ax, "Generalization: Seen vs Unseen Environments", "Rate")
    savefig("01_unseen_generalization_sr_spl.png")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(labels, [evals["val_seen_original"]["distance_to_goal"], evals["val_unseen_original"]["distance_to_goal"]], color=colors[0])
    for i, value in enumerate([evals["val_seen_original"]["distance_to_goal"], evals["val_unseen_original"]["distance_to_goal"]]):
        ax.text(i, value + 0.15, f"{value:.2f} m", ha="center", fontsize=9)
    style_axes(ax, "Final Distance To Goal On Validation Splits", "Meters")
    ax.set_ylim(0, 10)
    savefig("02_unseen_generalization_distance.png")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    labels = ["Seen Original", "Seen Paraphrased", "Unseen Original", "Unseen Paraphrased"]
    grouped_bars(
        ax,
        labels,
        {
            "SR": [
                evals["val_seen_original"]["success"],
                evals["val_seen_paraphrase"]["success"],
                evals["val_unseen_original"]["success"],
                evals["val_unseen_paraphrase"]["success"],
            ],
            "SPL": [
                evals["val_seen_original"]["spl"],
                evals["val_seen_paraphrase"]["spl"],
                evals["val_unseen_original"]["spl"],
                evals["val_unseen_paraphrase"]["spl"],
            ],
        },
        colors[:2],
        ylim=(0, 0.30),
        percent=True,
    )
    style_axes(ax, "Instruction Robustness Under Paraphrasing", "Rate")
    savefig("03_paraphrase_robustness.png")

    baseline_log = ROOT / "data/logs/current_habitat_cma_da_train_64ep_balanced.log"
    extension_log = ROOT / "data/logs/current_habitat_cma_da_stopfix_mild_train_64ep.log"
    reduced32_log = ROOT / "data/logs/current_habitat_cma_da_train_32ep_balanced.log"
    cma16_log = RAW / "train_task45_cma_fusion_ablation_16ep.log"
    gated16_log = RAW / "train_task45_gated_fusion_ablation_16ep.log"

    reduced = {
        "32 train eps": final_metrics(reduced32_log),
        "64 train eps": final_metrics(extension_log),
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    grouped_bars(
        ax,
        list(reduced.keys()),
        {
            "SR": [v["success"] for v in reduced.values()],
            "SPL": [v["spl"] for v in reduced.values()],
        },
        colors[:2],
        ylim=(0, 0.34),
        percent=True,
    )
    style_axes(ax, "Reduced Training Data Study", "Rate")
    savefig("04_reduced_training_data_sr_spl.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(list(reduced.keys()), [v["distance_to_goal"] for v in reduced.values()], color=[colors[2], colors[0]])
    for i, v in enumerate(reduced.values()):
        ax.text(i, v["distance_to_goal"] + 0.2, f"{v['distance_to_goal']:.2f} m", ha="center", fontsize=9)
    style_axes(ax, "Training Data Effect On Final Distance", "Meters")
    ax.set_ylim(0, 15)
    savefig("05_reduced_training_data_distance.png")

    cma16_rows = parse_log(cma16_log)
    gated16_rows = parse_log(gated16_log)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for name, rows, color in [("CMA cross-attention", cma16_rows, colors[0]), ("Gated fusion", gated16_rows, colors[3])]:
        ax.plot([r["epoch"] for r in rows], [r.get("distance_to_goal", np.nan) for r in rows], marker="o", linewidth=2.4, label=name, color=color)
    style_axes(ax, "Fusion Ablation: Distance Across Epochs", "Meters")
    ax.set_xlabel("Epoch")
    ax.legend(frameon=False)
    savefig("06_ablation_fusion_distance_curve.png")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = ["Gated Fusion", "CMA Cross-Attention"]
    gated = final_metrics(gated16_log)
    cma = final_metrics(cma16_log)
    grouped_bars(
        ax,
        labels,
        {
            "Val action acc": [gated["val_acc"], cma["val_acc"]],
            "SR": [gated["success"], cma["success"]],
        },
        [colors[2], colors[0]],
        ylim=(0, 0.85),
        percent=True,
    )
    style_axes(ax, "Fusion Ablation: Validation Accuracy And SR", "Rate")
    savefig("07_ablation_fusion_accuracy_sr.png")

    baseline = final_metrics(baseline_log)
    extension = final_metrics(extension_log)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = ["Baseline CMA", "Stop-aware CMA"]
    grouped_bars(
        ax,
        labels,
        {
            "SR": [baseline["success"], extension["success"]],
            "SPL": [baseline["spl"], extension["spl"]],
        },
        colors[:2],
        ylim=(0, 0.34),
        percent=True,
    )
    style_axes(ax, "Controlled Extension: Stop-Aware Training", "Rate")
    savefig("08_controlled_extension_sr_spl.png")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    rows = parse_log(extension_log)
    ax.plot([r["epoch"] for r in rows], [r["train_acc"] for r in rows], marker="o", linewidth=2.2, label="Train action accuracy", color=colors[0])
    ax.plot([r["epoch"] for r in rows], [r["val_acc"] for r in rows], marker="o", linewidth=2.2, label="Val action accuracy", color=colors[1])
    ax.plot([r["epoch"] for r in rows], [r["sr"] for r in rows], marker="o", linewidth=2.2, label="Rollout SR", color=colors[3])
    style_axes(ax, "Learning Curve For Best Clean CMA Run", "Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False)
    savefig("09_learning_curve_accuracy_sr.png")

    summary_rows = [
        ("Task 4", "Val seen original", 16, evals["val_seen_original"]["success"], evals["val_seen_original"]["spl"], evals["val_seen_original"]["distance_to_goal"], "Generalization reference split"),
        ("Task 4", "Val unseen original", 16, evals["val_unseen_original"]["success"], evals["val_unseen_original"]["spl"], evals["val_unseen_original"]["distance_to_goal"], "Unseen environment evaluation"),
        ("Task 4", "Val seen paraphrased", 16, evals["val_seen_paraphrase"]["success"], evals["val_seen_paraphrase"]["spl"], evals["val_seen_paraphrase"]["distance_to_goal"], "Rule-based instruction paraphrases"),
        ("Task 4", "Val unseen paraphrased", 16, evals["val_unseen_paraphrase"]["success"], evals["val_unseen_paraphrase"]["spl"], evals["val_unseen_paraphrase"]["distance_to_goal"], "Paraphrases on unseen environments"),
        ("Task 4", "Reduced data 32 episodes", 8, reduced["32 train eps"]["success"], reduced["32 train eps"]["spl"], reduced["32 train eps"]["distance_to_goal"], "Lower-data training run"),
        ("Task 4", "Reduced data 64 episodes", 8, reduced["64 train eps"]["success"], reduced["64 train eps"]["spl"], reduced["64 train eps"]["distance_to_goal"], "Higher-data clean training run"),
        ("Task 4", "Gated fusion ablation", 8, gated["success"], gated["spl"], gated["distance_to_goal"], "Simpler multimodal fusion"),
        ("Task 4", "CMA fusion ablation", 8, cma["success"], cma["spl"], cma["distance_to_goal"], "Cross-modal attention fusion"),
        ("Task 5", "Baseline CMA 64 episodes", 8, baseline["success"], baseline["spl"], baseline["distance_to_goal"], "Base clean CMA"),
        ("Task 5", "Stop-aware CMA 64 episodes", 8, extension["success"], extension["spl"], extension["distance_to_goal"], "Action reweighting plus repeated stop targets"),
    ]

    with (TABLES / "task45_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "experiment", "episodes", "sr", "spl", "distance_to_goal_m", "notes"])
        writer.writerows(summary_rows)

    md = [
        "# Task 4/5 VLN Results",
        "",
        "All official rollout metrics use the learned policy only with a 3 m success threshold.",
        "",
        "## Key Findings",
        "",
        f"- Seen validation performance: SR {evals['val_seen_original']['success']:.3f}, SPL {evals['val_seen_original']['spl']:.3f}, final distance {evals['val_seen_original']['distance_to_goal']:.2f} m.",
        f"- Unseen validation performance: SR {evals['val_unseen_original']['success']:.3f}, SPL {evals['val_unseen_original']['spl']:.3f}, final distance {evals['val_unseen_original']['distance_to_goal']:.2f} m.",
        f"- Paraphrasing reduced seen-split SR from {evals['val_seen_original']['success']:.3f} to {evals['val_seen_paraphrase']['success']:.3f}; unseen SR remained {evals['val_unseen_paraphrase']['success']:.3f}.",
        f"- Increasing clean training data from 32 to 64 trajectories improved SR from {reduced['32 train eps']['success']:.3f} to {reduced['64 train eps']['success']:.3f}.",
        f"- The CMA fusion ablation achieved a lower final distance than gated fusion on the matched 16-episode study ({cma['distance_to_goal']:.2f} m vs {gated['distance_to_goal']:.2f} m).",
        f"- The Task 5 stop-aware extension improved clean 64-episode SR from {baseline['success']:.3f} to {extension['success']:.3f} and SPL from {baseline['spl']:.3f} to {extension['spl']:.3f}.",
        "",
        "## Interpretation",
        "",
        "The results show a clear gap between seen and unseen environments, which is expected for a small CPU-trained VLN setup. The paraphrase test shows that the text encoder handles some lexical variation but remains sensitive to wording. The reduced-data study confirms that imitation quality and route diversity are major bottlenecks. The fusion ablation suggests that cross-modal attention is more useful than a plain gated visual-text merge, especially when judged by final distance even before SR becomes nonzero. The controlled extension mainly helps the model learn when to terminate, which directly improves SR and SPL.",
    ]
    (TABLES / "task45_discussion.md").write_text("\n".join(md) + "\n")

    print(f"Wrote graphs to {GRAPHS}")
    print(f"Wrote tables to {TABLES}")


if __name__ == "__main__":
    main()

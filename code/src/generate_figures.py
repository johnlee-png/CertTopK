"""Generate reproducible figures from canonical experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.plots import plot_xt_grid, plot_utility_decile_comparison


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def build_xt_grid_figure(xt_grid_path: Path, output_path: Path) -> None:
    xt_grid = np.load(xt_grid_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_xt_grid(xt_grid, ax=ax, title="Train-Only Expected Threat Grid")
    _save_current(output_path)


def build_exp1_figure(final_results: dict, output_path: Path) -> None:
    exp1 = final_results["exp1"]
    labels = ["E1", "E2", "E3", "E4", "E5", "E6"]
    auroc = [exp1[label]["auroc"] for label in labels]
    brier = [exp1[label]["brier_score"] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, auroc, color=["#4c78a8"] * 3 + ["#f58518"] * 3)
    axes[0].set_title("Pass Success AUROC")
    axes[0].set_ylim(0.84, 0.96)
    axes[0].set_ylabel("AUROC")

    axes[1].bar(labels, brier, color=["#4c78a8"] * 3 + ["#f58518"] * 3)
    axes[1].set_title("Pass Success Brier Score")
    axes[1].set_ylabel("Brier")

    fig.suptitle("Experiment 1: Event-only vs Event+360")
    _save_current(output_path)


def build_ablation_figure(final_results: dict, output_path: Path) -> None:
    cumulative = pd.DataFrame(final_results["exp5_full"])
    isolated = pd.DataFrame(final_results["exp5_full_isolated"])
    loo = pd.DataFrame(final_results["exp5_full_leave_one_out"])

    cumulative_labels = [cfg[0] for cfg in cumulative["config"]]
    cumulative_gain = cumulative["auroc"].diff().fillna(0.0)

    isolated_non_base = isolated[isolated["config"] != "A_event_only"].copy()
    isolated_non_base["group"] = isolated_non_base["config"].str.split("_", n=1).str[1]
    isolated_base = isolated.loc[isolated["config"] == "A_event_only", "auroc"].iloc[0]
    isolated_non_base["gain"] = isolated_non_base["auroc"] - isolated_base

    loo_all = loo.loc[loo["config"] == "full_all_groups", "auroc"].iloc[0]
    loo_drop = loo[loo["config"] != "full_all_groups"].copy()
    loo_drop["group"] = loo_drop["config"].str.replace("minus_", "", regex=False).str.split("_", n=1).str[1]
    loo_drop["drop"] = loo_all - loo_drop["auroc"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].bar(cumulative_labels, cumulative_gain, color="#4c78a8")
    axes[0].set_title("Cumulative Gain")
    axes[0].set_ylabel("AUROC delta")

    axes[1].bar(isolated_non_base["group"], isolated_non_base["gain"], color="#54a24b")
    axes[1].set_title("Single-Group on Baseline")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(loo_drop["group"], loo_drop["drop"], color="#e45756")
    axes[2].set_title("Leave-One-Out Drop")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle("Experiment 5: Receiver-Space Remains Dominant Across Ablations")
    _save_current(output_path)


def build_utility_deciles_figure(final_results: dict, output_path: Path) -> None:
    decile_event = pd.DataFrame(final_results["exp3_event"])
    decile_360 = pd.DataFrame(final_results["exp3_360"])

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_utility_decile_comparison(
        decile_event,
        decile_360,
        metric="avg_realized_delta_xt",
        ax=ax,
    )
    ax.set_title("Utility Deciles by Realized Delta xT")
    _save_current(output_path)


def build_exp9_figure(exp9_results: dict, output_path: Path) -> None:
    configs = [
        ("E3_event_only", "E3\nEvent-only"),
        ("E6_event_360_handcrafted", "E6\nEvent+360\n(handcrafted)"),
        ("E7_event_set_transformer", "E7\nEvent+SetTransformer\n(end-to-end NN)"),
        ("E8_event_plus_embedding_lgbm", "E8\nEvent+ST-embed\n(LGBM)"),
        ("E8b_event_360_plus_embedding_lgbm", "E8b\nEvent+360+ST-embed\n(LGBM)"),
    ]
    labels = [label for _, label in configs]
    points = [exp9_results[key]["auroc_bootstrap"]["point"] for key, _ in configs]
    los = [exp9_results[key]["auroc_bootstrap"]["ci_low"] for key, _ in configs]
    his = [exp9_results[key]["auroc_bootstrap"]["ci_high"] for key, _ in configs]
    errors = [[p - lo for p, lo in zip(points, los)], [hi - p for p, hi in zip(points, his)]]

    colors = ["#4c78a8", "#f58518", "#e45756", "#54a24b", "#b279a2"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, points, yerr=errors, capsize=4, color=colors)
    ax.set_ylabel("AUROC (match-block bootstrap 95% CI)")
    ax.set_ylim(0.88, 0.96)
    ax.set_title("Experiment 9: Set Transformer vs Hand-Crafted 360 Features")
    _save_current(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-results", default="experiments/results_rebuilt/final_results.json")
    parser.add_argument("--xt-grid", default="experiments/results_rebuilt/xt_grid_train_only.npy")
    parser.add_argument("--output-dir", default="experiments/results_rebuilt/figures")
    args = parser.parse_args()

    final_results = _load_json(args.final_results)
    output_dir = Path(args.output_dir)

    build_xt_grid_figure(Path(args.xt_grid), output_dir / "fig1_xt_grid.png")
    build_exp1_figure(final_results, output_dir / "fig2_exp1_comparison.png")
    build_ablation_figure(final_results, output_dir / "fig3_ablation.png")
    build_utility_deciles_figure(final_results, output_dir / "fig4_utility_deciles.png")

    exp9_path = Path("experiments/results/exp9_set_transformer.json")
    if exp9_path.exists():
        build_exp9_figure(_load_json(exp9_path), output_dir / "fig5_exp9_set_transformer.png")


if __name__ == "__main__":
    main()

"""Experiment 15 (Phase 9): un-xPass-style pass creativity metric.

Unlike Phases 1-8 (does a change improve pass-success prediction?), this is a
descriptive/analytical metric, not a predictive-accuracy comparison. We check
that it behaves sensibly: creative passes should skew toward progressive,
successful passes with high realized value, and the qualitative top examples
should look like genuinely inventive choices rather than noise.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import match_level_split
from src.features.creativity_metric import compute_creativity


def _json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def run_creativity_metric_experiment(
    processed_dir: str = "data/processed",
    output_dir: str = "experiments/results",
    seed: int = 42,
) -> dict:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed artifacts...")
    passes = pd.read_parquet(
        processed_dir / "passes_with_features.parquet",
        columns=["match_id", "freeze_frame", "start_x", "start_y", "end_x", "end_y",
                 "delta_xt", "pass_success", "is_progressive", "player_name"],
    )
    xt_grid = np.load(processed_dir / "xt_grid_train_only.npy")

    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)
    test_idx = test_df.index

    print(f"\nComputing creativity for {len(test_idx)} test-set passes...")
    passes_test = passes.loc[test_idx]
    creativity_df = compute_creativity(passes_test, xt_grid)

    has_candidates = creativity_df["n_candidates"] > 0
    print(f"  passes with >=1 visible teammate candidate: {has_candidates.sum()} / {len(creativity_df)}")
    print(f"  mean originality: {creativity_df.loc[has_candidates, 'originality'].mean():.4f}")
    print(f"  mean creativity: {creativity_df.loc[has_candidates, 'creativity'].mean():.6f}")

    combined = passes_test.join(creativity_df)
    combined = combined[combined["n_candidates"] > 0]

    print("\nSanity check: creativity by pass type...")
    prog_creativity = combined.loc[combined["is_progressive"], "creativity"].mean()
    nonprog_creativity = combined.loc[~combined["is_progressive"], "creativity"].mean()
    success_originality = combined.loc[combined["pass_success"] == 1, "originality"].mean()
    fail_originality = combined.loc[combined["pass_success"] == 0, "originality"].mean()
    print(f"  mean creativity, progressive passes: {prog_creativity:.6f}")
    print(f"  mean creativity, non-progressive passes: {nonprog_creativity:.6f}")
    print(f"  mean originality, successful passes: {success_originality:.4f}")
    print(f"  mean originality, failed passes: {fail_originality:.4f}")

    print("\nTop 20 most creative individual passes (test set):")
    top20 = combined.nlargest(20, "creativity")[
        ["match_id", "player_name", "start_x", "start_y", "end_x", "end_y",
         "n_candidates", "originality", "chosen_xt", "best_candidate_xt", "creativity"]
    ]
    print(top20.to_string(index=False).encode("ascii", errors="replace").decode("ascii"))

    print("\nTop 20 most creative PLAYERS (mean over their test-set passes, min 5 passes):")
    player_agg = (
        combined.groupby("player_name")
        .agg(n_passes=("creativity", "size"),
             mean_creativity=("creativity", "mean"),
             mean_originality=("originality", "mean"),
             success_rate=("pass_success", "mean"))
        .query("n_passes >= 5")
        .sort_values("mean_creativity", ascending=False)
        .head(20)
        .reset_index()
    )
    print(player_agg.to_string(index=False).encode("ascii", errors="replace").decode("ascii"))

    results = {
        "description": "un-xPass-style creativity = originality x realized value, test set",
        "summary": {
            "n_test_passes": int(len(creativity_df)),
            "n_with_candidates": int(has_candidates.sum()),
            "mean_originality": float(creativity_df.loc[has_candidates, "originality"].mean()),
            "mean_creativity": float(creativity_df.loc[has_candidates, "creativity"].mean()),
        },
        "sanity_checks": {
            "mean_creativity_progressive": float(prog_creativity),
            "mean_creativity_non_progressive": float(nonprog_creativity),
            "mean_originality_successful": float(success_originality),
            "mean_originality_failed": float(fail_originality),
        },
        "top_20_creative_passes": top20.to_dict(orient="records"),
        "top_20_creative_players": player_agg.to_dict(orient="records"),
        "split_manifest": {
            "seed": seed,
            "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
            "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
            "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
        },
    }

    output_path = output_dir / "exp15_creativity_metric.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_creativity_metric_experiment()

"""Experiment 20 (Phase 14): quantitative benchmark for the un-xPass-style creativity metric.

Phase 9 (exp15) validated the creativity metric only qualitatively (top passes/players
matched famous playmakers like Kevin De Bruyne). This adds an existing-data quantitative
benchmark: StatsBomb already flags passes that directly led to a shot (`pass_shot_assist`)
or a goal (`pass_goal_assist`). If the creativity metric captures something real, passes
that are creative (an unobvious choice of receiver) should be more likely to be assists
than passes are on average -- a testable, non-anecdotal claim.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.splits import match_level_split
from src.features.creativity_metric import compute_creativity
from src.evaluation.metrics import bootstrap_metric


def _json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def _auroc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def run_creativity_benchmark_experiment(
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
                 "delta_xt", "pass_success", "is_progressive", "player_name",
                 "pass_shot_assist", "pass_goal_assist"],
    )
    xt_grid = np.load(processed_dir / "xt_grid_train_only.npy")

    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)
    test_idx = test_df.index

    passes_test = passes.loc[test_idx].copy()
    passes_test["pass_shot_assist"] = passes_test["pass_shot_assist"].fillna(False).astype(bool)
    passes_test["pass_goal_assist"] = passes_test["pass_goal_assist"].fillna(False).astype(bool)

    print(f"\nComputing creativity for {len(passes_test)} test-set passes...")
    creativity_df = compute_creativity(passes_test, xt_grid)
    combined = passes_test.join(creativity_df)
    combined = combined[combined["n_candidates"] > 0]
    match_ids = combined["match_id"].values

    print(f"Test passes with >=1 visible teammate candidate: {len(combined)}")
    print(f"  shot-assists: {combined['pass_shot_assist'].sum()}  "
          f"goal-assists: {combined['pass_goal_assist'].sum()}\n")

    results = {"description": "Quantitative benchmark: does the creativity metric (Phase 9) "
                               "discriminate real shot/goal assists from ordinary passes?"}

    for target in ["pass_shot_assist", "pass_goal_assist"]:
        y = combined[target].astype(int).values
        print(f"=== Target: {target} (base rate {y.mean():.4%}) ===")
        target_results = {}
        for score_name in ["creativity", "originality"]:
            score = combined[score_name].values
            pt, lo, hi = bootstrap_metric(y, score, _auroc, match_ids=match_ids)
            mean_assist = combined.loc[combined[target], score_name].mean()
            mean_non_assist = combined.loc[~combined[target], score_name].mean()
            print(f"  {score_name}: AUROC={pt:.4f} [{lo:.4f},{hi:.4f}]  "
                  f"mean(assist)={mean_assist:.5f} vs mean(non-assist)={mean_non_assist:.5f}")
            target_results[score_name] = {
                "auroc": pt, "ci_low": lo, "ci_high": hi,
                "mean_assist": float(mean_assist), "mean_non_assist": float(mean_non_assist),
            }
        results[target] = target_results
        print()

    results["n_test_passes_with_candidates"] = len(combined)
    results["split_manifest"] = {
        "seed": seed,
        "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
        "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
        "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
    }

    output_path = output_dir / "exp20_creativity_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    run_creativity_benchmark_experiment()

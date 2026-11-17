"""Experiment 14 (Phase 8): xT calibration via a model-based shot-xG prior.

NEXT_STEPS.md Tier 1 direction #4 proposes calibrating the xT reward with an
external xG model (e.g. Understat). We use StatsBomb's own pre-fit shot xG
(`shot_statsbomb_xg`, already present in the data) instead of scraping and
matching shots against a second provider, which would introduce its own
alignment error. This replaces the raw empirical goals/shots ratio per zone
(noisy in sparsely-shot zones) with the mean model-based xG of shots in that
zone -- a variance-reduction/calibration change, not a new data source.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.data.splits import match_level_split
from src.features.expected_threat import build_xt_grid, compute_delta_xt, get_xt_value
from src.evaluation.metrics import evaluate_utility_deciles


def _json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    raise TypeError(f"Not serializable: {type(obj)}")


def run_xt_calibration_experiment(
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
        columns=["match_id", "start_x", "start_y", "end_x", "end_y", "is_progressive", "pass_success"],
    )
    all_events = pd.read_parquet(
        processed_dir / "events_360_all.parquet",
        columns=["match_id", "location", "type_name", "pass_end_location",
                 "carry_end_location", "shot_outcome_name", "shot_statsbomb_xg"],
    )

    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)
    train_idx, valid_idx, test_idx = train_df.index, valid_df.index, test_df.index
    train_match_ids = set(train_df["match_id"].unique())
    test_match_ids = set(test_df["match_id"].unique())

    all_events_typed = all_events[all_events["type_name"].isin(["Pass", "Carry", "Shot"])].copy()
    train_events = all_events_typed[all_events_typed["match_id"].isin(train_match_ids)].copy()
    test_events = all_events_typed[all_events_typed["match_id"].isin(test_match_ids)].copy()

    print("Building original (goals/shots ratio) xT grid...")
    xt_original = build_xt_grid(train_events)

    print("Building calibrated (StatsBomb shot-xG prior) xT grid...")
    xt_calibrated = build_xt_grid(train_events, external_xg=train_events["shot_statsbomb_xg"].values)

    print(f"  original grid max xT: {xt_original.max():.4f}")
    print(f"  calibrated grid max xT: {xt_calibrated.max():.4f}")
    grid_diff = np.abs(xt_original - xt_calibrated)
    print(f"  mean |difference| across all zones: {grid_diff.mean():.5f}, max: {grid_diff.max():.5f}")

    # ===== Out-of-sample calibration check: does the zone-xG predict real
    # test-set goal outcomes better under the calibrated grid? =====
    print("\nEvaluating out-of-sample shot-outcome calibration on the test set...")
    test_shots = test_events[test_events["type_name"] == "Shot"].copy()

    def _get_coord(loc, i):
        try:
            if loc is not None and hasattr(loc, "__len__") and len(loc) > i:
                return float(loc[i])
        except (TypeError, ValueError, IndexError):
            pass
        return np.nan

    test_shots["x"] = test_shots["location"].apply(lambda l: _get_coord(l, 0))
    test_shots["y"] = test_shots["location"].apply(lambda l: _get_coord(l, 1))
    test_shots = test_shots.dropna(subset=["x", "y"])
    test_shots["is_goal"] = (test_shots["shot_outcome_name"] == "Goal").astype(int)

    zone_xg_original = test_shots.apply(lambda r: get_xt_value(xt_original, r["x"], r["y"]), axis=1)
    zone_xg_calibrated = test_shots.apply(lambda r: get_xt_value(xt_calibrated, r["x"], r["y"]), axis=1)

    # xT grid values are the full value-iterated xT, not a raw probability in
    # [0,1] for shots specifically; clip for a fair Brier/log-loss comparison.
    p_original = np.clip(zone_xg_original.values, 1e-4, 1 - 1e-4)
    p_calibrated = np.clip(zone_xg_calibrated.values, 1e-4, 1 - 1e-4)
    y_true = test_shots["is_goal"].values

    brier_original = brier_score_loss(y_true, p_original)
    brier_calibrated = brier_score_loss(y_true, p_calibrated)
    logloss_original = log_loss(y_true, p_original)
    logloss_calibrated = log_loss(y_true, p_calibrated)

    print(f"  n test shots: {len(test_shots)}, goal rate: {y_true.mean():.4f}")
    print(f"  Brier   -- original: {brier_original:.4f}  calibrated: {brier_calibrated:.4f}")
    print(f"  LogLoss -- original: {logloss_original:.4f}  calibrated: {logloss_calibrated:.4f}")

    # ===== Downstream check: utility ranking realized quality =====
    print("\nRecomputing delta_xt and utility-decile realized quality with each grid...")
    passes_test = passes.loc[test_idx].copy()

    results_by_grid = {}
    for name, grid in [("original", xt_original), ("calibrated", xt_calibrated)]:
        passes_test["delta_xt"] = compute_delta_xt(passes_test, grid)
        passes_test["realized_delta_xt"] = passes_test["delta_xt"] * passes_test["pass_success"]
        passes_test["realized_progressive"] = (
            passes_test["is_progressive"] & passes_test["pass_success"].astype(bool)
        ).astype(int)
        passes_test["realized_final_third_entry"] = (
            (passes_test["end_x"] >= 80) & passes_test["pass_success"].astype(bool)
        ).astype(int)
        # Rank purely by reward (delta_xt) here since we are isolating the reward
        # component's quality, not the full P*R-(1-P)*C utility.
        passes_test["utility_score"] = passes_test["delta_xt"]
        deciles = evaluate_utility_deciles(passes_test)
        results_by_grid[name] = deciles.to_dict(orient="records")
        top_decile = deciles.iloc[-1]
        print(f"  {name}: top-decile avg_realized_delta_xt={top_decile['avg_realized_delta_xt']:.4f}, "
              f"realized_progressive_pct={top_decile['realized_progressive_pct']:.4f}")

    results = {
        "description": "xT calibration: StatsBomb shot-xG-averaged zones vs raw goals/shots ratio",
        "grid_comparison": {
            "max_xt_original": float(xt_original.max()),
            "max_xt_calibrated": float(xt_calibrated.max()),
            "mean_abs_diff": float(grid_diff.mean()),
            "max_abs_diff": float(grid_diff.max()),
        },
        "shot_calibration": {
            "n_test_shots": int(len(test_shots)),
            "goal_rate": float(y_true.mean()),
            "brier_original": float(brier_original),
            "brier_calibrated": float(brier_calibrated),
            "log_loss_original": float(logloss_original),
            "log_loss_calibrated": float(logloss_calibrated),
        },
        "reward_ranking_by_grid": results_by_grid,
        "split_manifest": {
            "seed": seed,
            "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
            "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
            "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
        },
    }

    output_path = output_dir / "exp14_xt_calibration.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_xt_calibration_experiment()

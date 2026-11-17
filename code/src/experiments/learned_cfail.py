"""Experiment 7: learned C_fail vs the location_based heuristic.

Reuses the already-processed parquet/xT-grid artifacts from src.pipeline so this
does not require re-loading raw StatsBomb JSON. Trains a failure-cost regressor on
realized turnover-cost labels (see src.data.turnover_labels) and compares it against
the existing heuristic on (a) correlation with realized post-turnover danger and
(b) realized-outcome utility ranking quality.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.splits import match_level_split
from src.data.turnover_labels import compute_realized_turnover_cost
from src.models.pass_success import train_success_model, predict_success_proba
from src.models.reward_model import train_reward_model, predict_reward
from src.models.failure_cost_model import train_failure_cost_model, predict_failure_cost
from src.models.utility import compute_failure_cost, compute_utility
from src.evaluation.metrics import bootstrap_metric, evaluate_utility_deciles


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


def _spearman(y_true, y_pred) -> float:
    return spearmanr(y_true, y_pred).correlation


def run_learned_cfail_experiment(
    processed_dir: str = "data/processed",
    output_dir: str = "experiments/results",
    seed: int = 42,
    window_events: int = 10,
    window_seconds: float = 15.0,
) -> dict:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed artifacts...")
    passes = pd.read_parquet(processed_dir / "passes_with_features.parquet")
    X_full = pd.read_parquet(processed_dir / "X_full.parquet")
    y = pd.read_parquet(processed_dir / "y_labels.parquet")["pass_success"]
    events_all = pd.read_parquet(
        processed_dir / "events_360_all.parquet",
        columns=["match_id", "index", "period", "minute", "second", "team_name", "location"],
    )
    xt_grid = np.load(processed_dir / "xt_grid_train_only.npy")

    passes["realized_delta_xt"] = passes["delta_xt"] * passes["pass_success"]
    passes["realized_progressive"] = (
        passes["is_progressive"] & passes["pass_success"].astype(bool)
    ).astype(int)
    passes["realized_final_third_entry"] = (
        (passes["end_x"] >= 80) & passes["pass_success"].astype(bool)
    ).astype(int)

    print("Splitting (must match the seed used to build the processed artifacts)...")
    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)
    train_idx, valid_idx, test_idx = train_df.index, valid_df.index, test_df.index

    print(f"\nLabeling realized turnover cost (window={window_events} events / {window_seconds}s)...")
    realized_cost = compute_realized_turnover_cost(
        passes, events_all, xt_grid, window_events=window_events, window_seconds=window_seconds,
    )
    turnover_idx = passes[passes["pass_outcome_name"].notna()].index
    cf_train = train_idx.intersection(turnover_idx)
    cf_valid = valid_idx.intersection(turnover_idx)
    cf_test = test_idx.intersection(turnover_idx)
    print(f"Turnovers: {len(cf_train)} train / {len(cf_valid)} valid / {len(cf_test)} test")

    # ===== Train learned C_fail model (train-only, no test/valid leakage) =====
    print("\nTraining learned C_fail model (LightGBM regressor)...")
    cfail_model = train_failure_cost_model(
        "lightgbm_regressor",
        X_full.loc[cf_train].fillna(0), realized_cost.loc[cf_train],
        X_full.loc[cf_valid].fillna(0), realized_cost.loc[cf_valid],
    )

    # ===== Cost realism: does the learned cost track realized post-turnover danger? =====
    print("\nEvaluating cost realism on test turnovers...")
    learned_cost_test_turnovers = predict_failure_cost(cfail_model, X_full.loc[cf_test].fillna(0))
    heuristic_cost_test_turnovers = compute_failure_cost(passes.loc[cf_test], method="location_based")

    match_ids_cf_test = passes.loc[cf_test, "match_id"].values
    learned_pt, learned_lo, learned_hi = bootstrap_metric(
        realized_cost.loc[cf_test].values, learned_cost_test_turnovers, _spearman,
        n_bootstrap=1000, match_ids=match_ids_cf_test,
    )
    heur_pt, heur_lo, heur_hi = bootstrap_metric(
        realized_cost.loc[cf_test].values, heuristic_cost_test_turnovers, _spearman,
        n_bootstrap=1000, match_ids=match_ids_cf_test,
    )
    print(f"  learned   spearman: {learned_pt:.4f} [{learned_lo:.4f}, {learned_hi:.4f}]")
    print(f"  heuristic spearman: {heur_pt:.4f} [{heur_lo:.4f}, {heur_hi:.4f}]")

    # ===== Success (E6-equivalent) and reward (R2-equivalent) models, held fixed =====
    print("\nTraining success (Event+360 LGBM) and reward (Event+360 LGBM) models for utility scoring...")
    success_model = train_success_model(
        "lightgbm",
        X_full.loc[train_idx].fillna(0), y.loc[train_idx],
        X_full.loc[valid_idx].fillna(0), y.loc[valid_idx],
    )
    p_succ_test = predict_success_proba(success_model, X_full.loc[test_idx].fillna(0))

    success_mask = y == 1
    r_train = train_idx.intersection(passes[success_mask].index)
    r_valid = valid_idx.intersection(passes[success_mask].index)
    reward_model = train_reward_model(
        "lightgbm_regressor",
        X_full.loc[r_train].fillna(0), passes.loc[r_train, "delta_xt"],
        X_full.loc[r_valid].fillna(0), passes.loc[r_valid, "delta_xt"],
    )
    r_pred_test = predict_reward(reward_model, X_full.loc[test_idx].fillna(0))

    # ===== Utility ranking: heuristic vs learned C_fail, same P_succ/R_succ =====
    print("\nComparing utility ranking (realized outcomes) between cost methods...")
    learned_cost_all_test = predict_failure_cost(cfail_model, X_full.loc[test_idx].fillna(0))

    c_fail_heuristic = compute_failure_cost(passes.loc[test_idx], method="location_based")
    c_fail_learned = compute_failure_cost(passes.loc[test_idx], method="learned", learned_cost=learned_cost_all_test)

    utility_heuristic = compute_utility(p_succ_test, r_pred_test, c_fail_heuristic)
    utility_learned = compute_utility(p_succ_test, r_pred_test, c_fail_learned)

    eval_heuristic = passes.loc[test_idx].copy()
    eval_heuristic["utility_score"] = utility_heuristic
    deciles_heuristic = evaluate_utility_deciles(eval_heuristic)

    eval_learned = passes.loc[test_idx].copy()
    eval_learned["utility_score"] = utility_learned
    deciles_learned = evaluate_utility_deciles(eval_learned)

    print("\n  Heuristic C_fail utility deciles:")
    print(deciles_heuristic.to_string(index=False))
    print("\n  Learned C_fail utility deciles:")
    print(deciles_learned.to_string(index=False))

    results = {
        "description": "Learned C_fail (LightGBM on realized post-turnover opponent xT) vs location_based heuristic",
        "labeling": {"window_events": window_events, "window_seconds": window_seconds},
        "cost_realism": {
            "learned": {"spearman": learned_pt, "ci_low": learned_lo, "ci_high": learned_hi},
            "heuristic_location_based": {"spearman": heur_pt, "ci_low": heur_lo, "ci_high": heur_hi},
            "n_test_turnovers": len(cf_test),
        },
        "utility_deciles_heuristic": deciles_heuristic.to_dict(orient="records"),
        "utility_deciles_learned": deciles_learned.to_dict(orient="records"),
        "split_manifest": {
            "seed": seed,
            "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
            "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
            "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
        },
    }

    output_path = output_dir / "exp7_learned_cfail.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_learned_cfail_experiment()

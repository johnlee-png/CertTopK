"""Main experiment pipeline: data loading -> feature extraction -> modeling -> evaluation."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import (
    get_matches_with_360,
    load_and_join_events_360,
    filter_open_play_passes,
    label_progressive_passes,
)
from src.data.splits import match_level_split
from src.features.event_features import extract_event_features, extract_pass_outcome
from src.features.context_360_features import extract_360_features
from src.features.expected_threat import build_xt_grid, compute_delta_xt, get_xt_value
from src.models.pass_success import train_success_model, predict_success_proba
from src.models.reward_model import train_reward_model, predict_reward
from src.models.utility import compute_failure_cost, compute_utility, rank_passes_by_utility
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    top_decile_precision,
    evaluate_utility_deciles,
)


def run_pipeline(base_path: str, output_dir: str = "experiments/results", seed: int = 42):
    """Run the full experiment pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # ==============================
    # Step 1: Data Loading
    # ==============================
    print("=" * 60)
    print("Step 1: Loading data...")
    print("=" * 60)

    match_ids = get_matches_with_360(base_path)
    print(f"Found {len(match_ids)} matches with 360 data")

    df = load_and_join_events_360(base_path, match_ids)
    print(f"Total events with 360: {len(df)}")

    # ==============================
    # Step 2: Filter & Label
    # ==============================
    print("\nStep 2: Filtering open-play passes...")
    passes = filter_open_play_passes(df)
    print(f"Open-play footed passes: {len(passes)}")

    passes = label_progressive_passes(passes)
    n_prog = passes["is_progressive"].sum()
    print(f"Progressive passes: {n_prog} ({n_prog/len(passes)*100:.1f}%)")

    # ==============================
    # Step 3: Data Split
    # ==============================
    print("\nStep 3: Match-level split...")
    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)

    train_idx = train_df.index
    valid_idx = valid_df.index
    test_idx = test_df.index
    train_match_ids = sorted(train_df["match_id"].unique().tolist())
    valid_match_ids = sorted(valid_df["match_id"].unique().tolist())
    test_match_ids = sorted(test_df["match_id"].unique().tolist())

    # ==============================
    # Step 4: Build xT Grid
    # ==============================
    print("\nStep 4: Building train-only xT grid...")
    all_events = df[df["type_name"].isin(["Pass", "Carry", "Shot"])].copy()
    train_events = all_events[all_events["match_id"].isin(train_match_ids)].copy()
    xt_grid = build_xt_grid(train_events)
    print(f"xT grid shape: {xt_grid.shape}, max xT: {xt_grid.max():.4f}")

    passes["delta_xt"] = compute_delta_xt(passes, xt_grid)
    passes["start_xt"] = passes.apply(
        lambda r: get_xt_value(xt_grid, r["start_x"], r["start_y"]), axis=1
    )
    passes["realized_delta_xt"] = passes["delta_xt"] * passes["pass_outcome_name"].isna().astype(int)
    passes["realized_progressive"] = (
        passes["is_progressive"] & passes["pass_outcome_name"].isna()
    ).astype(int)
    passes["realized_final_third_entry"] = (
        (passes["end_x"] >= 80) & passes["pass_outcome_name"].isna()
    ).astype(int)

    # ==============================
    # Step 5: Feature Extraction
    # ==============================
    print("\nStep 5: Extracting features...")
    event_features = extract_event_features(passes)
    context_features = extract_360_features(passes)
    y = extract_pass_outcome(passes)
    passes["pass_success"] = y

    # Combined feature sets
    event_only_cols = event_features.columns.tolist()
    all_cols = event_only_cols + context_features.columns.tolist()

    X_event = event_features
    X_full = pd.concat([event_features, context_features], axis=1)

    print(f"Event-only features: {len(event_only_cols)}")
    print(f"Event+360 features: {len(all_cols)}")

    # ==============================
    # Experiment 1: Pass Success Prediction
    # ==============================
    print("\n" + "=" * 60)
    print("Experiment 1: Pass Success Prediction")
    print("=" * 60)

    exp1_results = {}
    model_configs = [
        ("E1", "logistic_regression", X_event, "Event-only LR"),
        ("E2", "xgboost", X_event, "Event-only XGB"),
        ("E3", "lightgbm", X_event, "Event-only LGBM"),
        ("E4", "logistic_regression", X_full, "Event+360 LR"),
        ("E5", "xgboost", X_full, "Event+360 XGB"),
        ("E6", "lightgbm", X_full, "Event+360 LGBM"),
    ]

    trained_models = {}
    for tag, model_name, X, desc in model_configs:
        print(f"\n  Training {tag}: {desc}...")

        # Handle NaN for linear models
        X_train = X.loc[train_idx].fillna(0)
        X_val = X.loc[valid_idx].fillna(0)
        X_test = X.loc[test_idx].fillna(0)
        y_train = y.loc[train_idx]
        y_val = y.loc[valid_idx]
        y_test = y.loc[test_idx]

        model = train_success_model(model_name, X_train, y_train, X_val, y_val)
        y_prob = predict_success_proba(model, X_test)

        metrics = compute_classification_metrics(y_test.values, y_prob)
        exp1_results[tag] = {"description": desc, **metrics}
        trained_models[tag] = (model, X)

        print(f"    AUROC={metrics['auroc']:.4f}  LogLoss={metrics['log_loss']:.4f}  "
              f"Brier={metrics['brier_score']:.4f}  ECE={metrics['ece']:.4f}")

    results["experiment_1"] = exp1_results

    # ==============================
    # Experiment 2: Reward Prediction
    # ==============================
    print("\n" + "=" * 60)
    print("Experiment 2: Reward Prediction (Delta xT)")
    print("=" * 60)

    # Only use successful passes for reward prediction
    success_mask = y == 1
    reward_passes = passes[success_mask]

    r_train = reward_passes.index.intersection(train_idx)
    r_valid = reward_passes.index.intersection(valid_idx)
    r_test = reward_passes.index.intersection(test_idx)

    y_reward_train = passes.loc[r_train, "delta_xt"]
    y_reward_valid = passes.loc[r_valid, "delta_xt"]
    y_reward_test = passes.loc[r_test, "delta_xt"]

    exp2_results = {}
    reward_configs = [
        ("R1", "lightgbm_regressor", X_event, "Event-only LGBM"),
        ("R2", "lightgbm_regressor", X_full, "Event+360 LGBM"),
    ]

    reward_models = {}
    for tag, model_name, X, desc in reward_configs:
        print(f"\n  Training {tag}: {desc}...")
        X_train_r = X.loc[r_train].fillna(0)
        X_val_r = X.loc[r_valid].fillna(0)
        X_test_r = X.loc[r_test].fillna(0)

        model = train_reward_model(model_name, X_train_r, y_reward_train, X_val_r, y_reward_valid)
        y_pred = predict_reward(model, X_test_r)

        metrics = compute_regression_metrics(y_reward_test.values, y_pred)
        metrics["top_decile_precision"] = top_decile_precision(y_reward_test.values, y_pred)
        exp2_results[tag] = {"description": desc, **metrics}
        reward_models[tag] = (model, X)

        print(f"    RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
              f"Spearman={metrics['spearman_corr']:.4f}")

    results["experiment_2"] = exp2_results

    # ==============================
    # Experiment 3: Utility Ranking
    # ==============================
    print("\n" + "=" * 60)
    print("Experiment 3: Utility Ranking Evaluation")
    print("=" * 60)

    exp3_results = {}

    for label, success_tag, reward_tag in [("event_only", "E3", "R1"), ("event_360", "E6", "R2")]:
        s_model, s_X = trained_models[success_tag]
        r_model, r_X = reward_models[reward_tag]

        X_test_s = s_X.loc[test_idx].fillna(0)
        X_test_r = r_X.loc[test_idx].fillna(0)

        p_succ = predict_success_proba(s_model, X_test_s)
        r_pred = predict_reward(r_model, X_test_r)
        c_fail = compute_failure_cost(passes.loc[test_idx])

        utility = compute_utility(p_succ, r_pred, c_fail)

        test_eval = passes.loc[test_idx].copy()
        test_eval["utility_score"] = utility
        decile_stats = evaluate_utility_deciles(test_eval)
        exp3_results[label] = decile_stats.to_dict(orient="records")

        print(f"\n  {label} utility decile stats:")
        print(decile_stats.to_string(index=False))

    results["experiment_3"] = exp3_results
    results["split_manifest"] = {
        "seed": seed,
        "train_match_ids": train_match_ids,
        "valid_match_ids": valid_match_ids,
        "test_match_ids": test_match_ids,
    }

    # ==============================
    # Save Results
    # ==============================
    output_path = Path(output_dir) / "results.json"

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_serializable = json.loads(
        json.dumps(results, default=convert_types)
    )

    with open(output_path, "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_path}")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pass risk-reward modeling pipeline")
    parser.add_argument(
        "--data-path",
        default="data/raw/statsbomb-open-data/data",
        help="Path to StatsBomb data directory",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results",
        help="Output directory for results",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pipeline(args.data_path, args.output_dir, args.seed)

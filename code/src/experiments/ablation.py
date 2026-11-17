"""Experiment 5: Ablation study for 360 context feature groups."""

import pandas as pd
import numpy as np
from typing import Optional

from src.models.pass_success import train_success_model, predict_success_proba
from src.evaluation.metrics import compute_classification_metrics


# Feature group definitions
FEATURE_GROUPS = {
    "A_event_only": [],  # No 360 features (baseline)
    "B_passer_pressure": [
        "passer_opp_3m", "passer_opp_5m", "passer_nearest_def_dist",
        "passer_teammate_5m", "passer_local_superiority",
    ],
    "C_receiver_space": [
        "receiver_opp_3m", "receiver_opp_5m", "receiver_nearest_def_dist",
        "receiver_teammate_5m", "receiver_goalside_defenders", "receiver_free_space",
    ],
    "D_lane_obstruction": [
        "lane_defender_count", "lane_obstruction_ratio", "corridor_density",
    ],
    "E_team_structure": [
        "visible_teammates", "visible_opponents", "total_visible_players",
        "context_completeness", "pressure_escape", "between_lines_proxy",
        "forward_angle_openness", "receiver_support_count",
    ],
}


def get_cumulative_feature_sets(
    event_features: list[str],
) -> dict[str, list[str]]:
    """Build cumulative feature sets for ablation.

    A: event-only
    B: A + passer pressure
    C: B + receiver space
    D: C + lane obstruction
    E: D + team structure (full 360)
    """
    sets = {}
    cumulative = list(event_features)

    sets["A_event_only"] = list(cumulative)

    for group_name, group_feats in list(FEATURE_GROUPS.items())[1:]:
        cumulative = cumulative + group_feats
        sets[group_name] = list(cumulative)

    return sets


def get_isolated_feature_sets(
    event_features: list[str],
) -> dict[str, list[str]]:
    """Build event baseline plus one-group-at-a-time feature sets."""
    sets = {"A_event_only": list(event_features)}

    for group_name, group_feats in list(FEATURE_GROUPS.items())[1:]:
        sets[group_name] = list(event_features) + list(group_feats)

    return sets


def get_leave_one_out_feature_sets(
    event_features: list[str],
) -> dict[str, list[str]]:
    """Build full model and leave-one-group-out variants."""
    full_features = list(event_features)
    for group_feats in list(FEATURE_GROUPS.values())[1:]:
        full_features.extend(group_feats)

    sets = {"full_all_groups": list(full_features)}

    for group_name, group_feats in list(FEATURE_GROUPS.items())[1:]:
        sets[f"minus_{group_name}"] = [f for f in full_features if f not in group_feats]

    return sets


def run_ablation(
    X_full: pd.DataFrame,
    y: pd.Series,
    event_feature_cols: list[str],
    train_idx: pd.Index,
    valid_idx: pd.Index,
    test_idx: pd.Index,
    model_name: str = "lightgbm",
) -> pd.DataFrame:
    """Run ablation study across feature group combinations.

    Returns DataFrame with metrics for each ablation configuration.
    """
    feature_sets = get_cumulative_feature_sets(event_feature_cols)
    results = []

    for config_name, feature_cols in feature_sets.items():
        # Filter to available columns
        available = [c for c in feature_cols if c in X_full.columns]

        X_train = X_full.loc[train_idx, available].fillna(0)
        X_valid = X_full.loc[valid_idx, available].fillna(0)
        X_test = X_full.loc[test_idx, available].fillna(0)
        y_train = y.loc[train_idx]
        y_valid = y.loc[valid_idx]
        y_test = y.loc[test_idx]

        model = train_success_model(model_name, X_train, y_train, X_valid, y_valid)
        y_prob = predict_success_proba(model, X_test)

        metrics = compute_classification_metrics(y_test.values, y_prob)
        metrics["config"] = config_name
        metrics["n_features"] = len(available)

        # Identify the group added in this step
        group_keys = list(FEATURE_GROUPS.keys())
        idx = group_keys.index(config_name) if config_name in group_keys else 0
        if idx == 0:
            metrics["added_group"] = "baseline"
        else:
            metrics["added_group"] = group_keys[idx].split("_", 1)[1]

        results.append(metrics)
        print(f"  {config_name} ({len(available)} features): "
              f"AUROC={metrics['auroc']:.4f}  LogLoss={metrics['log_loss']:.4f}")

    return pd.DataFrame(results)


def run_feature_set_evaluation(
    X_full: pd.DataFrame,
    y: pd.Series,
    feature_sets: dict[str, list[str]],
    train_idx: pd.Index,
    valid_idx: pd.Index,
    test_idx: pd.Index,
    model_name: str = "lightgbm",
    analysis_name: str = "custom",
) -> pd.DataFrame:
    """Evaluate an explicit mapping of feature set names to columns."""
    results = []

    for config_name, feature_cols in feature_sets.items():
        available = [c for c in feature_cols if c in X_full.columns]

        X_train = X_full.loc[train_idx, available].fillna(0)
        X_valid = X_full.loc[valid_idx, available].fillna(0)
        X_test = X_full.loc[test_idx, available].fillna(0)
        y_train = y.loc[train_idx]
        y_valid = y.loc[valid_idx]
        y_test = y.loc[test_idx]

        model = train_success_model(model_name, X_train, y_train, X_valid, y_valid)
        y_prob = predict_success_proba(model, X_test)

        metrics = compute_classification_metrics(y_test.values, y_prob)
        metrics["analysis"] = analysis_name
        metrics["config"] = config_name
        metrics["n_features"] = len(available)
        results.append(metrics)
        print(
            f"  [{analysis_name}] {config_name} ({len(available)} features): "
            f"AUROC={metrics['auroc']:.4f}  LogLoss={metrics['log_loss']:.4f}"
        )

    return pd.DataFrame(results)


def run_ablation_progressive_subset(
    X_full: pd.DataFrame,
    y: pd.Series,
    is_progressive: pd.Series,
    event_feature_cols: list[str],
    train_idx: pd.Index,
    valid_idx: pd.Index,
    test_idx: pd.Index,
    model_name: str = "lightgbm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run ablation on both full dataset and progressive subset.

    Returns (full_results, progressive_results).
    """
    print("Ablation on full dataset:")
    full_results = run_ablation(
        X_full, y, event_feature_cols, train_idx, valid_idx, test_idx, model_name,
    )

    # Progressive subset
    prog_mask = is_progressive
    prog_train = train_idx.intersection(X_full[prog_mask].index)
    prog_valid = valid_idx.intersection(X_full[prog_mask].index)
    prog_test = test_idx.intersection(X_full[prog_mask].index)

    print("\nAblation on progressive pass subset:")
    prog_results = run_ablation(
        X_full, y, event_feature_cols, prog_train, prog_valid, prog_test, model_name,
    )

    return full_results, prog_results


def run_order_robust_ablation_suite(
    X_full: pd.DataFrame,
    y: pd.Series,
    is_progressive: pd.Series,
    event_feature_cols: list[str],
    train_idx: pd.Index,
    valid_idx: pd.Index,
    test_idx: pd.Index,
    model_name: str = "lightgbm",
) -> dict[str, pd.DataFrame]:
    """Run cumulative, isolated, and leave-one-out ablations."""
    prog_mask = is_progressive
    prog_train = train_idx.intersection(X_full[prog_mask].index)
    prog_valid = valid_idx.intersection(X_full[prog_mask].index)
    prog_test = test_idx.intersection(X_full[prog_mask].index)

    cumulative_sets = get_cumulative_feature_sets(event_feature_cols)
    isolated_sets = get_isolated_feature_sets(event_feature_cols)
    loo_sets = get_leave_one_out_feature_sets(event_feature_cols)

    results = {}
    print("Ablation on full dataset:")
    results["full_cumulative"] = run_feature_set_evaluation(
        X_full, y, cumulative_sets, train_idx, valid_idx, test_idx, model_name, "cumulative"
    )
    results["full_isolated"] = run_feature_set_evaluation(
        X_full, y, isolated_sets, train_idx, valid_idx, test_idx, model_name, "isolated"
    )
    results["full_leave_one_out"] = run_feature_set_evaluation(
        X_full, y, loo_sets, train_idx, valid_idx, test_idx, model_name, "leave_one_out"
    )

    print("\nAblation on progressive pass subset:")
    results["prog_cumulative"] = run_feature_set_evaluation(
        X_full, y, cumulative_sets, prog_train, prog_valid, prog_test, model_name, "cumulative"
    )
    results["prog_isolated"] = run_feature_set_evaluation(
        X_full, y, isolated_sets, prog_train, prog_valid, prog_test, model_name, "isolated"
    )
    results["prog_leave_one_out"] = run_feature_set_evaluation(
        X_full, y, loo_sets, prog_train, prog_valid, prog_test, model_name, "leave_one_out"
    )

    return results

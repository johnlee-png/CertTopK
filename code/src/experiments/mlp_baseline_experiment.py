"""Experiment 22 (Phase 16): a seventh accuracy-axis test -- does a different
tabular model family, not just a different freeze-frame architecture, close
the gap to the LightGBM hand-crafted-feature ceiling?

Sec. 4.1's six accuracy-axis experiments all pair a learned freeze-frame
encoder (or a data-scaling manipulation) with LightGBM as the downstream
tabular learner. This tests the remaining alternative explanation flagged in
Limitations: perhaps LightGBM itself, not the feature representation, is the
bottleneck. We train a shallow MLP directly on the same 22 hand-crafted
features (no learned freeze-frame representation at all) and compare.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.splits import match_level_split
from src.models.pass_success import train_success_model, predict_success_proba
from src.evaluation.metrics import bootstrap_metric

EVENT_ONLY_COLS = [
    "start_x", "start_y", "end_x", "end_y", "pass_length", "pass_angle", "forward_distance",
    "start_x_norm", "start_y_norm", "end_x_norm", "end_y_norm", "start_dist_to_goal",
    "end_dist_to_goal", "minute", "period", "under_pressure", "is_ground_pass", "is_high_pass",
    "is_low_pass", "is_right_foot", "start_in_own_third", "start_in_mid_third",
    "start_in_final_third", "end_in_final_third", "end_in_penalty_area", "is_cross", "is_switch",
]


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


def run_mlp_baseline_experiment(
    processed_dir: str = "data/processed",
    output_dir: str = "experiments/results",
    seed: int = 42,
) -> dict:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed artifacts...")
    passes = pd.read_parquet(processed_dir / "passes_with_features.parquet", columns=["match_id"])
    X_full = pd.read_parquet(processed_dir / "X_full.parquet")
    y = pd.read_parquet(processed_dir / "y_labels.parquet")["pass_success"]

    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)
    train_idx, valid_idx, test_idx = train_df.index, valid_df.index, test_df.index
    y_test = y.loc[test_idx].values
    match_ids_test = passes.loc[test_idx, "match_id"].values

    event_cols = [c for c in EVENT_ONLY_COLS if c in X_full.columns]
    all_cols = list(X_full.columns)

    results = {"description": "Shallow MLP on the same hand-crafted features, "
                               "as a 7th accuracy-axis (model-family) test"}

    for label, cols in [("E3_mlp_event_only", event_cols), ("E6_mlp_event_360", all_cols)]:
        print(f"\nTraining MLP ({label}, {len(cols)} features)...")
        model = train_success_model(
            "mlp",
            X_full.loc[train_idx, cols].fillna(0), y.loc[train_idx],
            X_full.loc[valid_idx, cols].fillna(0), y.loc[valid_idx],
        )
        p_test = predict_success_proba(model, X_full.loc[test_idx, cols].fillna(0))
        pt, lo, hi = bootstrap_metric(y_test, p_test, _auroc, match_ids=match_ids_test)
        print(f"  {label}: AUROC = {pt:.4f} [{lo:.4f}, {hi:.4f}]")
        results[label] = {"auroc": pt, "ci_low": lo, "ci_high": hi, "n_features": len(cols)}

    print("\nReference (LightGBM, from prior experiments):")
    print("  E3_lightgbm: AUROC = 0.9165 [0.9093, 0.9232]")
    print("  E6_lightgbm: AUROC = 0.9446 [0.9398, 0.9493]")
    results["reference_lightgbm"] = {
        "E3": {"auroc": 0.9164751568528255, "ci_low": 0.9093371015841696, "ci_high": 0.9232448142878678},
        "E6": {"auroc": 0.9445673409334321, "ci_low": 0.9398407098184274, "ci_high": 0.9492571656877485},
    }

    results["split_manifest"] = {
        "seed": seed,
        "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
        "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
        "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
    }

    output_path = output_dir / "exp22_mlp_baseline.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_mlp_baseline_experiment()

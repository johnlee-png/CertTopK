"""Experiment 11: occlusion-aware imputation vs implicit zero-imputation.

Empirically tests Theorem 3 (Monotone Downward Bias): pressure counts
computed only from *visible* freeze-frame players should systematically
undercount true pressure. We correct them using compute_occlusion_correction
(uniform-density estimate over the camera's invisible region) and compare
(a) the corrected vs raw pressure counts directly, and (b) pass-success AUROC
with corrected vs raw features, on the identical match-level split.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.splits import match_level_split
from src.features.occlusion_imputation import add_occlusion_corrected_features
from src.models.pass_success import train_success_model, predict_success_proba
from src.evaluation.metrics import compute_classification_metrics, bootstrap_metric


def _json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def _bootstrap_auroc(y_test, y_prob, match_ids_test):
    pt, lo, hi = bootstrap_metric(y_test, y_prob, roc_auc_score, n_bootstrap=1000, match_ids=match_ids_test)
    return {"point": pt, "ci_low": lo, "ci_high": hi}


def run_occlusion_imputation_experiment(
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
        columns=["match_id", "freeze_frame", "visible_area", "start_x", "start_y", "end_x", "end_y"],
    )
    X_full = pd.read_parquet(processed_dir / "X_full.parquet")
    y = pd.read_parquet(processed_dir / "y_labels.parquet")["pass_success"]

    print("Computing occlusion-aware corrections (this parses a polygon per pass)...")
    X_imputed, corrections = add_occlusion_corrected_features(passes, X_full)

    # ===== Direct empirical test of Theorem 3's direction =====
    print("\nVerifying Theorem 3 (Monotone Downward Bias) empirically...")
    raw_receiver_3m = X_full["receiver_opp_3m"]
    imputed_receiver_3m = X_imputed["receiver_opp_3m"]
    mean_raw = float(raw_receiver_3m.mean())
    mean_imputed = float(imputed_receiver_3m.mean())
    frac_increased = float((imputed_receiver_3m > raw_receiver_3m + 1e-9).mean())
    corr_with_completeness = float(
        pd.concat([corrections["receiver_opp_3m_correction"], X_full["context_completeness"]], axis=1)
        .corr().iloc[0, 1]
    )
    print(f"  mean raw receiver_opp_3m: {mean_raw:.4f}")
    print(f"  mean imputed receiver_opp_3m: {mean_imputed:.4f}  (theorem predicts imputed >= raw, always)")
    print(f"  fraction of passes where correction > 0: {frac_increased:.4f}")
    print(f"  corr(correction size, context_completeness): {corr_with_completeness:.4f} (theorem predicts negative)")

    theorem3_check = {
        "mean_raw_receiver_opp_3m": mean_raw,
        "mean_imputed_receiver_opp_3m": mean_imputed,
        "fraction_passes_corrected_upward": frac_increased,
        "correlation_correction_vs_completeness": corr_with_completeness,
    }

    # ===== AUROC comparison: raw (zero-imputed) vs occlusion-corrected =====
    print("\nSplitting (must match the seed used to build the processed artifacts)...")
    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)
    train_idx, valid_idx, test_idx = train_df.index, valid_df.index, test_df.index
    match_ids_test = passes.loc[test_idx, "match_id"].values

    print("\nTraining E6 (raw, implicit zero-imputation)...")
    model_raw = train_success_model(
        "lightgbm",
        X_full.loc[train_idx].fillna(0), y.loc[train_idx],
        X_full.loc[valid_idx].fillna(0), y.loc[valid_idx],
    )
    prob_raw = predict_success_proba(model_raw, X_full.loc[test_idx].fillna(0))
    metrics_raw = compute_classification_metrics(y.loc[test_idx].values, prob_raw)
    metrics_raw["auroc_bootstrap"] = _bootstrap_auroc(y.loc[test_idx].values, prob_raw, match_ids_test)
    print(f"  AUROC={metrics_raw['auroc']:.4f}")

    print("\nTraining E6-imputed (occlusion-aware correction)...")
    model_imputed = train_success_model(
        "lightgbm",
        X_imputed.loc[train_idx].fillna(0), y.loc[train_idx],
        X_imputed.loc[valid_idx].fillna(0), y.loc[valid_idx],
    )
    prob_imputed = predict_success_proba(model_imputed, X_imputed.loc[test_idx].fillna(0))
    metrics_imputed = compute_classification_metrics(y.loc[test_idx].values, prob_imputed)
    metrics_imputed["auroc_bootstrap"] = _bootstrap_auroc(y.loc[test_idx].values, prob_imputed, match_ids_test)
    print(f"  AUROC={metrics_imputed['auroc']:.4f}")

    results = {
        "description": "Occlusion-aware imputation (uniform-density estimate over camera-invisible region) vs raw zero-imputed 360 pressure counts",
        "theorem3_empirical_check": theorem3_check,
        "E6_raw": metrics_raw,
        "E6_occlusion_imputed": metrics_imputed,
        "split_manifest": {
            "seed": seed,
            "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
            "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
            "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
        },
    }

    output_path = output_dir / "exp11_occlusion_imputation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_occlusion_imputation_experiment()

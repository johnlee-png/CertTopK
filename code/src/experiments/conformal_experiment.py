"""Experiment 8: split conformal prediction sets for the pass-success classifier.

Calibrates on the validation split (not used to fit model weights, only for early
stopping), then evaluates marginal coverage and set-size efficiency on the test
split with match-block bootstrap CIs. Compares Event-only (E3) vs Event+360 (E6).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import match_level_split
from src.models.pass_success import train_success_model, predict_success_proba
from src.evaluation.metrics import bootstrap_metric
from src.evaluation.conformal import (
    calibrate_conformal_threshold,
    predict_conformal_sets,
    empirical_coverage,
    average_set_size,
    singleton_accuracy,
)

ALPHAS = [0.05, 0.1, 0.2]


def _json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def _evaluate_config(
    tag: str,
    desc: str,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx,
    valid_idx,
    test_idx,
    match_ids_test: np.ndarray,
) -> dict:
    print(f"\nTraining {tag}: {desc}...")
    model = train_success_model(
        "lightgbm",
        X.loc[train_idx].fillna(0), y.loc[train_idx],
        X.loc[valid_idx].fillna(0), y.loc[valid_idx],
    )
    p_valid = predict_success_proba(model, X.loc[valid_idx].fillna(0))
    p_test = predict_success_proba(model, X.loc[test_idx].fillna(0))
    y_valid = y.loc[valid_idx].values
    y_test = y.loc[test_idx].values

    per_alpha = {}
    for alpha in ALPHAS:
        q_hat = calibrate_conformal_threshold(y_valid, p_valid, alpha=alpha)
        pred_sets = predict_conformal_sets(p_test, q_hat)

        cov_pt, cov_lo, cov_hi = bootstrap_metric(
            y_test, pred_sets, empirical_coverage, n_bootstrap=1000, match_ids=match_ids_test,
        )
        size_pt, size_lo, size_hi = bootstrap_metric(
            y_test, pred_sets, average_set_size, n_bootstrap=1000, match_ids=match_ids_test,
        )
        singleton_acc = singleton_accuracy(y_test, pred_sets)
        singleton_frac = float((pred_sets.sum(axis=1) == 1).mean())

        per_alpha[str(alpha)] = {
            "target_coverage": 1 - alpha,
            "q_hat": q_hat,
            "coverage": {"point": cov_pt, "ci_low": cov_lo, "ci_high": cov_hi},
            "avg_set_size": {"point": size_pt, "ci_low": size_lo, "ci_high": size_hi},
            "singleton_fraction": singleton_frac,
            "singleton_accuracy": singleton_acc,
        }
        print(
            f"  alpha={alpha}: target_cov={1 - alpha:.2f}  "
            f"coverage={cov_pt:.4f} [{cov_lo:.4f},{cov_hi:.4f}]  "
            f"avg_set_size={size_pt:.4f}  singleton_acc={singleton_acc:.4f}"
        )

    return {"description": desc, "by_alpha": per_alpha}


def run_conformal_experiment(
    processed_dir: str = "data/processed",
    output_dir: str = "experiments/results",
    seed: int = 42,
) -> dict:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed artifacts...")
    passes = pd.read_parquet(processed_dir / "passes_with_features.parquet", columns=["match_id"])
    X_event = pd.read_parquet(processed_dir / "X_event.parquet")
    X_full = pd.read_parquet(processed_dir / "X_full.parquet")
    y = pd.read_parquet(processed_dir / "y_labels.parquet")["pass_success"]

    print("Splitting (must match the seed used to build the processed artifacts)...")
    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)
    train_idx, valid_idx, test_idx = train_df.index, valid_df.index, test_df.index
    match_ids_test = passes.loc[test_idx, "match_id"].values

    results = {
        "description": "Split conformal (LAC) prediction sets for pass-success classifiers, calibrated on validation split",
        "E3_event_only": _evaluate_config(
            "E3", "Event-only LGBM", X_event, y, train_idx, valid_idx, test_idx, match_ids_test,
        ),
        "E6_event_360": _evaluate_config(
            "E6", "Event+360 LGBM", X_full, y, train_idx, valid_idx, test_idx, match_ids_test,
        ),
        "split_manifest": {
            "seed": seed,
            "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
            "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
            "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
        },
    }

    output_path = output_dir / "exp8_conformal.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_conformal_experiment()

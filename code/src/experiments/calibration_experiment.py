"""Experiment 17 (Phase 11): probability calibration (Platt scaling / isotonic regression).

Theorem 5's corollary identifies the 85.4% class imbalance as a known cause of upward
probability bias in LightGBM, and names post-hoc calibration (Platt scaling or isotonic
regression) as a high-priority extension. This implements and evaluates it on the real
E6 (event+360) success model: does calibration improve ECE without hurting AUROC, and
does it meaningfully re-rank utility (as Theorem 5 predicts for high-R+C aggressive passes)?
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from src.data.splits import match_level_split
from src.models.pass_success import train_success_model, predict_success_proba
from src.models.reward_model import train_reward_model, predict_reward
from src.models.utility import compute_failure_cost, compute_utility
from src.evaluation.metrics import bootstrap_metric, expected_calibration_error, evaluate_utility_deciles


def _json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def _to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def run_calibration_experiment(
    processed_dir: str = "data/processed",
    output_dir: str = "experiments/results",
    seed: int = 42,
) -> dict:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed artifacts...")
    passes = pd.read_parquet(processed_dir / "passes_with_features.parquet")
    X_full = pd.read_parquet(processed_dir / "X_full.parquet")
    y = pd.read_parquet(processed_dir / "y_labels.parquet")["pass_success"]

    passes["realized_delta_xt"] = passes["delta_xt"] * passes["pass_success"]
    passes["realized_progressive"] = (
        passes["is_progressive"] & passes["pass_success"].astype(bool)
    ).astype(int)
    passes["realized_final_third_entry"] = (
        (passes["end_x"] >= 80) & passes["pass_success"].astype(bool)
    ).astype(int)

    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)
    train_idx, valid_idx, test_idx = train_df.index, valid_df.index, test_df.index
    print(f"Split: {len(train_df)} train / {len(valid_df)} valid / {len(test_df)} test matches")

    print("\nTraining E6 (event+360 LightGBM) success model...")
    success_model = train_success_model(
        "lightgbm",
        X_full.loc[train_idx].fillna(0), y.loc[train_idx],
        X_full.loc[valid_idx].fillna(0), y.loc[valid_idx],
    )
    p_raw_valid = predict_success_proba(success_model, X_full.loc[valid_idx].fillna(0))
    p_raw_test = predict_success_proba(success_model, X_full.loc[test_idx].fillna(0))
    y_valid, y_test = y.loc[valid_idx].values, y.loc[test_idx].values
    match_ids_test = passes.loc[test_idx, "match_id"].values

    print("\nFitting Platt scaling (logistic regression on logit(p)) and isotonic regression")
    print("on the validation set, evaluating on the held-out test set...")
    platt = LogisticRegression()
    platt.fit(_to_logit(p_raw_valid).reshape(-1, 1), y_valid)
    p_platt_test = platt.predict_proba(_to_logit(p_raw_test).reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw_valid, y_valid)
    p_iso_test = iso.predict(p_raw_test)

    auroc_raw = roc_auc_score(y_test, p_raw_test)
    auroc_platt = roc_auc_score(y_test, p_platt_test)
    auroc_iso = roc_auc_score(y_test, p_iso_test)
    print(f"\nAUROC: raw={auroc_raw:.4f}, platt={auroc_platt:.4f}, isotonic={auroc_iso:.4f}"
          f"  (calibration is a monotone-ish rescaling, so AUROC should barely move)")

    ece_raw_pt, ece_raw_lo, ece_raw_hi = bootstrap_metric(
        y_test, p_raw_test, expected_calibration_error, match_ids=match_ids_test)
    ece_platt_pt, ece_platt_lo, ece_platt_hi = bootstrap_metric(
        y_test, p_platt_test, expected_calibration_error, match_ids=match_ids_test)
    ece_iso_pt, ece_iso_lo, ece_iso_hi = bootstrap_metric(
        y_test, p_iso_test, expected_calibration_error, match_ids=match_ids_test)
    print(f"\nECE (match-block bootstrap 95% CI):")
    print(f"  raw:      {ece_raw_pt:.4f} [{ece_raw_lo:.4f}, {ece_raw_hi:.4f}]")
    print(f"  platt:    {ece_platt_pt:.4f} [{ece_platt_lo:.4f}, {ece_platt_hi:.4f}]")
    print(f"  isotonic: {ece_iso_pt:.4f} [{ece_iso_lo:.4f}, {ece_iso_hi:.4f}]")

    print("\nTraining reward (R2-equivalent) model for utility scoring...")
    success_mask = y == 1
    r_train = train_idx.intersection(passes[success_mask].index)
    r_valid = valid_idx.intersection(passes[success_mask].index)
    reward_model = train_reward_model(
        "lightgbm_regressor",
        X_full.loc[r_train].fillna(0), passes.loc[r_train, "delta_xt"],
        X_full.loc[r_valid].fillna(0), passes.loc[r_valid, "delta_xt"],
    )
    r_pred_test = predict_reward(reward_model, X_full.loc[test_idx].fillna(0))
    c_fail_test = compute_failure_cost(passes.loc[test_idx], method="location_based")

    U_raw = compute_utility(p_raw_test, r_pred_test, c_fail_test)
    U_platt = compute_utility(p_platt_test, r_pred_test, c_fail_test)
    distortion = U_raw - U_platt  # exactly (p_raw-p_platt)*(R+C) since U is linear in P

    print("\nTheorem 5 check: does raw-vs-calibrated utility distortion concentrate in")
    print("high (R+C) 'aggressive' passes, as the corollary predicts?")
    rc_sum = r_pred_test + c_fail_test
    spearman_distortion_rc = spearmanr(distortion, rc_sum).correlation
    print(f"  Spearman corr(distortion, R+C) = {spearman_distortion_rc:.4f}")

    eval_raw = passes.loc[test_idx].copy()
    eval_raw["utility_score"] = U_raw
    deciles_raw = evaluate_utility_deciles(eval_raw)

    eval_platt = passes.loc[test_idx].copy()
    eval_platt["utility_score"] = U_platt
    deciles_platt = evaluate_utility_deciles(eval_platt)

    print("\n  Raw (uncalibrated) utility deciles:")
    print(deciles_raw.to_string(index=False))
    print("\n  Platt-calibrated utility deciles:")
    print(deciles_platt.to_string(index=False))

    rank_raw = pd.Series(U_raw, index=test_idx).rank(ascending=False)
    rank_platt = pd.Series(U_platt, index=test_idx).rank(ascending=False)
    rank_shift = (rank_raw - rank_platt).abs()
    top_1000_raw = set(rank_raw.nsmallest(1000).index)
    top_1000_platt = set(rank_platt.nsmallest(1000).index)
    overlap = len(top_1000_raw & top_1000_platt) / 1000
    print(f"\nTop-1000-by-utility overlap (raw vs Platt-calibrated): {overlap:.1%}")
    print(f"Mean |rank shift|: {rank_shift.mean():.1f} (out of {len(test_idx)} test passes)")

    results = {
        "description": "Platt scaling / isotonic regression calibration of the E6 success model, "
                        "and its downstream effect on utility ranking (Theorem 5 corollary)",
        "auroc": {"raw": auroc_raw, "platt": auroc_platt, "isotonic": auroc_iso},
        "ece": {
            "raw": {"point": ece_raw_pt, "ci_low": ece_raw_lo, "ci_high": ece_raw_hi},
            "platt": {"point": ece_platt_pt, "ci_low": ece_platt_lo, "ci_high": ece_platt_hi},
            "isotonic": {"point": ece_iso_pt, "ci_low": ece_iso_lo, "ci_high": ece_iso_hi},
        },
        "theorem5_check": {
            "spearman_corr_distortion_vs_R_plus_C": float(spearman_distortion_rc),
        },
        "utility_deciles_raw": deciles_raw.to_dict(orient="records"),
        "utility_deciles_platt": deciles_platt.to_dict(orient="records"),
        "ranking_stability": {
            "top_1000_overlap_fraction": overlap,
            "mean_abs_rank_shift": float(rank_shift.mean()),
            "n_test_passes": len(test_idx),
        },
        "split_manifest": {
            "seed": seed,
            "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
            "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
            "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
        },
    }

    output_path = output_dir / "exp17_calibration.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_calibration_experiment()

"""Experiment 19 (Phase 13): real external held-out competition validation.

Supp. Theorem S2's Remark claims "we observe an AUROC drop of approximately 0.012 when
evaluating our World-Cup-trained model on Bundesliga test matches", but no script or
result file backing this claim could be found (see NUMBER_AUDIT_SUMMARY.md). This runs
the actual experiment: train E6 (event+360 LightGBM) on all non-Bundesliga matches,
evaluate on an in-domain held-out test (other competitions) vs. the fully external
Bundesliga competition (all 34 matches, Bayer Leverkusen's 2023/24 StatsBomb 360 season).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import match_level_split
from src.models.pass_success import train_success_model, predict_success_proba
from src.evaluation.metrics import bootstrap_metric
from sklearn.metrics import roc_auc_score

BUNDESLIGA_MATCH_IDS = {
    3895292, 3895320, 3895158, 3895107, 3895340, 3895286, 3895302, 3895333, 3895348,
    3895220, 3895250, 3895266, 3895275, 3895180, 3895134, 3895121, 3895074, 3895139,
    3895086, 3895309, 3895258, 3895244, 3895232, 3895210, 3895202, 3895194, 3895182,
    3895167, 3895153, 3895113, 3895095, 3895067, 3895060, 3895052,
}


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


def run_bundesliga_holdout_experiment(
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

    is_bundesliga = passes["match_id"].isin(BUNDESLIGA_MATCH_IDS)
    n_bundesliga_matches = passes.loc[is_bundesliga, "match_id"].nunique()
    print(f"Bundesliga (Bayer Leverkusen 2023/24): {n_bundesliga_matches} matches, "
          f"{is_bundesliga.sum()} passes")
    assert n_bundesliga_matches == 34, f"expected 34 Bundesliga matches, found {n_bundesliga_matches}"

    non_bundesliga = passes[~is_bundesliga]
    print(f"Non-Bundesliga pool: {non_bundesliga['match_id'].nunique()} matches, {len(non_bundesliga)} passes")

    train_df, valid_df, indomain_test_df = match_level_split(non_bundesliga, random_state=seed)
    train_idx, valid_idx, indomain_test_idx = train_df.index, valid_df.index, indomain_test_df.index
    bundesliga_idx = passes[is_bundesliga].index

    print("\nTraining E6 (event+360 LightGBM) on non-Bundesliga train/valid only...")
    model = train_success_model(
        "lightgbm",
        X_full.loc[train_idx].fillna(0), y.loc[train_idx],
        X_full.loc[valid_idx].fillna(0), y.loc[valid_idx],
    )

    p_indomain = predict_success_proba(model, X_full.loc[indomain_test_idx].fillna(0))
    p_bundesliga = predict_success_proba(model, X_full.loc[bundesliga_idx].fillna(0))

    y_indomain = y.loc[indomain_test_idx].values
    y_bundesliga = y.loc[bundesliga_idx].values
    match_ids_indomain = passes.loc[indomain_test_idx, "match_id"].values
    match_ids_bundesliga = passes.loc[bundesliga_idx, "match_id"].values

    auroc_indomain_pt, auroc_indomain_lo, auroc_indomain_hi = bootstrap_metric(
        y_indomain, p_indomain, _auroc, match_ids=match_ids_indomain)
    auroc_bundesliga_pt, auroc_bundesliga_lo, auroc_bundesliga_hi = bootstrap_metric(
        y_bundesliga, p_bundesliga, _auroc, match_ids=match_ids_bundesliga)

    drop = auroc_indomain_pt - auroc_bundesliga_pt

    print(f"\nIn-domain held-out test (other competitions, {indomain_test_df['match_id'].nunique()} matches, "
          f"{len(indomain_test_idx)} passes):")
    print(f"  AUROC = {auroc_indomain_pt:.4f} [{auroc_indomain_lo:.4f}, {auroc_indomain_hi:.4f}]")
    print(f"\nBundesliga external held-out (34 matches, {len(bundesliga_idx)} passes):")
    print(f"  AUROC = {auroc_bundesliga_pt:.4f} [{auroc_bundesliga_lo:.4f}, {auroc_bundesliga_hi:.4f}]")
    print(f"\nAUROC drop (in-domain -> Bundesliga): {drop:.4f}")
    print(f"Paper's unverified claim (Supp. Thm. S2 Remark): approximately 0.012")
    print(f"CIs overlap: {auroc_bundesliga_lo <= auroc_indomain_hi}")

    results = {
        "description": "Real external held-out competition validation: train on all non-Bundesliga "
                        "matches, evaluate in-domain (other competitions) vs. Bundesliga (fully held out).",
        "n_bundesliga_matches": n_bundesliga_matches,
        "n_bundesliga_passes": int(is_bundesliga.sum()),
        "in_domain_test": {
            "n_matches": int(indomain_test_df["match_id"].nunique()),
            "n_passes": len(indomain_test_idx),
            "auroc": auroc_indomain_pt, "ci_low": auroc_indomain_lo, "ci_high": auroc_indomain_hi,
        },
        "bundesliga_test": {
            "n_matches": 34,
            "n_passes": len(bundesliga_idx),
            "auroc": auroc_bundesliga_pt, "ci_low": auroc_bundesliga_lo, "ci_high": auroc_bundesliga_hi,
        },
        "auroc_drop": drop,
        "paper_unverified_claim": 0.012,
        "split_manifest": {
            "seed": seed,
            "bundesliga_match_ids": sorted(BUNDESLIGA_MATCH_IDS),
            "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
            "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
            "indomain_test_match_ids": sorted(indomain_test_df["match_id"].unique().tolist()),
        },
    }

    output_path = output_dir / "exp19_bundesliga_holdout.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_bundesliga_holdout_experiment()

"""Experiment 10: Wyscout (Pappalardo 2019) pretraining -> StatsBomb fine-tuning.

Wyscout has no 360/freeze-frame data, so this only targets the event-only pathway.
Because Wyscout's schema doesn't map 1:1 onto all 27 StatsBomb event-only features
(different tag system instead of named outcome/height/body-part fields), both the
"no pretrain" baseline and the pretrained model are restricted to the smaller
shared feature subset defined in scripts/convert_wyscout_to_statsbomb_coords.py,
so the comparison isolates the effect of pretraining rather than feature-set size.

LightGBM's `init_model` continues boosting from an existing booster, which is the
standard way to "fine-tune" a gradient-boosted tree model (there's no gradient
weight-update fine-tuning like a neural net has).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.data.splits import match_level_split
from src.evaluation.metrics import compute_classification_metrics, bootstrap_metric
from scripts.convert_wyscout_to_statsbomb_coords import SHARED_FEATURE_COLUMNS

LGBM_PARAMS = dict(n_estimators=500, num_leaves=63, learning_rate=0.05, random_state=42, verbose=-1)


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


def run_wyscout_pretrain_experiment(
    processed_dir: str = "data/processed",
    output_dir: str = "experiments/results",
    seed: int = 42,
) -> dict:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading StatsBomb processed artifacts...")
    passes = pd.read_parquet(processed_dir / "passes_with_features.parquet", columns=["match_id"])
    X_event = pd.read_parquet(processed_dir / "X_event.parquet", columns=SHARED_FEATURE_COLUMNS)
    y = pd.read_parquet(processed_dir / "y_labels.parquet")["pass_success"]

    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)
    train_idx, valid_idx, test_idx = train_df.index, valid_df.index, test_df.index
    match_ids_test = passes.loc[test_idx, "match_id"].values

    X_train = X_event.loc[train_idx].fillna(0)
    X_valid = X_event.loc[valid_idx].fillna(0)
    X_test = X_event.loc[test_idx].fillna(0)
    y_train, y_valid, y_test = y.loc[train_idx], y.loc[valid_idx], y.loc[test_idx]

    results = {}

    # ===== Baseline: StatsBomb-only, shared feature subset, no pretraining =====
    print("\nTraining baseline (StatsBomb shared-feature-subset only, no pretrain)...")
    baseline = LGBMClassifier(**LGBM_PARAMS)
    baseline.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    baseline_prob = baseline.predict_proba(X_test)[:, 1]
    baseline_metrics = compute_classification_metrics(y_test.values, baseline_prob)
    baseline_metrics["auroc_bootstrap"] = _bootstrap_auroc(y_test.values, baseline_prob, match_ids_test)
    results["statsbomb_shared_only_no_pretrain"] = baseline_metrics
    print(f"  AUROC={baseline_metrics['auroc']:.4f}")

    # ===== Pretrain on Wyscout =====
    print("\nLoading Wyscout data and pretraining...")
    wyscout = pd.read_parquet(processed_dir / "wyscout_passes.parquet")
    X_wy = wyscout[SHARED_FEATURE_COLUMNS].fillna(0)
    y_wy = wyscout["pass_success"]
    print(f"  Wyscout passes: {len(wyscout)} ({y_wy.mean() * 100:.1f}% success)")

    pretrained = LGBMClassifier(**LGBM_PARAMS)
    pretrained.fit(X_wy, y_wy)

    zero_shot_prob = pretrained.predict_proba(X_test)[:, 1]
    zero_shot_metrics = compute_classification_metrics(y_test.values, zero_shot_prob)
    zero_shot_metrics["auroc_bootstrap"] = _bootstrap_auroc(y_test.values, zero_shot_prob, match_ids_test)
    results["wyscout_zero_shot_no_finetune"] = zero_shot_metrics
    print(f"  Wyscout-only zero-shot AUROC on StatsBomb test={zero_shot_metrics['auroc']:.4f}")

    # ===== Fine-tune: continue boosting from the Wyscout-pretrained model =====
    print("\nFine-tuning on StatsBomb (continuing boosting from the Wyscout model)...")
    finetuned = LGBMClassifier(**LGBM_PARAMS)
    finetuned.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], init_model=pretrained.booster_)
    finetuned_prob = finetuned.predict_proba(X_test)[:, 1]
    finetuned_metrics = compute_classification_metrics(y_test.values, finetuned_prob)
    finetuned_metrics["auroc_bootstrap"] = _bootstrap_auroc(y_test.values, finetuned_prob, match_ids_test)
    results["wyscout_pretrained_finetuned"] = finetuned_metrics
    print(f"  AUROC={finetuned_metrics['auroc']:.4f}")

    results["dataset_info"] = {
        "wyscout_n_passes": int(len(wyscout)),
        "wyscout_success_rate": float(y_wy.mean()),
        "shared_feature_columns": SHARED_FEATURE_COLUMNS,
    }
    results["split_manifest"] = {
        "seed": seed,
        "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
        "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
        "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
    }

    output_path = output_dir / "exp10_wyscout_pretrain.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_wyscout_pretrain_experiment()

"""Experiment 13 (Phase 7): Sequence Transformer over the previous 3-5 actions
vs the static-context baselines.

Compares:
  E3   Event-only LGBM (baseline, no context at all)
  E6   Event+360 LGBM (hand-crafted 22 context features, existing baseline)
  E11  Event + pre-pass action-sequence embedding (end-to-end neural net)
  E12  Event + sequence embedding (LGBM hybrid)
  E12b Event + hand-crafted 360 + sequence embedding (LGBM hybrid)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.data.splits import match_level_split
from src.models.pass_success import train_success_model, predict_success_proba
from src.features.sequence_context import build_sequence_tensors
from src.models.sequence_transformer import (
    train_sequence_model, predict_sequence_model, extract_sequence_embeddings,
)
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


def run_sequence_transformer_experiment(
    processed_dir: str = "data/processed",
    output_dir: str = "experiments/results",
    seed: int = 42,
) -> dict:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading processed artifacts...")
    passes = pd.read_parquet(
        processed_dir / "passes_with_features.parquet",
        columns=["match_id", "index", "period", "minute", "second", "team_name"],
    )
    X_event = pd.read_parquet(processed_dir / "X_event.parquet")
    X_full = pd.read_parquet(processed_dir / "X_full.parquet")
    y = pd.read_parquet(processed_dir / "y_labels.parquet")["pass_success"]
    all_events = pd.read_parquet(
        processed_dir / "events_360_all.parquet",
        columns=["match_id", "index", "period", "minute", "second", "team_name", "type_name", "location"],
    )

    print("Splitting (must match the seed used to build the processed artifacts)...")
    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)
    train_idx, valid_idx, test_idx = train_df.index, valid_df.index, test_df.index
    match_ids_test = passes.loc[test_idx, "match_id"].values

    results = {}

    print("\nTraining E3: Event-only LGBM...")
    e3_model = train_success_model(
        "lightgbm", X_event.loc[train_idx].fillna(0), y.loc[train_idx],
        X_event.loc[valid_idx].fillna(0), y.loc[valid_idx],
    )
    e3_prob = predict_success_proba(e3_model, X_event.loc[test_idx].fillna(0))
    e3_metrics = compute_classification_metrics(y.loc[test_idx].values, e3_prob)
    e3_metrics["auroc_bootstrap"] = _bootstrap_auroc(y.loc[test_idx].values, e3_prob, match_ids_test)
    results["E3_event_only"] = e3_metrics
    print(f"  AUROC={e3_metrics['auroc']:.4f}")

    print("\nTraining E6: Event+360 LGBM (hand-crafted)...")
    e6_model = train_success_model(
        "lightgbm", X_full.loc[train_idx].fillna(0), y.loc[train_idx],
        X_full.loc[valid_idx].fillna(0), y.loc[valid_idx],
    )
    e6_prob = predict_success_proba(e6_model, X_full.loc[test_idx].fillna(0))
    e6_metrics = compute_classification_metrics(y.loc[test_idx].values, e6_prob)
    e6_metrics["auroc_bootstrap"] = _bootstrap_auroc(y.loc[test_idx].values, e6_prob, match_ids_test)
    results["E6_event_360_handcrafted"] = e6_metrics
    print(f"  AUROC={e6_metrics['auroc']:.4f}")

    print("\nBuilding pre-pass action-sequence tensors (previous <=5 events per pass)...")
    seq_np, mask_np = build_sequence_tensors(passes, all_events)
    print(f"  seq tensor: {seq_np.shape}, mask: {mask_np.shape}, "
          f"avg history length: {mask_np.sum(axis=1).mean():.2f}")

    scaler = StandardScaler()
    event_train_np = scaler.fit_transform(X_event.loc[train_idx].fillna(0).values)
    event_valid_np = scaler.transform(X_event.loc[valid_idx].fillna(0).values)
    event_test_np = scaler.transform(X_event.loc[test_idx].fillna(0).values)

    def _slice(arr, idx):
        pos = passes.index.get_indexer(idx)
        return arr[pos]

    seq_train = torch.from_numpy(_slice(seq_np, train_idx))
    mask_train = torch.from_numpy(_slice(mask_np, train_idx))
    seq_valid = torch.from_numpy(_slice(seq_np, valid_idx))
    mask_valid = torch.from_numpy(_slice(mask_np, valid_idx))
    seq_test = torch.from_numpy(_slice(seq_np, test_idx))
    mask_test = torch.from_numpy(_slice(mask_np, test_idx))

    event_train_t = torch.from_numpy(event_train_np.astype(np.float32))
    event_valid_t = torch.from_numpy(event_valid_np.astype(np.float32))
    event_test_t = torch.from_numpy(event_test_np.astype(np.float32))
    y_train_t = torch.from_numpy(y.loc[train_idx].values.astype(np.float32))
    y_valid_t = torch.from_numpy(y.loc[valid_idx].values.astype(np.float32))

    print("\nTraining E11: Event + action-sequence embedding (end-to-end)...")
    seq_model = train_sequence_model(
        event_train_t, seq_train, mask_train, y_train_t,
        event_valid_t, seq_valid, mask_valid, y_valid_t,
        device=device,
    )
    e11_prob = predict_sequence_model(seq_model, event_test_t, seq_test, mask_test, device)
    e11_metrics = compute_classification_metrics(y.loc[test_idx].values, e11_prob)
    e11_metrics["auroc_bootstrap"] = _bootstrap_auroc(y.loc[test_idx].values, e11_prob, match_ids_test)
    results["E11_event_sequence"] = e11_metrics
    print(f"  AUROC={e11_metrics['auroc']:.4f}")

    print("\nExtracting sequence embeddings for the LGBM hybrid...")
    emb_train = extract_sequence_embeddings(seq_model, seq_train, mask_train, device)
    emb_valid = extract_sequence_embeddings(seq_model, seq_valid, mask_valid, device)
    emb_test = extract_sequence_embeddings(seq_model, seq_test, mask_test, device)
    emb_cols = [f"seq_emb_{i}" for i in range(emb_train.shape[1])]

    emb_train_df = pd.DataFrame(emb_train, index=train_idx, columns=emb_cols)
    emb_valid_df = pd.DataFrame(emb_valid, index=valid_idx, columns=emb_cols)
    emb_test_df = pd.DataFrame(emb_test, index=test_idx, columns=emb_cols)

    print("\nTraining E12: Event + sequence embedding (LGBM hybrid)...")
    X12_train = pd.concat([X_event.loc[train_idx], emb_train_df], axis=1)
    X12_valid = pd.concat([X_event.loc[valid_idx], emb_valid_df], axis=1)
    X12_test = pd.concat([X_event.loc[test_idx], emb_test_df], axis=1)
    e12_model = train_success_model("lightgbm", X12_train.fillna(0), y.loc[train_idx], X12_valid.fillna(0), y.loc[valid_idx])
    e12_prob = predict_success_proba(e12_model, X12_test.fillna(0))
    e12_metrics = compute_classification_metrics(y.loc[test_idx].values, e12_prob)
    e12_metrics["auroc_bootstrap"] = _bootstrap_auroc(y.loc[test_idx].values, e12_prob, match_ids_test)
    results["E12_event_plus_sequence_lgbm"] = e12_metrics
    print(f"  AUROC={e12_metrics['auroc']:.4f}")

    print("\nTraining E12b: Event+360(handcrafted) + sequence embedding (LGBM hybrid)...")
    X12b_train = pd.concat([X_full.loc[train_idx], emb_train_df], axis=1)
    X12b_valid = pd.concat([X_full.loc[valid_idx], emb_valid_df], axis=1)
    X12b_test = pd.concat([X_full.loc[test_idx], emb_test_df], axis=1)
    e12b_model = train_success_model("lightgbm", X12b_train.fillna(0), y.loc[train_idx], X12b_valid.fillna(0), y.loc[valid_idx])
    e12b_prob = predict_success_proba(e12b_model, X12b_test.fillna(0))
    e12b_metrics = compute_classification_metrics(y.loc[test_idx].values, e12b_prob)
    e12b_metrics["auroc_bootstrap"] = _bootstrap_auroc(y.loc[test_idx].values, e12b_prob, match_ids_test)
    results["E12b_event_360_plus_sequence_lgbm"] = e12b_metrics
    print(f"  AUROC={e12b_metrics['auroc']:.4f}")

    results["split_manifest"] = {
        "seed": seed,
        "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
        "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
        "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
    }

    output_path = output_dir / "exp13_sequence_transformer.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_sequence_transformer_experiment()

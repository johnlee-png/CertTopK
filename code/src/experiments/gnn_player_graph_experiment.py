"""Experiment 12 (Phase 6): GNN player-graph encoder vs hand-crafted 360 features
and vs the Set Transformer (Phase 3).

Compares:
  E3   Event-only LGBM (baseline, no 360 info)
  E6   Event+360 LGBM (hand-crafted 22 context features, existing baseline)
  E9   Event + GNN player-graph embedding (end-to-end neural net)
  E10  Event + GNN embedding (LGBM hybrid)
  E10b Event + hand-crafted 360 + GNN embedding (LGBM hybrid)
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
from src.models.set_transformer import freeze_frames_to_tensor
from src.models.gnn_player_graph import train_gnn_model, predict_gnn_model, extract_gnn_embeddings
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


def run_gnn_player_graph_experiment(
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
    passes = pd.read_parquet(processed_dir / "passes_with_features.parquet", columns=["match_id", "freeze_frame"])
    X_event = pd.read_parquet(processed_dir / "X_event.parquet")
    X_full = pd.read_parquet(processed_dir / "X_full.parquet")
    y = pd.read_parquet(processed_dir / "y_labels.parquet")["pass_success"]

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

    print("\nParsing freeze_frame into padded player-set/graph tensors...")
    players_np, mask_np = freeze_frames_to_tensor(passes)

    scaler = StandardScaler()
    event_train_np = scaler.fit_transform(X_event.loc[train_idx].fillna(0).values)
    event_valid_np = scaler.transform(X_event.loc[valid_idx].fillna(0).values)
    event_test_np = scaler.transform(X_event.loc[test_idx].fillna(0).values)

    def _slice(arr, idx):
        pos = passes.index.get_indexer(idx)
        return arr[pos]

    players_train = torch.from_numpy(_slice(players_np, train_idx))
    mask_train = torch.from_numpy(_slice(mask_np, train_idx))
    players_valid = torch.from_numpy(_slice(players_np, valid_idx))
    mask_valid = torch.from_numpy(_slice(mask_np, valid_idx))
    players_test = torch.from_numpy(_slice(players_np, test_idx))
    mask_test = torch.from_numpy(_slice(mask_np, test_idx))

    event_train_t = torch.from_numpy(event_train_np.astype(np.float32))
    event_valid_t = torch.from_numpy(event_valid_np.astype(np.float32))
    event_test_t = torch.from_numpy(event_test_np.astype(np.float32))
    y_train_t = torch.from_numpy(y.loc[train_idx].values.astype(np.float32))
    y_valid_t = torch.from_numpy(y.loc[valid_idx].values.astype(np.float32))

    print("\nTraining E9: Event + GNN player-graph embedding (end-to-end)...")
    gnn_model = train_gnn_model(
        event_train_t, players_train, mask_train, y_train_t,
        event_valid_t, players_valid, mask_valid, y_valid_t,
        device=device,
    )
    e9_prob = predict_gnn_model(gnn_model, event_test_t, players_test, mask_test, device)
    e9_metrics = compute_classification_metrics(y.loc[test_idx].values, e9_prob)
    e9_metrics["auroc_bootstrap"] = _bootstrap_auroc(y.loc[test_idx].values, e9_prob, match_ids_test)
    results["E9_event_gnn"] = e9_metrics
    print(f"  AUROC={e9_metrics['auroc']:.4f}")

    print("\nExtracting GNN embeddings for the LGBM hybrid...")
    emb_train = extract_gnn_embeddings(gnn_model, players_train, mask_train, device)
    emb_valid = extract_gnn_embeddings(gnn_model, players_valid, mask_valid, device)
    emb_test = extract_gnn_embeddings(gnn_model, players_test, mask_test, device)
    emb_cols = [f"gnn_emb_{i}" for i in range(emb_train.shape[1])]

    emb_train_df = pd.DataFrame(emb_train, index=train_idx, columns=emb_cols)
    emb_valid_df = pd.DataFrame(emb_valid, index=valid_idx, columns=emb_cols)
    emb_test_df = pd.DataFrame(emb_test, index=test_idx, columns=emb_cols)

    print("\nTraining E10: Event + GNN embedding (LGBM hybrid)...")
    X10_train = pd.concat([X_event.loc[train_idx], emb_train_df], axis=1)
    X10_valid = pd.concat([X_event.loc[valid_idx], emb_valid_df], axis=1)
    X10_test = pd.concat([X_event.loc[test_idx], emb_test_df], axis=1)
    e10_model = train_success_model("lightgbm", X10_train.fillna(0), y.loc[train_idx], X10_valid.fillna(0), y.loc[valid_idx])
    e10_prob = predict_success_proba(e10_model, X10_test.fillna(0))
    e10_metrics = compute_classification_metrics(y.loc[test_idx].values, e10_prob)
    e10_metrics["auroc_bootstrap"] = _bootstrap_auroc(y.loc[test_idx].values, e10_prob, match_ids_test)
    results["E10_event_plus_gnn_lgbm"] = e10_metrics
    print(f"  AUROC={e10_metrics['auroc']:.4f}")

    print("\nTraining E10b: Event+360(handcrafted) + GNN embedding (LGBM hybrid)...")
    X10b_train = pd.concat([X_full.loc[train_idx], emb_train_df], axis=1)
    X10b_valid = pd.concat([X_full.loc[valid_idx], emb_valid_df], axis=1)
    X10b_test = pd.concat([X_full.loc[test_idx], emb_test_df], axis=1)
    e10b_model = train_success_model("lightgbm", X10b_train.fillna(0), y.loc[train_idx], X10b_valid.fillna(0), y.loc[valid_idx])
    e10b_prob = predict_success_proba(e10b_model, X10b_test.fillna(0))
    e10b_metrics = compute_classification_metrics(y.loc[test_idx].values, e10b_prob)
    e10b_metrics["auroc_bootstrap"] = _bootstrap_auroc(y.loc[test_idx].values, e10b_prob, match_ids_test)
    results["E10b_event_360_plus_gnn_lgbm"] = e10b_metrics
    print(f"  AUROC={e10b_metrics['auroc']:.4f}")

    results["split_manifest"] = {
        "seed": seed,
        "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
        "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
        "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
    }

    output_path = output_dir / "exp12_gnn_player_graph.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_gnn_player_graph_experiment()

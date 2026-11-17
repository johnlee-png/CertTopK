"""Experiment 16 (Phase 10): position-normalised player utility profiling.

Implements Supplementary Theorem S5 (Positional Utility Confounding) for
real players: raw PassRank utility U(p) is confounded by position (centre
backs pass from deep, low-reward/high-cost zones; attacking midfielders pass
from high-reward zones), so a fair cross-position comparison needs
Utilde(p) = (U(p) - mu_pos(p)) / sigma_pos(p).

This is the "player profiling" use case named in the main paper's Conclusion,
demonstrated here for the first time with real numbers.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import match_level_split
from src.models.pass_success import train_success_model, predict_success_proba
from src.models.reward_model import train_reward_model, predict_reward
from src.models.utility import compute_failure_cost, compute_utility
from src.data.turnover_labels import compute_realized_turnover_cost
from src.models.failure_cost_model import train_failure_cost_model, predict_failure_cost


def _json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def run_player_profiling_experiment(
    processed_dir: str = "data/processed",
    output_dir: str = "experiments/results",
    seed: int = 42,
    min_passes: int = 15,
) -> dict:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed artifacts...")
    passes_full = pd.read_parquet(
        processed_dir / "passes_with_features.parquet",
        columns=["match_id", "player_name", "position_name", "delta_xt", "start_x",
                 "pass_outcome_name", "index", "period", "minute", "second", "team_name"],
    )
    X_full = pd.read_parquet(processed_dir / "X_full.parquet")
    y = pd.read_parquet(processed_dir / "y_labels.parquet")["pass_success"]
    all_events = pd.read_parquet(
        processed_dir / "events_360_all.parquet",
        columns=["match_id", "index", "period", "minute", "second", "team_name", "location"],
    )
    xt_grid = np.load(processed_dir / "xt_grid_train_only.npy")

    train_df, valid_df, test_df = match_level_split(passes_full, random_state=seed)
    train_idx, valid_idx, test_idx = train_df.index, valid_df.index, test_df.index

    print("\nTraining E6 (success) + R2 (reward) models...")
    success_model = train_success_model(
        "lightgbm", X_full.loc[train_idx].fillna(0), y.loc[train_idx],
        X_full.loc[valid_idx].fillna(0), y.loc[valid_idx],
    )
    p_succ_test = predict_success_proba(success_model, X_full.loc[test_idx].fillna(0))

    success_mask = y == 1
    r_train = train_idx.intersection(passes_full[success_mask].index)
    r_valid = valid_idx.intersection(passes_full[success_mask].index)
    reward_model = train_reward_model(
        "lightgbm_regressor", X_full.loc[r_train].fillna(0), passes_full.loc[r_train, "delta_xt"],
        X_full.loc[r_valid].fillna(0), passes_full.loc[r_valid, "delta_xt"],
    )
    r_pred_test = predict_reward(reward_model, X_full.loc[test_idx].fillna(0))

    print("Computing raw utility U(p) on the test set (original heuristic C_fail)...")
    passes_test = passes_full.loc[test_idx].copy()
    c_fail_heuristic_test = compute_failure_cost(passes_test, method="location_based")
    passes_test["U"] = compute_utility(p_succ_test, r_pred_test, c_fail_heuristic_test)

    print("\nAlso computing utility with the Phase 1 LEARNED C_fail (for comparison)...")
    realized_cost = compute_realized_turnover_cost(passes_full, all_events, xt_grid)
    turnover_idx = passes_full[passes_full["pass_outcome_name"].notna()].index
    cf_train = train_idx.intersection(turnover_idx)
    cf_valid = valid_idx.intersection(turnover_idx)
    cfail_model = train_failure_cost_model(
        "lightgbm_regressor",
        X_full.loc[cf_train].fillna(0), realized_cost.loc[cf_train],
        X_full.loc[cf_valid].fillna(0), realized_cost.loc[cf_valid],
    )
    learned_cost_test = predict_failure_cost(cfail_model, X_full.loc[test_idx].fillna(0))
    c_fail_learned_test = compute_failure_cost(passes_test, method="learned", learned_cost=learned_cost_test)
    passes_test["U_learned_cfail"] = compute_utility(p_succ_test, r_pred_test, c_fail_learned_test)

    print("\nPosition means, heuristic C_fail (original) vs learned C_fail (Phase 1):")
    pos_compare = passes_test.groupby("position_name")[["U", "U_learned_cfail"]].mean()
    pos_compare["start_x_mean"] = passes_test.groupby("position_name")["start_x"].mean()
    print(pos_compare.sort_values("U", ascending=False).to_string())

    print("\nApplying Supp. Theorem S5 position normalisation (using heuristic C_fail utility)...")
    pos_stats = passes_test.groupby("position_name")["U"].agg(["mean", "std", "count"])
    pos_stats["std"] = pos_stats["std"].replace(0, 1.0).fillna(1.0)
    print(pos_stats.sort_values("mean", ascending=False).to_string())

    pos_stats_renamed = pos_stats[["mean", "std"]].rename(columns={"mean": "mean_pos", "std": "std_pos"})
    passes_test = passes_test.join(pos_stats_renamed, on="position_name")
    passes_test["U_tilde"] = (passes_test["U"] - passes_test["mean_pos"]) / passes_test["std_pos"]

    print(f"\nRaw utility <-> position confound correlation check:")
    print(f"  corr(U, start_x): {passes_test['U'].corr(passes_test['start_x']):.4f}  "
          f"(positive = more advanced starting position -> higher raw utility, "
          f"exactly the confound Thm S5 describes)")
    print(f"  corr(U_tilde, start_x): {passes_test['U_tilde'].corr(passes_test['start_x']):.4f}  "
          f"(should be much closer to 0 after normalisation)")

    player_agg = (
        passes_test.groupby(["player_name", "position_name"])
        .agg(n_passes=("U", "size"), mean_U=("U", "mean"), mean_U_tilde=("U_tilde", "mean"))
        .query(f"n_passes >= {min_passes}")
        .reset_index()
    )

    print(f"\nTop 15 players by RAW utility (min {min_passes} passes):")
    top_raw = player_agg.sort_values("mean_U", ascending=False).head(15)
    print(top_raw.to_string(index=False).encode("ascii", errors="replace").decode("ascii"))

    print(f"\nTop 15 players by POSITION-NORMALISED utility (min {min_passes} passes):")
    top_norm = player_agg.sort_values("mean_U_tilde", ascending=False).head(15)
    print(top_norm.to_string(index=False).encode("ascii", errors="replace").decode("ascii"))

    raw_rank = player_agg.set_index("player_name")["mean_U"].rank(ascending=False)
    norm_rank = player_agg.set_index("player_name")["mean_U_tilde"].rank(ascending=False)
    rank_shift = (raw_rank - norm_rank).abs()
    biggest_movers = pd.DataFrame({
        "player_name": rank_shift.index,
        "position_name": player_agg.set_index("player_name")["position_name"],
        "raw_rank": raw_rank.values,
        "normalised_rank": norm_rank.values,
        "rank_shift": rank_shift.values,
    }).sort_values("rank_shift", ascending=False).head(10)
    print("\nBiggest rank movers (position normalisation changes their standing the most):")
    print(biggest_movers.to_string(index=False).encode("ascii", errors="replace").decode("ascii"))

    results = {
        "description": "Supp. Theorem S5 position-normalised utility, applied to real players",
        "position_means_heuristic_vs_learned_cfail": pos_compare.reset_index().to_dict(orient="records"),
        "position_stats": pos_stats.reset_index().to_dict(orient="records"),
        "confound_check": {
            "corr_U_vs_start_x": float(passes_test["U"].corr(passes_test["start_x"])),
            "corr_U_tilde_vs_start_x": float(passes_test["U_tilde"].corr(passes_test["start_x"])),
        },
        "top_15_raw_utility": top_raw.to_dict(orient="records"),
        "top_15_normalised_utility": top_norm.to_dict(orient="records"),
        "biggest_rank_movers": biggest_movers.to_dict(orient="records"),
        "split_manifest": {
            "seed": seed,
            "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
            "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
            "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
        },
    }

    output_path = output_dir / "exp16_player_profiling.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_player_profiling_experiment()

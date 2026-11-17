"""Experiment 21 (Phase 15): team-strength double-normalisation of player utility.

Phase 10 (exp16) normalises raw utility U(p) by position, per Supp. Theorem S5,
but leaves a flagged confound unresolved: a competent centre back on a weak
team may face more sustained pressure -- and therefore a harder passing
environment -- than an equally competent centre back on a dominant team.
This experiment adds a second covariate (team strength, proxied by each
team's own pass-success rate within that match) and jointly regresses out
both position and team strength, checking whether this further reduces the
confound and further reshuffles player rankings.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data.splits import match_level_split
from src.models.pass_success import train_success_model, predict_success_proba
from src.models.reward_model import train_reward_model, predict_reward
from src.models.utility import compute_failure_cost, compute_utility


def _json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def run_team_strength_profiling_experiment(
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
        columns=["match_id", "player_name", "position_name", "team_name", "delta_xt",
                 "start_x", "pass_outcome_name", "pass_success"],
    )
    X_full = pd.read_parquet(processed_dir / "X_full.parquet")
    y = pd.read_parquet(processed_dir / "y_labels.parquet")["pass_success"]

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

    print("Computing raw utility U(p) on the test set (heuristic C_fail, matches Phase 10)...")
    passes_test = passes_full.loc[test_idx].copy()
    c_fail_test = compute_failure_cost(passes_test, method="location_based")
    passes_test["U"] = compute_utility(p_succ_test, r_pred_test, c_fail_test)

    print("\nComputing team-strength proxy: each team's own pass-success rate within that match...")
    team_success = (
        passes_test.groupby(["match_id", "team_name"])["pass_success"]
        .mean()
        .rename("team_strength")
        .reset_index()
    )
    passes_test = passes_test.merge(team_success, on=["match_id", "team_name"], how="left")

    print("\nApplying Supp. Theorem S5 position normalisation (as in Phase 10)...")
    pos_stats = passes_test.groupby("position_name")["U"].agg(["mean", "std"])
    pos_stats["std"] = pos_stats["std"].replace(0, 1.0).fillna(1.0)
    pos_stats_renamed = pos_stats.rename(columns={"mean": "mean_pos", "std": "std_pos"})
    passes_test = passes_test.join(pos_stats_renamed, on="position_name")
    passes_test["U_tilde"] = (passes_test["U"] - passes_test["mean_pos"]) / passes_test["std_pos"]

    print("\nFitting joint regression U ~ position + team_strength for double normalisation...")
    position_dummies = pd.get_dummies(passes_test["position_name"], drop_first=True)
    X_reg = pd.concat([position_dummies, passes_test[["team_strength"]]], axis=1).astype(float)
    reg = LinearRegression()
    reg.fit(X_reg, passes_test["U"])
    predicted_U = reg.predict(X_reg)
    residual_std = (passes_test["U"] - predicted_U).std()
    passes_test["U_double"] = (passes_test["U"] - predicted_U) / residual_std
    team_strength_coef = float(reg.coef_[-1])

    print(f"Regression coefficient on team_strength: {team_strength_coef:+.4f} "
          f"(positive -> playing for a dominant team in that match is associated with higher raw utility)")

    corr_U_team = passes_test["U"].corr(passes_test["team_strength"])
    corr_Utilde_team = passes_test["U_tilde"].corr(passes_test["team_strength"])
    corr_Udouble_team = passes_test["U_double"].corr(passes_test["team_strength"])
    print(f"\ncorr(U, team_strength)        = {corr_U_team:.4f}  (raw utility vs team dominance)")
    print(f"corr(U_tilde, team_strength)  = {corr_Utilde_team:.4f}  (position-normalised, Phase 10)")
    print(f"corr(U_double, team_strength) = {corr_Udouble_team:.4f}  (position+team-normalised, this experiment)")

    player_agg = (
        passes_test.groupby(["player_name", "position_name"])
        .agg(n_passes=("U", "size"), mean_U=("U", "mean"),
             mean_U_tilde=("U_tilde", "mean"), mean_U_double=("U_double", "mean"),
             mean_team_strength=("team_strength", "mean"))
        .query(f"n_passes >= {min_passes}")
        .reset_index()
    )

    print(f"\nTop 15 players by position-normalised utility (Phase 10, U_tilde):")
    top_tilde = player_agg.sort_values("mean_U_tilde", ascending=False).head(15)
    print(top_tilde.to_string(index=False).encode("ascii", errors="replace").decode("ascii"))

    print(f"\nTop 15 players by position+team-normalised utility (this experiment, U_double):")
    top_double = player_agg.sort_values("mean_U_double", ascending=False).head(15)
    print(top_double.to_string(index=False).encode("ascii", errors="replace").decode("ascii"))

    tilde_rank = player_agg.set_index("player_name")["mean_U_tilde"].rank(ascending=False)
    double_rank = player_agg.set_index("player_name")["mean_U_double"].rank(ascending=False)
    rank_shift = (tilde_rank - double_rank).abs()
    biggest_movers = pd.DataFrame({
        "player_name": rank_shift.index,
        "position_name": player_agg.set_index("player_name")["position_name"],
        "team_strength": player_agg.set_index("player_name")["mean_team_strength"],
        "tilde_rank": tilde_rank.values,
        "double_rank": double_rank.values,
        "rank_shift": rank_shift.values,
    }).sort_values("rank_shift", ascending=False).head(15)
    print("\nBiggest movers, position-only vs position+team normalisation:")
    print(biggest_movers.to_string(index=False).encode("ascii", errors="replace").decode("ascii"))

    results = {
        "description": "Team-strength double-normalisation of player utility "
                       "(position via Phase 10 + team-match pass-success-rate proxy, this experiment)",
        "regression_coef_team_strength": team_strength_coef,
        "confound_check": {
            "corr_U_vs_team_strength": float(corr_U_team),
            "corr_U_tilde_vs_team_strength": float(corr_Utilde_team),
            "corr_U_double_vs_team_strength": float(corr_Udouble_team),
        },
        "top_15_position_normalised": top_tilde.to_dict(orient="records"),
        "top_15_position_and_team_normalised": top_double.to_dict(orient="records"),
        "biggest_rank_movers": biggest_movers.to_dict(orient="records"),
        "split_manifest": {
            "seed": seed,
            "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
            "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
            "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
        },
    }

    output_path = output_dir / "exp21_team_strength_profiling.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_team_strength_profiling_experiment()

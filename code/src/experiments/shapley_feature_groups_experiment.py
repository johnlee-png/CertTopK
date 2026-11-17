"""Experiment 18 (Phase 12): exact Shapley decomposition of the 4 spatial feature groups.

Theorem 4's Corollary claims that averaging over all feature-addition orderings
(the Shapley value) places receiver space's contribution in [0.020, 0.025], resolving
the isolated (+0.025) vs cumulative (+0.020) ablation-order discrepancy reported in
Sec. 5.6. This computes the *exact* Shapley value for all 4 groups (16 model trainings,
covering every subset of {passer_pressure, receiver_space, lane_obstruction, team_structure}
added on top of the fixed event-only baseline) and checks that claim directly.
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.splits import match_level_split
from src.models.pass_success import train_success_model, predict_success_proba
from src.experiments.ablation import FEATURE_GROUPS

EVENT_ONLY_COLS = [
    "start_x", "start_y", "end_x", "end_y", "pass_length", "pass_angle", "forward_distance",
    "start_x_norm", "start_y_norm", "end_x_norm", "end_y_norm", "start_dist_to_goal",
    "end_dist_to_goal", "minute", "period", "under_pressure", "is_ground_pass", "is_high_pass",
    "is_low_pass", "is_right_foot", "start_in_own_third", "start_in_mid_third",
    "start_in_final_third", "end_in_final_third", "end_in_penalty_area", "is_cross", "is_switch",
]

GROUPS = {k: v for k, v in FEATURE_GROUPS.items() if k != "A_event_only"}
GROUP_NAMES = list(GROUPS.keys())  # B_passer_pressure, C_receiver_space, D_lane_obstruction, E_team_structure


def _json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def run_shapley_feature_groups_experiment(
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

    n = len(GROUP_NAMES)
    print(f"\nGroups (n={n}): {GROUP_NAMES}")
    print(f"Training {2 ** n} LightGBM models (one per subset of groups, on top of "
          f"{len(EVENT_ONLY_COLS)} event-only features)...\n")

    v = {}  # frozenset(subset of GROUP_NAMES) -> test AUROC
    for k in range(n + 1):
        for subset in itertools.combinations(GROUP_NAMES, k):
            cols = list(EVENT_ONLY_COLS)
            for g in subset:
                cols += GROUPS[g]
            cols = [c for c in cols if c in X_full.columns]

            model = train_success_model(
                "lightgbm",
                X_full.loc[train_idx, cols].fillna(0), y.loc[train_idx],
                X_full.loc[valid_idx, cols].fillna(0), y.loc[valid_idx],
            )
            p_test = predict_success_proba(model, X_full.loc[test_idx, cols].fillna(0))
            auroc = roc_auc_score(y_test, p_test)
            v[frozenset(subset)] = auroc
            label = "{" + ",".join(g.split("_", 1)[1] for g in subset) + "}" if subset else "{}"
            print(f"  v({label}) = AUROC = {auroc:.4f}  ({len(cols)} features)")

    print("\nComputing exact Shapley value for each group "
          "(weighted average marginal contribution over all 8 orderings of the other 3 groups)...")
    shapley = {}
    for i in GROUP_NAMES:
        others = [g for g in GROUP_NAMES if g != i]
        total = 0.0
        for k in range(len(others) + 1):
            for S in itertools.combinations(others, k):
                S_set, Si_set = frozenset(S), frozenset(S) | {i}
                weight = (math.factorial(len(S)) * math.factorial(n - len(S) - 1)) / math.factorial(n)
                total += weight * (v[Si_set] - v[S_set])
        shapley[i] = total

    print("\nExact Shapley values (AUROC contribution):")
    for g, val in shapley.items():
        print(f"  {g}: {val:+.4f}")

    total_shapley = sum(shapley.values())
    total_gain = v[frozenset(GROUP_NAMES)] - v[frozenset()]
    print(f"\nEfficiency check: sum(Shapley) = {total_shapley:.4f} vs "
          f"v(all groups)-v(none) = {total_gain:.4f}  (should match exactly)")
    assert abs(total_shapley - total_gain) < 1e-9

    isolated_receiver = v[frozenset(["C_receiver_space"])] - v[frozenset()]
    cumulative_receiver = v[frozenset(GROUP_NAMES[:2])] - v[frozenset(GROUP_NAMES[:1])]
    receiver_shapley = shapley["C_receiver_space"]

    print(f"\n=== Corollary check ===")
    print(f"Isolated gain for receiver_space (added alone to event baseline):   {isolated_receiver:+.4f}")
    print(f"Cumulative gain for receiver_space (added after passer_pressure):    {cumulative_receiver:+.4f}")
    print(f"Exact Shapley value for receiver_space:                             {receiver_shapley:+.4f}")
    print(f"Corollary claims Shapley value should be in [0.020, 0.025]. "
          f"Actual: {receiver_shapley:.4f} -> "
          f"{'INSIDE the claimed range' if 0.020 <= receiver_shapley <= 0.025 else 'OUTSIDE the claimed range'}")

    results = {
        "description": "Exact Shapley decomposition of 4 spatial feature groups (16 model trainings), "
                        "testing Theorem 4 Corollary's claim that Shapley(receiver_space) in [0.020, 0.025]",
        "group_names": GROUP_NAMES,
        "v_by_subset": {",".join(sorted(s)) if s else "empty": val for s, val in v.items()},
        "shapley_values": shapley,
        "efficiency_check": {"sum_shapley": total_shapley, "v_all_minus_v_none": total_gain},
        "corollary_check": {
            "isolated_receiver_space": isolated_receiver,
            "cumulative_receiver_space": cumulative_receiver,
            "exact_shapley_receiver_space": receiver_shapley,
            "claimed_range": [0.020, 0.025],
            "inside_claimed_range": bool(0.020 <= receiver_shapley <= 0.025),
        },
        "split_manifest": {
            "seed": seed,
            "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
            "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
            "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
        },
    }

    output_path = output_dir / "exp18_shapley_feature_groups.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_shapley_feature_groups_experiment()

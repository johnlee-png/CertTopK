"""Run all 6 experiments in sequence with consolidated output."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import (
    get_matches_with_360,
    load_and_join_events_360,
    load_events,
    filter_open_play_passes,
    label_progressive_passes,
)
from src.data.splits import match_level_split
from src.features.event_features import extract_event_features, extract_pass_outcome
from src.features.context_360_features import extract_360_features
from src.features.expected_threat import build_xt_grid, compute_delta_xt, get_xt_value
from src.models.pass_success import train_success_model, predict_success_proba
from src.models.reward_model import train_reward_model, predict_reward
from src.models.utility import compute_failure_cost, compute_utility
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    top_decile_precision,
    bootstrap_metric,
    evaluate_utility_deciles,
)
from src.experiments.ablation import run_order_robust_ablation_suite
from src.experiments.case_study import select_case_study_passes, generate_case_study_report
from src.reporting import generate_results_summary


def _json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    raise TypeError(f"Not serializable: {type(obj)}")


def run_all(base_path: str, output_dir: str = "experiments/results", seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    # ===== Data Loading =====
    print("=" * 70)
    print("DATA LOADING")
    print("=" * 70)

    match_ids = get_matches_with_360(base_path)
    print(f"Matches with 360: {len(match_ids)}")

    df = load_and_join_events_360(base_path, match_ids)
    print(f"Events with 360: {len(df)}")
    used_match_ids = sorted(df["match_id"].unique().tolist())
    excluded_match_ids = sorted(set(match_ids) - set(used_match_ids))

    passes = filter_open_play_passes(df)
    passes = label_progressive_passes(passes)
    print(f"Open-play footed passes: {len(passes)}")
    print(f"Progressive: {passes['is_progressive'].sum()} ({passes['is_progressive'].mean()*100:.1f}%)")

    # Split
    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)
    train_idx, valid_idx, test_idx = train_df.index, valid_df.index, test_df.index
    train_match_ids = sorted(train_df["match_id"].unique().tolist())
    valid_match_ids = sorted(valid_df["match_id"].unique().tolist())
    test_match_ids = sorted(test_df["match_id"].unique().tolist())

    # xT Grid built on train matches only
    all_events = df[df["type_name"].isin(["Pass", "Carry", "Shot"])].copy()
    train_events = all_events[all_events["match_id"].isin(train_match_ids)].copy()
    xt_grid = build_xt_grid(train_events)
    passes["delta_xt"] = compute_delta_xt(passes, xt_grid)
    passes["start_xt"] = passes.apply(
        lambda r: get_xt_value(xt_grid, r["start_x"], r["start_y"]), axis=1
    )

    # Features and realized evaluation targets
    event_features = extract_event_features(passes)
    context_features = extract_360_features(passes)
    y = extract_pass_outcome(passes)
    passes["pass_success"] = y
    passes["realized_delta_xt"] = passes["delta_xt"] * passes["pass_success"]
    passes["realized_progressive"] = (passes["is_progressive"] & passes["pass_success"].astype(bool)).astype(int)
    passes["realized_final_third_entry"] = (
        (passes["end_x"] >= 80) & passes["pass_success"].astype(bool)
    ).astype(int)

    X_event = event_features
    X_full = pd.concat([event_features, context_features], axis=1)

    # ===== Experiment 1: Success Prediction =====
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Pass Success Prediction")
    print("=" * 70)

    exp1 = {}
    configs = [
        ("E1", "logistic_regression", X_event, "Event-only LR"),
        ("E2", "xgboost", X_event, "Event-only XGB"),
        ("E3", "lightgbm", X_event, "Event-only LGBM"),
        ("E4", "logistic_regression", X_full, "Event+360 LR"),
        ("E5", "xgboost", X_full, "Event+360 XGB"),
        ("E6", "lightgbm", X_full, "Event+360 LGBM"),
    ]

    trained = {}
    for tag, mname, X, desc in configs:
        Xtr = X.loc[train_idx].fillna(0)
        Xva = X.loc[valid_idx].fillna(0)
        Xte = X.loc[test_idx].fillna(0)
        model = train_success_model(mname, Xtr, y.loc[train_idx], Xva, y.loc[valid_idx])
        yp = predict_success_proba(model, Xte)
        m = compute_classification_metrics(y.loc[test_idx].values, yp)
        exp1[tag] = {"desc": desc, **m}
        trained[tag] = (model, X, yp)
        print(f"  {tag} {desc}: AUROC={m['auroc']:.4f} Brier={m['brier_score']:.4f}")

    all_results["exp1_success"] = exp1

    # Bootstrap CI for best models
    print("\n  Bootstrap CI (E3 vs E6):")
    for tag in ["E3", "E6"]:
        _, _, yp = trained[tag]
        from sklearn.metrics import roc_auc_score
        pt, lo, hi = bootstrap_metric(
            y.loc[test_idx].values,
            yp,
            roc_auc_score,
            n_bootstrap=1000,
            match_ids=passes.loc[test_idx, "match_id"].values,
        )
        print(f"    {tag} AUROC: {pt:.4f} [{lo:.4f}, {hi:.4f}]")

    # ===== Experiment 2: Reward Prediction =====
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Reward Prediction")
    print("=" * 70)

    success_mask = y == 1
    r_train = train_idx.intersection(passes[success_mask].index)
    r_valid = valid_idx.intersection(passes[success_mask].index)
    r_test = test_idx.intersection(passes[success_mask].index)

    exp2 = {}
    reward_models = {}
    for tag, mname, X, desc in [
        ("R1", "lightgbm_regressor", X_event, "Event-only"),
        ("R2", "lightgbm_regressor", X_full, "Event+360"),
    ]:
        rm = train_reward_model(mname, X.loc[r_train].fillna(0), passes.loc[r_train, "delta_xt"],
                                X.loc[r_valid].fillna(0), passes.loc[r_valid, "delta_xt"])
        rp = predict_reward(rm, X.loc[r_test].fillna(0))
        m = compute_regression_metrics(passes.loc[r_test, "delta_xt"].values, rp)
        m["top_decile"] = top_decile_precision(passes.loc[r_test, "delta_xt"].values, rp)
        exp2[tag] = {"desc": desc, **m}
        reward_models[tag] = (rm, X)
        print(f"  {tag} {desc}: RMSE={m['rmse']:.4f} Spearman={m['spearman_corr']:.4f}")

    all_results["exp2_reward"] = exp2

    # ===== Experiment 3: Utility Ranking =====
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Utility Ranking")
    print("=" * 70)

    exp3 = {}
    utility_arrays = {}
    for label, s_tag, r_tag in [("event_only", "E3", "R1"), ("event_360", "E6", "R2")]:
        s_model, s_X, _ = trained[s_tag]
        r_model, r_X = reward_models[r_tag]

        p_succ = predict_success_proba(s_model, s_X.loc[test_idx].fillna(0))
        r_pred = predict_reward(r_model, r_X.loc[test_idx].fillna(0))
        c_fail = compute_failure_cost(passes.loc[test_idx])
        util = compute_utility(p_succ, r_pred, c_fail)
        utility_arrays[label] = util

        teval = passes.loc[test_idx].copy()
        teval["utility_score"] = util
        deciles = evaluate_utility_deciles(teval)
        exp3[label] = deciles.to_dict(orient="records")
        print(f"\n  {label}:")
        print(deciles.to_string(index=False))

    all_results["exp3_utility"] = exp3

    # ===== Experiment 4: Progressive Subset =====
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Progressive Pass Subset")
    print("=" * 70)

    prog_mask = passes["is_progressive"]
    prog_test = test_idx.intersection(passes[prog_mask].index)

    exp4 = {}

    # Progressive-trained models
    prog_train = train_idx.intersection(passes[prog_mask].index)
    prog_valid = valid_idx.intersection(passes[prog_mask].index)
    for tag, X, desc in [
        ("E3p", X_event, "Event-only prog-trained"),
        ("E6p", X_full, "Event+360 prog-trained"),
    ]:
        model_name = "lightgbm"
        model = train_success_model(
            model_name,
            X.loc[prog_train].fillna(0),
            y.loc[prog_train],
            X.loc[prog_valid].fillna(0),
            y.loc[prog_valid],
        )
        yp = predict_success_proba(model, X.loc[prog_test].fillna(0))
        m = compute_classification_metrics(y.loc[prog_test].values, yp)
        exp4[tag] = {"desc": desc, **m}
        print(f"  {tag}: AUROC={m['auroc']:.4f} Brier={m['brier_score']:.4f}")

    # Full-data models evaluated on progressive subset
    for src_tag, out_tag, desc in [
        ("E3", "E3->prog", "Event full->prog"),
        ("E6", "E6->prog", "360 full->prog"),
    ]:
        _, X, _ = trained[src_tag]
        model = trained[src_tag][0]
        yp = predict_success_proba(model, X.loc[prog_test].fillna(0))
        m = compute_classification_metrics(y.loc[prog_test].values, yp)
        exp4[out_tag] = {"desc": desc, **m}
        print(f"  {out_tag}: AUROC={m['auroc']:.4f} Brier={m['brier_score']:.4f}")

    all_results["exp4_progressive"] = exp4

    # ===== Experiment 5: Ablation =====
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Ablation Study")
    print("=" * 70)

    ablation_results = run_order_robust_ablation_suite(
        X_full, y, passes["is_progressive"],
        event_features.columns.tolist(),
        train_idx, valid_idx, test_idx,
    )
    all_results["exp5_ablation_full"] = ablation_results["full_cumulative"].to_dict(orient="records")
    all_results["exp5_ablation_progressive"] = ablation_results["prog_cumulative"].to_dict(orient="records")
    all_results["exp5_ablation_full_isolated"] = ablation_results["full_isolated"].to_dict(orient="records")
    all_results["exp5_ablation_progressive_isolated"] = ablation_results["prog_isolated"].to_dict(orient="records")
    all_results["exp5_ablation_full_leave_one_out"] = ablation_results["full_leave_one_out"].to_dict(orient="records")
    all_results["exp5_ablation_progressive_leave_one_out"] = ablation_results["prog_leave_one_out"].to_dict(orient="records")

    # ===== Experiment 6: Case Study =====
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Tactical Case Study")
    print("=" * 70)

    cases = select_case_study_passes(
        passes.loc[test_idx],
        utility_arrays["event_only"],
        utility_arrays["event_360"],
        n=10,
    )
    report = generate_case_study_report(cases, f"{output_dir}/case_study_report.md")
    print(f"  Case study report saved to {output_dir}/case_study_report.md")

    for k, v in cases.items():
        print(f"  {k}: {len(v)} cases")

    # ===== Save All =====
    all_results["split_manifest"] = {
        "seed": seed,
        "train_match_ids": train_match_ids,
        "valid_match_ids": valid_match_ids,
        "test_match_ids": test_match_ids,
    }

    with open(f"{output_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=_json_serializable)

    final_results = {
        "exp1": exp1,
        "exp2": exp2,
        "exp3_event": exp3["event_only"],
        "exp3_360": exp3["event_360"],
        "exp4": exp4,
        "exp5_full": all_results["exp5_ablation_full"],
        "exp5_prog": all_results["exp5_ablation_progressive"],
        "exp5_full_isolated": all_results["exp5_ablation_full_isolated"],
        "exp5_prog_isolated": all_results["exp5_ablation_progressive_isolated"],
        "exp5_full_leave_one_out": all_results["exp5_ablation_full_leave_one_out"],
        "exp5_prog_leave_one_out": all_results["exp5_ablation_progressive_leave_one_out"],
        "split_manifest": all_results["split_manifest"],
    }

    with open(f"{output_dir}/final_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=_json_serializable)

    provenance = {
        "requested_match_count": len(match_ids),
        "used_match_count": len(used_match_ids),
        "excluded_match_count": len(excluded_match_ids),
        "requested_match_ids": match_ids,
        "used_match_ids": used_match_ids,
        "excluded_match_ids": excluded_match_ids,
    }
    with open(f"{output_dir}/data_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    # Save xT grid
    np.save(f"{output_dir}/xt_grid_train_only.npy", xt_grid)

    generate_results_summary(
        final_results_path=f"{output_dir}/final_results.json",
        output_path=f"{output_dir}/RESULTS_SUMMARY.md",
        provenance_path=f"{output_dir}/data_provenance.json",
    )

    print(f"\n{'=' * 70}")
    print(f"All results saved to {output_dir}/")
    print(f"{'=' * 70}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/raw/statsbomb-open-data/data")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_all(args.data_path, args.output_dir, args.seed)

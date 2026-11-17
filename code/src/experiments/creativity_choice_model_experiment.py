"""Experiment 23 (Phase 17): a fitted softmax choice model replaces the
originality heuristic, testing the Future Work direction flagged after
Phase 14 -- moving PassRank's creativity metric closer to un-xPass's
receiver-choice-model formulation.

Phase 9's originality heuristic ranks visible teammates by xT and reports
what fraction were "more threatening" than the actually chosen receiver.
This experiment instead fits a single-parameter softmax (conditional-logit)
choice model, P(choose candidate i) = exp(beta * xT_i) / sum_j exp(beta * xT_j),
via maximum likelihood on the training split, then scores test-set passes
by 1 - P_model(chosen receiver) as a calibrated, model-based originality.
We re-run Phase 14's assist-prediction benchmark to see whether this
improves discrimination over the heuristic.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

from src.data.splits import match_level_split
from src.features.expected_threat import get_xt_value
from src.evaluation.metrics import bootstrap_metric


def _json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def _extract_choice_situations(passes, xt_grid, pitch_length=120.0, pitch_width=80.0):
    """For each pass, return (candidate_xts, chosen_index) where chosen_index
    is the candidate nearest to the pass's actual end location. Passes with
    fewer than 2 candidates are skipped (no real choice to model)."""
    situations = []
    valid_idx = []
    for idx, ff, end_x, end_y in zip(
        passes.index, passes["freeze_frame"].values, passes["end_x"].values, passes["end_y"].values
    ):
        if ff is None or len(ff) == 0:
            continue
        candidates = []
        for p in ff:
            if not p.get("teammate", False) or p.get("actor", False):
                continue
            loc = p.get("location")
            if loc is None or len(loc) < 2:
                continue
            candidates.append(loc)
        if len(candidates) < 2:
            continue
        candidate_xts = np.array([get_xt_value(xt_grid, c[0], c[1], pitch_length, pitch_width) for c in candidates])
        dists_to_end = np.array([np.hypot(c[0] - end_x, c[1] - end_y) for c in candidates])
        chosen_index = int(np.argmin(dists_to_end))
        situations.append((candidate_xts, chosen_index))
        valid_idx.append(idx)
    return situations, valid_idx


def _neg_log_likelihood(beta, situations):
    total = 0.0
    for xts, chosen in situations:
        scores = beta * xts
        scores = scores - scores.max()  # numerical stability
        log_probs = scores - np.log(np.sum(np.exp(scores)))
        total -= log_probs[chosen]
    return total / len(situations)


def run_creativity_choice_model_experiment(
    processed_dir: str = "data/processed",
    output_dir: str = "experiments/results",
    seed: int = 42,
) -> dict:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed artifacts...")
    passes = pd.read_parquet(
        processed_dir / "passes_with_features.parquet",
        columns=["match_id", "freeze_frame", "start_x", "start_y", "end_x", "end_y",
                 "delta_xt", "pass_success", "is_progressive", "player_name",
                 "pass_shot_assist", "pass_goal_assist"],
    )
    xt_grid = np.load(processed_dir / "xt_grid_train_only.npy")

    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)
    train_idx, test_idx = train_df.index, test_df.index

    print("\nExtracting choice situations from the training split...")
    train_situations, _ = _extract_choice_situations(passes.loc[train_idx], xt_grid)
    print(f"  {len(train_situations)} training passes with >=2 candidates")

    print("\nFitting softmax choice model (single parameter beta) via MLE...")
    result = minimize_scalar(_neg_log_likelihood, args=(train_situations,), bounds=(-2, 50), method="bounded")
    beta_hat = result.x
    print(f"  beta_hat = {beta_hat:.4f}  (negative log-likelihood = {result.fun:.4f})")
    print(f"  (beta>0 means: candidates with higher xT are, on average, more likely to be chosen -- "
          f"a sanity check that the model learned a sensible direction)")

    print("\nScoring test-set passes with the fitted choice model...")
    passes_test = passes.loc[test_idx].copy()
    passes_test["pass_shot_assist"] = passes_test["pass_shot_assist"].fillna(False).astype(bool)
    passes_test["pass_goal_assist"] = passes_test["pass_goal_assist"].fillna(False).astype(bool)

    test_situations, valid_test_idx = _extract_choice_situations(passes_test, xt_grid)
    passes_test = passes_test.loc[valid_test_idx].copy()

    p_chosen = np.empty(len(test_situations))
    for i, (xts, chosen) in enumerate(test_situations):
        scores = beta_hat * xts
        scores = scores - scores.max()
        probs = np.exp(scores) / np.sum(np.exp(scores))
        p_chosen[i] = probs[chosen]

    passes_test["originality_choice"] = 1 - p_chosen
    realized_value = passes_test["delta_xt"].values * passes_test["pass_success"].values
    passes_test["creativity_choice"] = passes_test["originality_choice"].values * realized_value

    print(f"\nTest passes scored: {len(passes_test)}")
    print(f"  mean originality_choice: {passes_test['originality_choice'].mean():.4f}")
    print(f"  mean creativity_choice:  {passes_test['creativity_choice'].mean():.6f}")

    match_ids = passes_test["match_id"].values
    results = {"description": "Softmax choice-model creativity metric (Phase 17), "
                               "benchmarked against Phase 14's heuristic-based AUROC",
               "beta_hat": float(beta_hat)}

    print("\n=== Re-running Phase 14's assist-prediction benchmark with the choice-model score ===")
    for target in ["pass_shot_assist", "pass_goal_assist"]:
        y = passes_test[target].astype(int).values
        target_results = {}
        for score_name in ["creativity_choice", "originality_choice"]:
            score = passes_test[score_name].values
            pt, lo, hi = bootstrap_metric(y, score, lambda yt, yp: roc_auc_score(yt, yp), match_ids=match_ids)
            print(f"  {target} / {score_name}: AUROC={pt:.4f} [{lo:.4f},{hi:.4f}]")
            target_results[score_name] = {"auroc": pt, "ci_low": lo, "ci_high": hi}
        results[target] = target_results

    print("\nFor comparison, Phase 14's original heuristic-based results were:")
    print("  pass_shot_assist / creativity: AUROC=0.6562 [0.6139,0.7001]")
    print("  pass_goal_assist / creativity: AUROC=0.7102 [0.5987,0.8110]")
    results["phase14_heuristic_reference"] = {
        "pass_shot_assist": {"auroc": 0.6561689147937575, "ci_low": 0.6139223185229141, "ci_high": 0.7000820060072251},
        "pass_goal_assist": {"auroc": 0.7102155165903977, "ci_low": 0.5986753705308833, "ci_high": 0.8109906263653076},
    }

    results["n_train_situations"] = len(train_situations)
    results["n_test_situations"] = len(test_situations)
    results["split_manifest"] = {
        "seed": seed,
        "train_match_ids": sorted(train_df["match_id"].unique().tolist()),
        "valid_match_ids": sorted(valid_df["match_id"].unique().tolist()),
        "test_match_ids": sorted(test_df["match_id"].unique().tolist()),
    }

    output_path = output_dir / "exp23_creativity_choice_model.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_creativity_choice_model_experiment()

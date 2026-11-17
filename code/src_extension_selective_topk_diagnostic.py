"""Go/no-go experiment for certified top-k with selective (expensive) inference.

Each tuple's utility is U = P*R - (1-P)*C. A cheap model gives P_c for every
tuple; an expensive model gives P_e but we want to call it on as few tuples
as possible. The target answer is the top-k under U_e (what you get by running
the expensive model on everything, method A).

Bounds without true P, both with coverage ~1-alpha per tuple, giving
U_e in [U_c - (R+C) q_i, U_c + (R+C) q_i]:
  binned    per-bin split-conformal quantile of |P_e - P_c| on the validation
            split (bins on P_c)
  adaptive  locally weighted conformal: a LightGBM model of |P_e - P_c| from
            the cheap features is fit on one half of the validation groups
            (matches / patients), and the normalized residual is calibrated
            on the other half

Methods compared at the same expensive-call budget b (= calls used by C):
  A  expensive model on all tuples (reference, n calls)
  B  cheap model only (0 calls)
  C  proposed: refine only tuples that violate the certificate
     min_{i in T} L_i >= max_{j not in T} H_j, stop once it holds
  D  refine b random tuples (mean over 20 seeds)
  E  refine the b tuples closest to the cheap cutoff, measured in bound widths
     (an uncertainty-ordering heuristic in the spirit of Everest, not a
     reimplementation of it)

Soccer reuses the pipeline of src_extension_theorem_new_regret_diagnostic.py
(same seed/split), with the reward model trained on event features so that R
is cheap. Diabetes reuses paper/src_extension/diabetes_new_regret_diagnostic.py.
"""
import json
import math
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.splits import match_level_split
from src.models.pass_success import train_success_model, predict_success_proba
from src.models.reward_model import train_reward_model, predict_reward
from src.models.utility import compute_failure_cost

sys.path.insert(0, str(Path(__file__).resolve().parent / "paper" / "src_extension"))
import diabetes_new_regret_diagnostic as dia  # noqa: E402

K_FRACTIONS = [0.01, 0.05, 0.10, 0.20]
ALPHAS = [0.10, 0.05, 0.01]
N_BINS = 10
N_RANDOM = 20


def utility(p, R, C):
    return p * R - (1 - p) * C


def conformal_halfwidth(pc_val, pe_val, pc_test, alpha):
    """Per-bin split-conformal quantile of |P_e - P_c|, bins on P_c."""
    edges = np.unique(np.quantile(pc_val, np.linspace(0, 1, N_BINS + 1)))
    bin_val = np.clip(np.searchsorted(edges, pc_val, side="right") - 1, 0, len(edges) - 2)
    bin_test = np.clip(np.searchsorted(edges, pc_test, side="right") - 1, 0, len(edges) - 2)
    resid = np.abs(pe_val - pc_val)
    q = np.empty(len(edges) - 1)
    for b in range(len(q)):
        r = np.sort(resid[bin_val == b])
        level = math.ceil((len(r) + 1) * (1 - alpha))
        q[b] = r[min(level, len(r)) - 1]
    return q[bin_test]


def adaptive_halfwidth(xc_val, pc_val, pe_val, groups_val, xc_test, pc_test, alpha):
    """Locally weighted split conformal: q_i = s(x_i) * Q, fit/calibrate on disjoint groups."""
    uniq = np.unique(groups_val)
    rng = np.random.default_rng(42)
    fit_groups = set(rng.choice(uniq, len(uniq) // 2, replace=False).tolist())
    fit = np.array([g in fit_groups for g in groups_val])

    def feats(xc, pc):
        X = np.asarray(xc, dtype=float)
        return np.column_stack([X, pc])

    resid = np.abs(pe_val - pc_val)
    scale_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                    min_child_samples=50, verbosity=-1, random_state=42)
    scale_model.fit(feats(xc_val[fit], pc_val[fit]), resid[fit])
    floor = np.quantile(resid[fit], 0.05) + 1e-6
    s_cal = np.maximum(scale_model.predict(feats(xc_val[~fit], pc_val[~fit])), floor)
    r = np.sort(resid[~fit] / s_cal)
    Q = r[min(math.ceil((len(r) + 1) * (1 - alpha)), len(r)) - 1]
    return np.maximum(scale_model.predict(feats(xc_test, pc_test)), floor) * Q


def topk(score, k):
    return np.argpartition(-score, k - 1)[:k]


def compare(sel, ref, U_e, k):
    s, r = set(sel.tolist()), set(ref.tolist())
    return len(s & r) / k, float(U_e[ref].sum() - U_e[sel].sum())


def certify(U_c, U_e, w, k):
    lo, hi = U_c - w, U_c + w
    refined = np.zeros(len(U_c), dtype=bool)
    batch = max(1, math.ceil(0.01 * k))
    rounds = 0
    while True:
        rounds += 1
        T = topk(lo, k)
        inT = np.zeros(len(U_c), dtype=bool)
        inT[T] = True
        m, M = lo[T].min(), hi[~inT].max()
        if m >= M:
            return T, refined, True, rounds
        viol = ~refined & ((inT & (lo < M)) | (~inT & (hi > m)))
        if not viol.any():
            return T, refined, False, rounds
        idx = np.flatnonzero(viol)
        pick = idx[np.argsort(-(hi[idx] - lo[idx]))[:batch]]
        lo[pick] = hi[pick] = U_e[pick]
        refined[pick] = True


def mixed_topk(U_c, U_e, refine_idx, k):
    score = U_c.copy()
    score[refine_idx] = U_e[refine_idx]
    return topk(score, k)


def run_domain(name, d):
    pc, pe, R, C = d["pc"], d["pe"], d["R"], d["C"]
    U_c, U_e = utility(pc, R, C), utility(pe, R, C)
    n = len(pc)
    rows = []
    for bound, alpha in [(b, a) for b in ("binned", "adaptive") for a in ALPHAS]:
        if bound == "binned":
            q = conformal_halfwidth(d["pc_val"], d["pe_val"], pc, alpha)
        else:
            q = adaptive_halfwidth(d["xc_val"], d["pc_val"], d["pe_val"], d["groups_val"],
                                   d["xc"], pc, alpha)
        w = (R + C) * q
        coverage = float(np.mean(np.abs(pe - pc) <= q))
        for k_frac in K_FRACTIONS:
            k = int(round(k_frac * n))
            ref = topk(U_e, k)
            t0 = time.perf_counter()
            T, refined, certified, rounds = certify(U_c, U_e, w, k)
            t_cert = time.perf_counter() - t0
            b = int(refined.sum())

            ov_b, rg_b = compare(topk(U_c, k), ref, U_e, k)
            ov_c, rg_c = compare(T, ref, U_e, k)

            rng = np.random.default_rng(0)
            rand = [compare(mixed_topk(U_c, U_e, rng.choice(n, b, replace=False), k), ref, U_e, k)
                    for _ in range(N_RANDOM)]
            ov_d, rg_d = float(np.mean([x[0] for x in rand])), float(np.mean([x[1] for x in rand]))

            cut = np.sort(U_c)[::-1][k - 1]
            closeness = np.abs(U_c - cut) / np.maximum(w, 1e-12)
            ov_e, rg_e = compare(mixed_topk(U_c, U_e, np.argsort(closeness)[:b], k), ref, U_e, k)

            rows.append({
                "domain": name, "bound": bound, "alpha": alpha, "bound_coverage": coverage,
                "mean_halfwidth_P": float(q.mean()),
                "k_fraction": k_frac, "n": n, "k": k,
                "certified": certified, "rounds": rounds, "certify_seconds": t_cert,
                "expensive_calls_C": b, "call_fraction_C": b / n,
                "cost_ratio_expensive_over_cheap": d["cost_ratio"],
                "overlap_B": ov_b, "regret_B": rg_b,
                "overlap_C": ov_c, "regret_C": rg_c,
                "overlap_D_random": ov_d, "regret_D_random": rg_d,
                "overlap_E_uncertainty": ov_e, "regret_E_uncertainty": rg_e,
            })
    return rows


def timed_predict(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def build_soccer(seed=42):
    processed_dir = Path("data/processed")
    passes = pd.read_parquet(processed_dir / "passes_with_features.parquet")
    X_event = pd.read_parquet(processed_dir / "X_event.parquet").fillna(0)
    X_full = pd.read_parquet(processed_dir / "X_full.parquet").fillna(0)
    y = pd.read_parquet(processed_dir / "y_labels.parquet")["pass_success"]

    train_df, valid_df, test_df = match_level_split(passes, random_state=seed)
    tr, va, te = train_df.index, valid_df.index, test_df.index

    print("[soccer] Training cheap (event-only) and expensive (event+360) success models...")
    cheap = train_success_model("lightgbm", X_event.loc[tr], y.loc[tr], X_event.loc[va], y.loc[va])
    exp = train_success_model("lightgbm", X_full.loc[tr], y.loc[tr], X_full.loc[va], y.loc[va])
    pc_val = np.asarray(predict_success_proba(cheap, X_event.loc[va]))
    pe_val = np.asarray(predict_success_proba(exp, X_full.loc[va]))
    pc, t_c = timed_predict(lambda: np.asarray(predict_success_proba(cheap, X_event.loc[te])))
    pe, t_e = timed_predict(lambda: np.asarray(predict_success_proba(exp, X_full.loc[te])))

    succ = y == 1
    print("[soccer] Training reward model on event features...")
    reward = train_reward_model(
        "lightgbm_regressor",
        X_event.loc[tr.intersection(passes[succ].index)], passes.loc[tr.intersection(passes[succ].index), "delta_xt"],
        X_event.loc[va.intersection(passes[succ].index)], passes.loc[va.intersection(passes[succ].index), "delta_xt"],
    )
    R = np.asarray(predict_reward(reward, X_event.loc[te]))
    C = np.asarray(compute_failure_cost(passes.loc[te], method="location_based"))
    R_val = np.asarray(predict_reward(reward, X_event.loc[va]))
    C_val = np.asarray(compute_failure_cost(passes.loc[va], method="location_based"))
    return {"pc_val": pc_val, "pe_val": pe_val, "pc": pc, "pe": pe, "R": R, "C": C,
            "R_val": R_val, "C_val": C_val,
            "xc_val": X_event.loc[va].to_numpy(), "xc": X_event.loc[te].to_numpy(),
            "groups_val": passes.loc[va, "match_id"].to_numpy(), "cost_ratio": t_e / t_c}


def build_diabetes(seed=42):
    df = pd.read_csv(dia.DATA_DIR / "diabetic_data.csv")
    df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)
    gf = dia.GroupedFrame(df, group_col="patient_nbr")
    train_gf, valid_gf, test_gf = gf.split(train_ratio=0.70, valid_ratio=0.15, random_state=seed)
    tr, va, te = train_gf.df, valid_gf.df, test_gf.df

    print("[diabetes] Training cheap (demographic) and expensive (all features) models...")
    feats = {"cheap": dia.DEMOGRAPHIC_COLS, "exp": dia.ALL_COLS}
    models = {key: dia.train_lgb(dia.prep_features(tr, cols), tr["readmitted_30"],
                                 dia.prep_features(va, cols), va["readmitted_30"])
              for key, cols in feats.items()}
    pc_val = models["cheap"].predict_proba(dia.prep_features(va, feats["cheap"]))[:, 1]
    pe_val = models["exp"].predict_proba(dia.prep_features(va, feats["exp"]))[:, 1]
    pc, t_c = timed_predict(lambda: models["cheap"].predict_proba(dia.prep_features(te, feats["cheap"]))[:, 1])
    pe, t_e = timed_predict(lambda: models["exp"].predict_proba(dia.prep_features(te, feats["exp"]))[:, 1])

    R = 1.0 + te["number_inpatient"].to_numpy(dtype=float)
    C = 0.05 * (te["time_in_hospital"].to_numpy(dtype=float) / df["time_in_hospital"].max())
    R_val = 1.0 + va["number_inpatient"].to_numpy(dtype=float)
    C_val = 0.05 * (va["time_in_hospital"].to_numpy(dtype=float) / df["time_in_hospital"].max())
    return {"pc_val": pc_val, "pe_val": pe_val, "pc": pc, "pe": pe, "R": R, "C": C,
            "R_val": R_val, "C_val": C_val,
            "xc_val": dia.prep_features(va, feats["cheap"]).to_numpy(),
            "xc": dia.prep_features(te, feats["cheap"]).to_numpy(),
            "groups_val": va["patient_nbr"].to_numpy(), "cost_ratio": t_e / t_c}


def main():
    rows = []
    for name, build in [("soccer", build_soccer), ("diabetes", build_diabetes)]:
        rows += run_domain(name, build())

    out_dir = Path("paper/src_extension")
    with open(out_dir / "selective_topk_diagnostic_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "selective_topk_diagnostic_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(table[["domain", "bound", "alpha", "bound_coverage", "k_fraction", "k", "certified", "call_fraction_C",
                 "overlap_B", "overlap_C", "overlap_D_random", "overlap_E_uncertainty",
                 "regret_B", "regret_C", "regret_D_random", "regret_E_uncertainty"]].round(4).to_string(index=False))
    print(f"\nSaved to {out_dir / 'selective_topk_diagnostic_results.json'} (+ .csv)")


if __name__ == "__main__":
    main()

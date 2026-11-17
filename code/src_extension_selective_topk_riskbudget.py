"""Set-level guarantee for certified top-k with selective inference.

The per-tuple certificate of src_extension_selective_topk_diagnostic.py uses
bounds that each hold with probability 1-alpha, so across thousands of
unrefined tuples some bounds fail and a "certified" answer can be wrong
(diabetes, k = 1%: overlap 0.77).

Risk-budget certificate. Let s be the current scores (U_e for refined tuples,
U_c otherwise) and theta the midpoint between the k-th and (k+1)-th score.
The returned top-k is correct if no unrefined tuple's U_e lies on the other
side of theta. For an unrefined tuple that needs |P_e - P_c| >= d_i with
d_i = |s_i - theta| / (R_i + C_i); its conformal p-value r_i estimates that
probability (residuals normalized by a per-bin scale, pooled across bins so the
p-value floor is 1/(n_cal+1)). The number of tuples on the wrong side has
expectation at most sum_i r_i, and each wrong-side tuple misplaces at most one
tuple of the answer, so refining the riskiest tuples until
sum_i r_i <= eps * k targets an expected miss rate of at most eps. theta is
chosen from the data, so this is an approximate guarantee; it is checked
empirically over five splits per domain.

Budget-Mondrian variant. The pooled p-value is calibrated on all validation
tuples, but the tuples near a small-k cutoff are mostly high-budget ones, whose
|P_e - P_c| can be systematically larger (diabetes: R grows with prior
inpatient visits, a feature the cheap model lacks). The Mondrian variant
computes each p-value only against validation tuples in the same budget
(R + C) cell, cells cut at the 50/80/95% validation quantiles.

Rank-Mondrian variant (approximates the selection-conditional calibration of
Jin & Ren, "Confidence on the Focal", JRSS-B 2025). Selection is by rank of
U_c, so the validation tuples are ranked by their own U_c with the same rule
and each test tuple's p-value uses only validation tuples at a similar rank
position (cells cut at the 50/80/90/95/98/99% rank quantiles).

Scale-normalized variant. Within a rank cell the residual still grows with R
when R is built from features the cheap model lacks, so the residual is
normalized by a per-tuple scale s(x) predicted from information available
without the expensive model (cheap features, P_c, R, C). s is fit on half of
the validation tuples and the normalized residuals of the other half give
pooled p-values.
"""
import lightgbm as lgb
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src_extension_selective_topk_diagnostic import (
    K_FRACTIONS, build_diabetes, build_soccer, certify, compare, conformal_halfwidth, topk, utility,
)
from src_extension_selective_topk_homecredit import build as build_home_credit

SEEDS = [42, 1, 2, 3, 4]
EPSILONS = [0.01, 0.05]
N_BINS = 10


class PooledConformal:
    def __init__(self, pc_val, pe_val):
        self.edges = np.unique(np.quantile(pc_val, np.linspace(0, 1, N_BINS + 1)))
        resid = np.abs(pe_val - pc_val)
        b = self._bin(pc_val)
        self.scale = np.array([np.quantile(resid[b == j], 0.9) + 1e-9 for j in range(len(self.edges) - 1)])
        self.z_sorted = np.sort(resid / self.scale[b])

    def _bin(self, p):
        return np.clip(np.searchsorted(self.edges, p, side="right") - 1, 0, len(self.edges) - 2)

    def pvalue(self, pc, budget, d, idx=None):
        z = d / self.scale[self._bin(pc)]
        n_ge = len(self.z_sorted) - np.searchsorted(self.z_sorted, z, side="left")
        return (1 + n_ge) / (len(self.z_sorted) + 1)


class BudgetMondrianConformal(PooledConformal):
    CUTS = [0.50, 0.80, 0.95]

    def __init__(self, pc_val, pe_val, budget_val):
        super().__init__(pc_val, pe_val)
        self.budget_edges = np.quantile(budget_val, self.CUTS)
        z = np.abs(pe_val - pc_val) / self.scale[self._bin(pc_val)]
        cell = np.searchsorted(self.budget_edges, budget_val, side="right")
        self.z_by_cell = [np.sort(z[cell == c]) for c in range(len(self.CUTS) + 1)]

    def pvalue(self, pc, budget, d, idx=None):
        z = d / self.scale[self._bin(pc)]
        cell = np.searchsorted(self.budget_edges, budget, side="right")
        p = np.empty(len(z))
        for c, zs in enumerate(self.z_by_cell):
            m = cell == c
            n_ge = len(zs) - np.searchsorted(zs, z[m], side="left")
            p[m] = (1 + n_ge) / (len(zs) + 1)
        return p


def rank_quantile(u):
    return np.argsort(np.argsort(u)) / (len(u) - 1)


class RankMondrianConformal(PooledConformal):
    CUTS = [0.50, 0.80, 0.90, 0.95, 0.98, 0.99]

    def __init__(self, pc_val, pe_val, U_val, U_test):
        super().__init__(pc_val, pe_val)
        z = np.abs(pe_val - pc_val) / self.scale[self._bin(pc_val)]
        cell_val = np.searchsorted(self.CUTS, rank_quantile(U_val), side="right")
        self.z_by_cell = [np.sort(z[cell_val == c]) for c in range(len(self.CUTS) + 1)]
        self.cell_test = np.searchsorted(self.CUTS, rank_quantile(U_test), side="right")

    def pvalue(self, pc, budget, d, idx=None):
        z = d / self.scale[self._bin(pc)]
        cell = self.cell_test[idx]
        p = np.empty(len(z))
        for c, zs in enumerate(self.z_by_cell):
            m = cell == c
            n_ge = len(zs) - np.searchsorted(zs, z[m], side="left")
            p[m] = (1 + n_ge) / (len(zs) + 1)
        return p


class ScaleNormalizedConformal:
    def __init__(self, d):
        rng = np.random.default_rng(42)
        n = len(d["pc_val"])
        fit = rng.random(n) < 0.5
        Xv = self._X(d["xc_val"], d["pc_val"], d["R_val"], d["C_val"])
        resid = np.abs(d["pe_val"] - d["pc_val"])
        self.model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                       min_child_samples=50, verbosity=-1, random_state=42)
        self.model.fit(Xv[fit], resid[fit])
        self.floor = np.quantile(resid[fit], 0.05) + 1e-6
        s_cal = np.maximum(self.model.predict(Xv[~fit]), self.floor)
        self.z_sorted = np.sort(resid[~fit] / s_cal)
        self.s_test = np.maximum(self.model.predict(self._X(d["xc"], d["pc"], d["R"], d["C"])), self.floor)

    @staticmethod
    def _X(xc, pc, R, C):
        return np.column_stack([np.asarray(xc, dtype=float), pc, R, C])

    def pvalue(self, pc, budget, d, idx=None):
        z = d / self.s_test[idx]
        n_ge = len(self.z_sorted) - np.searchsorted(self.z_sorted, z, side="left")
        return (1 + n_ge) / (len(self.z_sorted) + 1)


def risk_budget(U_c, U_e, pc, R, C, conf, k, eps):
    s = U_c.copy()
    refined = np.zeros(len(s), dtype=bool)
    batch = max(1, math.ceil(0.01 * k))
    budget = R + C
    while True:
        part = -np.partition(-s, [k - 1, k])
        theta = 0.5 * (part[k - 1] + part[k])
        unref = np.flatnonzero(~refined)
        r = conf.pvalue(pc[unref], budget[unref], np.abs(s[unref] - theta) / budget[unref], unref)
        risk = float(r.sum())
        if risk <= eps * k or len(unref) == 0:
            return topk(s, k), refined, risk
        pick = unref[np.argsort(-r)[:batch]]
        s[pick] = U_e[pick]
        refined[pick] = True


def run(domain, seed, d):
    pc, pe, R, C = d["pc"], d["pe"], d["R"], d["C"]
    U_c, U_e = utility(pc, R, C), utility(pe, R, C)
    n = len(pc)
    confs = {"risk_budget": PooledConformal(d["pc_val"], d["pe_val"]),
             "risk_budget_mondrian": BudgetMondrianConformal(d["pc_val"], d["pe_val"], d["R_val"] + d["C_val"]),
             "risk_budget_rank": RankMondrianConformal(
                 d["pc_val"], d["pe_val"], utility(d["pc_val"], d["R_val"], d["C_val"]), U_c),
             "risk_budget_scale": ScaleNormalizedConformal(d)}
    w_old = (R + C) * conformal_halfwidth(d["pc_val"], d["pe_val"], pc, 0.05)
    rows = []
    for k_frac in K_FRACTIONS:
        k = int(round(k_frac * n))
        ref = topk(U_e, k)
        T, refined, _, _ = certify(U_c, U_e, w_old, k)
        ov, _ = compare(T, ref, U_e, k)
        rows.append({"domain": domain, "seed": seed, "method": "per_tuple_alpha0.05", "eps": None,
                     "k_fraction": k_frac, "k": k, "call_fraction": float(refined.mean()), "miss_rate": 1 - ov,
                     "risk_bound_over_k": None})
        for method, conf in confs.items():
            for eps in EPSILONS:
                T, refined, risk = risk_budget(U_c, U_e, pc, R, C, conf, k, eps)
                ov, rg = compare(T, ref, U_e, k)
                rows.append({"domain": domain, "seed": seed, "method": method, "eps": eps,
                             "k_fraction": k_frac, "k": k, "call_fraction": float(refined.mean()),
                             "miss_rate": 1 - ov, "risk_bound_over_k": risk / k, "regret": rg})
    return rows


def main():
    rows = []
    con = duckdb.connect()
    for seed in SEEDS:
        print(f"===== seed {seed} =====")
        rows += run("soccer", seed, build_soccer(seed))
        rows += run("diabetes", seed, build_diabetes(seed))
        rows += run("home_credit", seed, build_home_credit(con, seed))

    out_dir = Path("paper/src_extension")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "selective_topk_riskbudget_results.csv", index=False)
    with open(out_dir / "selective_topk_riskbudget_results.json", "w") as f:
        json.dump(rows, f, indent=2)

    table["eps_key"] = table["eps"].fillna(-1)
    summary = table.groupby(["domain", "method", "eps_key", "k_fraction"]).agg(
        calls_mean=("call_fraction", "mean"), miss_mean=("miss_rate", "mean"), miss_max=("miss_rate", "max"),
    ).reset_index()
    summary["eps"] = summary["eps_key"].where(summary["eps_key"] >= 0)
    summary["seeds_over_eps"] = [
        int((table[(table.domain == r.domain) & (table.method == r.method) & (table.eps_key == r.eps_key)
                   & (table.k_fraction == r.k_fraction)]["miss_rate"] > (r.eps if r.eps == r.eps else 0.05)).sum())
        for r in summary.itertuples()
    ]
    summary.drop(columns="eps_key").to_csv(out_dir / "selective_topk_riskbudget_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.drop(columns="eps_key").round(4).to_string(index=False))
    print(f"\nSaved to {out_dir / 'selective_topk_riskbudget_results.json'} (+ .csv, _summary.csv)")


if __name__ == "__main__":
    main()

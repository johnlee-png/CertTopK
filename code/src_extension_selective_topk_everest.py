"""Everest baseline (Lai et al., "Top-K Deep Video Analytics: A Probabilistic
Approach", SIGMOD 2021) on the same splits and queries as
src_extension_selective_topk_crc.py.

Everest, as described in its Section 3.3:
  * every uncleaned tuple f has a score distribution with CDF F_f, tuples are
    assumed independent;
  * the answer R_i is the top-K of the cleaned (oracle-verified) tuples
    ("certain-result condition");
  * confidence p_i = prod over uncleaned f of F_f(S_k), S_k = K-th score of R_i
    (its Eq. 2);
  * the next tuple to clean maximizes E[X_f], the expected confidence after
    cleaning f (its Eqs. 4-6), with S_p the (K-1)-th score:
      E[X_f] = F_f(S_k) G_f(S_k) + int_{S_k}^{S_p} G_f(s) dF_f(s) + (1 - F_f(S_p)) G_f(S_p),
      G_f(t) = prod over uncleaned g != f of F_g(t);
  * stop when p_i >= thres. Its guarantee: Pr(R_i = true top-K) >= thres,
    under the assumed score distributions.

Adaptation to this setting (stated in the paper text as well):
  * Score model. U_e = U_c + (R + C)(P_e - P_c) (affine identity), and
    P_e - P_c = s(x) * z with s the scale model of the CRC script fit on one half
    of the fit split and z drawn from the signed normalized residuals of the
    other half. So F_f(t) = F_z((t - U_c,f) / ((R_f + C_f) s_f)), continuous
    instead of Everest's quantized mixture.
  * Start. Everest starts from frames already labeled during training; here
    the cheap top-K are cleaned first so that a certain answer exists.
  * The integral is evaluated on 32 points between S_k and S_p, and E[X_f] is
    computed exactly for every uncleaned tuple (no early-stopping bound).
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src_extension_selective_topk_crc import DOMAINS, K_FRACTIONS, kept_rows, topk
from src_extension_selective_topk_diagnostic import utility

THRESHOLDS = [0.5, 0.9, 0.99]
N_INT = 32


def fit_score_model(fit, seed):
    """Scale model on one half of the fit queries, residual distribution on the other half."""
    q = np.unique(fit["qid"])
    rng = np.random.default_rng(seed + 11)
    half = np.isin(fit["qid"], rng.choice(q, len(q) // 2, replace=False))
    X = np.column_stack([fit["xc"], fit["pc"], fit["R"], fit["C"]])
    resid = fit["pe"] - fit["pc"]
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                              verbosity=-1, random_state=42)
    model.fit(X[half], np.abs(resid[half]))
    floor = float(np.quantile(np.abs(resid[half]), 0.05)) + 1e-6
    s_other = np.maximum(model.predict(X[~half]), floor)
    z = np.sort(resid[~half] / s_other)
    scale = lambda d: np.maximum(model.predict(np.column_stack([d["xc"], d["pc"], d["R"], d["C"]])), floor)
    return scale, z


def make_cdf(z):
    n = len(z)

    def F(x):
        return np.searchsorted(z, x, side="right") / n
    return F


def everest_query(uc, ue, width, k, Fz, thres):
    """Returns (answer indices, number cleaned, final confidence)."""
    n = len(uc)
    cleaned = np.zeros(n, dtype=bool)
    cleaned[topk(uc, k)] = True

    def cdf(idx, t):
        return Fz((np.asarray(t)[None, :] - uc[idx, None]) / width[idx, None])

    # F_f on the grid depends only on (s_k, s_p) and the tuple, so it is computed
    # again only when the cleaned tuple changes the top-k boundary; otherwise the
    # cleaned tuple's row is dropped. Same numbers as recomputing every step.
    key, u_idx, Fg, logF = None, None, None, None
    while True:
        c_idx = np.flatnonzero(cleaned)
        order = c_idx[np.argsort(-ue[c_idx], kind="stable")]  # ties: smaller index first, as in crc.topk
        answer = order[:k]
        s_k, s_p = ue[order[k - 1]], ue[order[k - 2]] if k >= 2 else ue[order[k - 1]]
        if (s_k, s_p) != key:
            key = (s_k, s_p)
            u_idx = np.flatnonzero(~cleaned)
            if len(u_idx) == 0:
                return answer, int(cleaned.sum()), 1.0
            grid = np.linspace(s_k, s_p, N_INT) if s_p > s_k else np.array([s_k, s_k])
            Fg = np.clip(cdf(u_idx, grid), 1e-300, 1.0)
            logF = np.log(Fg)
        if len(u_idx) == 0:
            return answer, int(cleaned.sum()), 1.0
        conf = float(np.exp(logF[:, 0].sum()))
        if conf >= thres:
            return answer, int(cleaned.sum()), conf
        logG = logF.sum(axis=0)[None, :] - logF
        G = np.exp(logG)
        mass = np.diff(Fg, axis=1)
        mid = 0.5 * (G[:, 1:] + G[:, :-1])
        ex = Fg[:, 0] * G[:, 0] + (mass * mid).sum(axis=1) + (1 - Fg[:, -1]) * G[:, -1]
        pos = int(np.argmax(ex))
        cleaned[u_idx[pos]] = True
        u_idx, Fg, logF = np.delete(u_idx, pos), np.delete(Fg, pos, axis=0), np.delete(logF, pos, axis=0)


def run(domain, seed):
    data = DOMAINS[domain](seed)
    scale, z = fit_score_model(data["fit"], seed)
    Fz = make_cdf(z)
    d = data["test"]
    U_c, U_e = utility(d["pc"], d["R"], d["C"]), utility(d["pe"], d["R"], d["C"])
    width = (d["R"] + d["C"]) * scale(d)
    rows = []
    for k_frac in K_FRACTIONS:
        for thres in THRESHOLDS:
            miss, calls, exact, confs = [], [], [], []
            for q in np.unique(d["qid"]):
                ix = np.flatnonzero(d["qid"] == q)
                k = max(1, int(round(k_frac * len(ix))))
                if k >= len(ix):
                    continue
                ans, n_clean, conf = everest_query(U_c[ix], U_e[ix], width[ix], k, Fz, thres)
                ref = set(topk(U_e[ix], k).tolist())
                hit = len(ref & set(ans.tolist()))
                miss.append(1 - hit / k)
                exact.append(float(hit == k))
                calls.append(n_clean / len(ix))
                confs.append(conf)
            rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "thres": thres,
                         "n_test_queries": len(miss), "test_miss": float(np.mean(miss)),
                         "exact_rate": float(np.mean(exact)), "claimed_conf": float(np.mean(confs)),
                         "test_call_fraction": float(np.mean(calls))})
    return rows


def main():
    seeds = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else list(DOMAINS)
    out_dir = Path("paper/src_extension")
    keep = kept_rows(out_dir / "selective_topk_everest_results.csv", domains)
    rows = []
    for domain in domains:
        for seed in seeds:
            t0 = time.time()
            rows += run(domain, seed)
            print(f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s", flush=True)
            pd.concat([keep, pd.DataFrame(rows)]).to_csv(out_dir / "selective_topk_everest_results.csv", index=False)
    t = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
    t.to_json(out_dir / "selective_topk_everest_results.json", orient="records", indent=2)
    summ = t.groupby(["domain", "k_fraction", "thres"]).agg(
        seeds=("seed", "nunique"), miss_mean=("test_miss", "mean"), exact_rate=("exact_rate", "mean"),
        claimed_conf=("claimed_conf", "mean"), calls_mean=("test_call_fraction", "mean")).reset_index()
    summ.to_csv(out_dir / "selective_topk_everest_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

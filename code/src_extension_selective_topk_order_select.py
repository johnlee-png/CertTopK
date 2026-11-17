"""M2: choose the refinement order on the fit split only, then calibrate it with CRC.

Candidate orders: the static priorities P2, P1, TOPM and Everest's adaptive
order. For each (domain, seed, k, eps) the order is chosen using only the fit
queries (part of D_0): for every order the fit queries' loss curves are
computed, the plug-in lambda (first grid value whose mean fit envelope is
<= eps) is found, and the order with the smallest mean fit call fraction at
that lambda is chosen. Because the choice is a function of D_0 alone, the
chosen rule family is a function of D_0 and Theorem 1 holds for it unchanged
(the calibration and test queries are never looked at). The chosen order's
calibration and test results are exactly those of its own CRC run, so they are
read from selective_topk_crc_results.csv (P2, P1, TOPM) and
selective_topk_everest_crc_results.csv (Everest order); "best in hindsight"
is the cheapest of the four on the test queries among those meeting
mean miss <= eps.
Writes paper/src_extension/selective_topk_order_select_{choices,summary}.csv.
Usage: py src_extension_selective_topk_order_select.py <n_seeds> [domains] [workers]   (fit curves)
       py src_extension_selective_topk_order_select.py summarize                         (after the CRC runs)
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
import src_extension_selective_topk_everest as ev
import src_extension_selective_topk_everest_crc as evc

STATIC = ["P2", "P1", "TOPM"]
OUT = Path("paper/src_extension")


def plug_in(Lfit_env, Ffit, eps):
    ok = np.flatnonzero(Lfit_env.mean(axis=0) <= eps)
    j = ok[0] if len(ok) else Lfit_env.shape[1] - 1
    return float(Ffit[:, j].mean())


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    escale, z = ev.fit_score_model(data["fit"], seed)
    Fz = ev.make_cdf(z)
    fitd = data["fit"]
    width = (fitd["R"] + fitd["C"]) * escale(fitd)
    rows = []
    for k_frac in crc.K_FRACTIONS:
        fit = crc.query_arrays(fitd, scale, k_frac, seed, stream=0)
        curves = {}
        for name in STATIC:
            pooled = np.concatenate([q["pri"][name] for q in fit])
            grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
            cs = [crc.loss_curve(q, name, grid) for q in fit]
            curves[name] = (np.array([crc.envelope(c[0]) for c in cs]), np.array([c[1] for c in cs]))
        L, F = evc.curves(fitd, k_frac, Fz, width)
        curves["EVEREST_CRC"] = (np.array([crc.envelope(l) for l in L]), F)
        for eps in crc.EPSILONS:
            fit_calls = {name: plug_in(Le, Fc, eps) for name, (Le, Fc) in curves.items()}
            rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps,
                         "chosen": min(fit_calls, key=fit_calls.get),
                         **{f"fit_calls_{n}": v for n, v in fit_calls.items()}})
    return rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def summarize():
    ch = pd.read_csv(OUT / "selective_topk_order_select_choices.csv")
    crc_r = pd.read_csv(OUT / "selective_topk_crc_results.csv")
    evc_r = pd.read_csv(OUT / "selective_topk_everest_crc_results.csv")
    key = ["domain", "seed", "k_fraction", "eps"]
    res = pd.concat([crc_r[crc_r.priority.isin(STATIC)][key + ["priority", "test_miss", "test_call_fraction"]],
                     evc_r[key + ["priority", "test_miss", "test_call_fraction"]]], ignore_index=True)
    piv_calls = res.pivot_table(index=key, columns="priority", values="test_call_fraction")
    piv_miss = res.pivot_table(index=key, columns="priority", values="test_miss")
    rows = []
    for _, c in ch.iterrows():
        k = (c.domain, c.seed, c.k_fraction, c.eps)
        rows.append({**{n: c[n] for n in key}, "chosen": c.chosen,
                     "chosen_calls": piv_calls.loc[k, c.chosen], "chosen_miss": piv_miss.loc[k, c.chosen],
                     **{f"calls_{n}": piv_calls.loc[k, n] for n in piv_calls.columns}})
    t = pd.DataFrame(rows)
    order_cols = [f"calls_{n}" for n in piv_calls.columns]
    g = t.groupby(["domain", "k_fraction", "eps"])
    summ = g.agg(chosen_calls=("chosen_calls", "mean"), chosen_miss=("chosen_miss", "mean"),
                 **{c: (c, "mean") for c in order_cols}).reset_index()
    # best fixed order in hindsight (by mean test calls over seeds; every order meets the guarantee on average)
    summ["best_fixed_order"] = summ[order_cols].idxmin(axis=1).str.replace("calls_", "")
    summ["best_fixed_calls"] = summ[order_cols].min(axis=1)
    summ["ratio_chosen_to_best"] = summ.chosen_calls / summ.best_fixed_calls
    summ["most_chosen"] = g.chosen.agg(lambda s: s.value_counts().idxmax()).to_numpy()
    summ["share_most_chosen"] = g.chosen.agg(lambda s: s.value_counts(normalize=True).max()).to_numpy()
    summ.to_csv(OUT / "selective_topk_order_select_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        return summarize()
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else list(crc.DOMAINS)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    keep = crc.kept_rows(OUT / "selective_topk_order_select_choices.csv", domains)
    rows = []
    with Pool(workers) as pool:
        for r, msg in pool.imap_unordered(task, [(d, s) for d in domains for s in range(n_seeds)]):
            rows += r
            print(msg, flush=True)
            pd.concat([keep, pd.DataFrame(rows)]).to_csv(OUT / "selective_topk_order_select_choices.csv", index=False)


if __name__ == "__main__":
    main()

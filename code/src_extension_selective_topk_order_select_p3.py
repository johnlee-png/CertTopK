"""M2 with the loss-aware priority P3 added to the candidate orders {P2, P1, TOPM, Everest order}.

For each (domain, seed) it computes P3's fit-split plug-in call fraction exactly as order_select.py does for the other
orders (same fit queries, same grid rule), merges it into selective_topk_order_select_choices.csv, and re-chooses the
order with the smallest fit call fraction. The choice uses the fit split only, so Theorem 1 holds for the chosen rule.
Test results of the chosen order come from its own CRC run (crc / everest_crc / lossaware results files).
Writes paper/src_extension/selective_topk_order_select_p3_{choices,summary}.csv.
Usage: py src_extension_selective_topk_order_select_p3.py <n_seeds> [domains] [workers] | summarize
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
import src_extension_selective_topk_lossaware as la
from src_extension_selective_topk_order_select import plug_in

OUT = Path("paper/src_extension")


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    f = data["fit"]
    z_pool = (f["pe"] - f["pc"]) / scale(f)
    rows = []
    for k_frac in crc.K_FRACTIONS:
        fit = crc.query_arrays(f, scale, k_frac, seed, stream=0)
        la.add_p3(fit, f, scale, z_pool, k_frac, np.random.default_rng([seed, 0, 99]))
        pooled = np.concatenate([q["pri"]["P3"] for q in fit])
        grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
        cs = [crc.loss_curve(q, "P3", grid) for q in fit]
        le, fc = np.array([crc.envelope(c[0]) for c in cs]), np.array([c[1] for c in cs])
        for eps in crc.EPSILONS:
            rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps,
                         "fit_calls_P3": plug_in(le, fc, eps)})
    return rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def summarize():
    key = ["domain", "seed", "k_fraction", "eps"]
    ch = pd.read_csv(OUT / "selective_topk_order_select_choices.csv")
    p3 = pd.read_csv(OUT / "selective_topk_order_select_p3_fitcalls.csv")
    ch = ch.merge(p3, on=key, how="inner")
    cols = [c for c in ch.columns if c.startswith("fit_calls_")]
    ch["chosen"] = ch[cols].idxmin(axis=1).str.replace("fit_calls_", "")
    ch.to_csv(OUT / "selective_topk_order_select_p3_choices.csv", index=False)
    crc_r = pd.read_csv(OUT / "selective_topk_crc_results.csv")
    evc_r = pd.read_csv(OUT / "selective_topk_everest_crc_results.csv")
    la_r = pd.read_csv(OUT / "selective_topk_lossaware_results.csv")
    res = pd.concat([crc_r[crc_r.priority.isin(["P2", "P1", "TOPM"])][key + ["priority", "test_miss", "test_call_fraction"]],
                     evc_r[key + ["priority", "test_miss", "test_call_fraction"]],
                     la_r[la_r.priority == "P3"][key + ["priority", "test_miss", "test_call_fraction"]]], ignore_index=True)
    pc = res.pivot_table(index=key, columns="priority", values="test_call_fraction")
    pm = res.pivot_table(index=key, columns="priority", values="test_miss")
    rows = []
    for _, c in ch.iterrows():
        k = (c.domain, c.seed, c.k_fraction, c.eps)
        rows.append({**{n: c[n] for n in key}, "chosen": c.chosen, "chosen_calls": pc.loc[k, c.chosen],
                     "chosen_miss": pm.loc[k, c.chosen], **{f"calls_{n}": pc.loc[k, n] for n in pc.columns}})
    t = pd.DataFrame(rows)
    oc = [f"calls_{n}" for n in pc.columns]
    g = t.groupby(["domain", "k_fraction", "eps"])
    summ = g.agg(chosen_calls=("chosen_calls", "mean"), chosen_miss=("chosen_miss", "mean"),
                 **{c: (c, "mean") for c in oc}).reset_index()
    summ["best_fixed_order"] = summ[oc].idxmin(axis=1).str.replace("calls_", "")
    summ["best_fixed_calls"] = summ[oc].min(axis=1)
    summ["ratio_chosen_to_best"] = summ.chosen_calls / summ.best_fixed_calls
    summ["ratio_chosen_to_P2"] = summ.chosen_calls / summ.calls_P2
    summ["most_chosen"] = g.chosen.agg(lambda s: s.value_counts().idxmax()).to_numpy()
    summ["share_most_chosen"] = g.chosen.agg(lambda s: s.value_counts(normalize=True).max()).to_numpy()
    summ.to_csv(OUT / "selective_topk_order_select_p3_summary.csv", index=False)
    print(summ[["domain", "k_fraction", "eps", "most_chosen", "share_most_chosen", "chosen_calls", "calls_P2",
                "best_fixed_order", "ratio_chosen_to_best", "chosen_miss"]].round(3).to_string(index=False))
    print("cells", len(summ), "miss<=eps", int((summ.chosen_miss <= summ.eps).sum()),
          "ratio to best median/mean/max", round(summ.ratio_chosen_to_best.median(), 3),
          round(summ.ratio_chosen_to_best.mean(), 3), round(summ.ratio_chosen_to_best.max(), 3),
          "ratio to P2 median/mean", round(summ.ratio_chosen_to_P2.median(), 3), round(summ.ratio_chosen_to_P2.mean(), 3))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        return summarize()
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else ["soccer", "diabetes", "home_credit", "kdd98", "telco",
                                                                "ieee_fraud", "lending_club"]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    rows = []
    with Pool(workers) as pool:
        for r, msg in pool.imap_unordered(task, [(d, s) for d in domains for s in range(n_seeds)]):
            rows += r
            print(msg, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "selective_topk_order_select_p3_fitcalls.csv", index=False)
    summarize()


if __name__ == "__main__":
    main()

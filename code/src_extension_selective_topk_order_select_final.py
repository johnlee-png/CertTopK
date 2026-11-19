"""Final operator: the order is chosen on the fit split among
{P2, P3, P3c, P1, TOPM, Everest order}, where P3c is P3 with the simulated probabilities clipped to [0, 1], and then
calibrated with CRC. Adds the fit-split plug-in calls of P3 and P3c (same rule as order_select.py) for all eight
workloads, merges them with selective_topk_order_select_choices.csv, re-chooses by minimum fit calls, and reads the test
results of the chosen order from its own CRC run (crc / everest_crc / lossaware / c5 P3CLIP).
Two objectives: "calls" (minimum fit call fraction) and, where per-candidate costs are measured, "time"
(fit call fraction x expensive cost + priority cost), see summarize().
Writes paper/src_extension/selective_topk_order_select_final_{fitcalls,choices,summary}.csv.
Usage: py src_extension_selective_topk_order_select_final.py <n_seeds> [domains] [workers] | summarize
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_order_select import plug_in
from src_extension_selective_topk_p3_extra import arrays, iid_sampler, p3_priority
from src_extension_selective_topk_baselines import topk_mask
from src_vldb_topk_ties import flip_prob

OUT = Path("paper/src_extension")
B = 200
KEY = ["domain", "seed", "k_fraction", "eps"]


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    f = data["fit"]
    z_all = (f["pe"] - f["pc"]) / scale(f)
    s_all = scale(f)
    width = (f["R"] + f["C"]) * s_all
    rows = []
    for k_frac in crc.K_FRACTIONS:
        fit = arrays(f, scale, k_frac, seed, 0)
        rng = np.random.default_rng([seed, 0, 99])
        rng_c = np.random.default_rng([seed, 0, 499])
        for q in fit:
            ix, k = q["_ix"], q["k"]
            q["pri"]["P3"] = p3_priority(q["uc"], width[ix], k, q["pri"]["P2"], B, iid_sampler(z_all), rng)
            p_sim = f["pc"][ix][None, :] + s_all[ix][None, :] * rng_c.choice(z_all, size=(B, len(ix)))
            u_sim = np.clip(p_sim, 0, 1) * (f["R"] + f["C"])[ix][None, :] - f["C"][ix][None, :]
            flip = flip_prob(u_sim, topk_mask(q["uc"], k), k)
            p2 = q["pri"]["P2"]
            q["pri"]["P3CLIP"] = 1.0 - flip + 1e-6 * p2 / (p2.max() + 1e-12)
        r = {eps: {} for eps in crc.EPSILONS}
        for name in ("P3", "P3CLIP"):
            pooled = np.concatenate([q["pri"][name] for q in fit])
            grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
            cs = [crc.loss_curve(q, name, grid) for q in fit]
            le, fc = np.array([crc.envelope(c[0]) for c in cs]), np.array([c[1] for c in cs])
            for eps in crc.EPSILONS:
                r[eps][f"fit_calls_{name}"] = plug_in(le, fc, eps)
        for eps in crc.EPSILONS:
            rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps, **r[eps]})
    return rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def test_results():
    crc_r = pd.read_csv(OUT / "selective_topk_crc_results.csv")
    evc = pd.read_csv(OUT / "selective_topk_everest_crc_results.csv")
    la = pd.read_csv(OUT / "selective_topk_lossaware_results.csv")
    c5 = pd.read_csv(OUT / "selective_topk_c5_results.csv").rename(columns={"test_calls": "test_call_fraction"})
    c5 = c5[(c5.method == "CRC") & (c5.priority == "P3CLIP")]
    cols = KEY + ["priority", "test_miss", "test_call_fraction"]
    return pd.concat([crc_r[crc_r.priority.isin(["P2", "P1", "TOPM"])][cols], evc[cols], la[la.priority == "P3"][cols],
                      c5[cols]], ignore_index=True)


def summarize():
    ch = pd.read_csv(OUT / "selective_topk_order_select_choices.csv")
    ch = ch[[c for c in ch.columns if c in KEY or c.startswith("fit_calls_")]]
    fc = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("selective_topk_order_select_final_fitcalls*.csv"))],
                   ignore_index=True).drop_duplicates(KEY)
    ch = ch.merge(fc, on=KEY, how="inner")
    cols = [c for c in ch.columns if c.startswith("fit_calls_")]
    ch["chosen"] = ch[cols].idxmin(axis=1).str.replace("fit_calls_", "")
    ch.to_csv(OUT / "selective_topk_order_select_final_choices.csv", index=False)
    res = test_results()
    pc = res.pivot_table(index=KEY, columns="priority", values="test_call_fraction")
    pm = res.pivot_table(index=KEY, columns="priority", values="test_miss")
    rows = []
    for _, c in ch.iterrows():
        k = (c.domain, c.seed, c.k_fraction, c.eps)
        rows.append({**{n: c[n] for n in KEY}, "chosen": c.chosen, "chosen_calls": pc.loc[k, c.chosen],
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
    summ.to_csv(OUT / "selective_topk_order_select_final_summary.csv", index=False)
    with pd.option_context("display.width", 250):
        print(summ[["domain", "k_fraction", "eps", "most_chosen", "share_most_chosen", "chosen_calls", "calls_P2",
                    "best_fixed_order", "ratio_chosen_to_best", "chosen_miss"]].round(3).to_string(index=False))
    s7 = summ[summ.domain != "lc_llm"]
    for lab, s in (("7 datasets", s7), ("all", summ)):
        finite = s[np.isfinite(s.ratio_chosen_to_best)]
        print(lab, "cells", len(s), "miss<=eps", int((s.chosen_miss <= s.eps).sum()),
              "ratio to best median/mean/max", round(finite.ratio_chosen_to_best.median(), 3),
              round(finite.ratio_chosen_to_best.mean(), 3), round(finite.ratio_chosen_to_best.max(), 3),
              "ratio to P2 mean/median", round(s.ratio_chosen_to_P2.mean(), 3), round(s.ratio_chosen_to_P2.median(), 3))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        return summarize()
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else ["soccer", "diabetes", "home_credit", "kdd98", "telco",
                                                                "ieee_fraud", "lending_club", "lc_llm"]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    path = OUT / os.environ.get("FINAL_FITCALLS", "selective_topk_order_select_final_fitcalls.csv")
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["domain"], r["seed"]) for r in rows}
    todo = [(d, s) for d in domains for s in range(n_seeds) if (d, s) not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    with Pool(workers) as pool:
        for r, msg in pool.imap_unordered(task, todo):
            rows += r
            pd.DataFrame(rows).to_csv(path, index=False)
            print(msg, flush=True)
    summarize()


if __name__ == "__main__":
    main()

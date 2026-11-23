"""Across-seed summary of the end-to-end runs (Home Credit seeds 0-4; LC-LLM seeds 0-2 at concurrency 1, seed 0 also
at concurrency 4 and 8). Per configuration and eps: speed-up over ALL of each order and of Auto-calls / Auto-time /
the fastest order in hindsight, as mean and range over seeds; Auto-time's excess over the fastest order; the spread R
of the cost model's errors (Proposition 2) and whether Auto-time's time stayed within R of the best.
Writes paper/src_extension/e2e/e2e_seeds_{long,summary}.csv.
Usage: py src_vldb_e2e_seeds.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("paper/src_extension/e2e")
RUNS = [("hc", "path", s, "hc" if s == 0 else f"hc_s{s}") for s in range(5)] + \
       [("llm", "conc", s, "llm_c1" if s == 0 else f"llm_c1_s{s}") for s in range(3)]
RULES = ["Auto-calls", "Auto-time", "Guarded", "Auto-val", "post-hoc best"]


def main():
    rows = []
    for wl, dim, seed, tag in RUNS:
        s = pd.read_csv(D / f"{tag}_summary.csv")
        for (size, v, eps), g in s[s.eps >= 0].groupby(["size", dim, "eps"]):
            g = g.set_index("policy")
            base = g.drop(index=RULES, errors="ignore")
            best = g.loc["post-hoc best", "mean_s"]
            r = base.mean_s / base.pred_s
            R = float(r.max() / r.min())
            rec = {"wl": wl, "seed": seed, "size": size, "cfg": v, "eps": eps, "R": R,
                   "all_s": g.loc["ALL", "mean_s"],
                   "at_excess": g.loc["Auto-time", "mean_s"] / g.loc["post-hoc best", "mean_s"] - 1,
                   "ac_over_at": g.loc["Auto-calls", "mean_s"] / g.loc["Auto-time", "mean_s"],
                   "p5_over_p3c": g.loc["P5", "mean_s"] / g.loc["P3c", "mean_s"],
                   "at_order": g.loc["Auto-time", "order"], "ac_order": g.loc["Auto-calls", "order"],
                   "at_loss": g.loc["Auto-time", "loss"]}
            for p in ["P0", "P2", "P3c", "P5"] + RULES:
                rec[f"sp_{p}"] = g.loc[p, "speedup_vs_all"] if p in g.index else np.nan
            for p in ("Guarded", "Auto-val"):
                rec[f"excess_{p}"] = g.loc[p, "mean_s"] / best - 1 if p in g.index else np.nan
                rec[f"order_{p}"] = g.loc[p, "order"] if p in g.index else None
            rec["within_R"] = g.loc["Auto-time", "mean_s"] <= R * g.loc["post-hoc best", "mean_s"] * (1 + 1e-9)
            rows.append(rec)
    long = pd.DataFrame(rows)
    long.to_csv(D / "e2e_seeds_long.csv", index=False)
    sp = [c for c in long.columns if c.startswith("sp_")]
    agg = long.groupby(["wl", "size", "cfg", "eps"]).agg(
        n_seeds=("seed", "nunique"), **{f"{c}_mean": (c, "mean") for c in sp}, **{f"{c}_min": (c, "min") for c in sp},
        **{f"{c}_max": (c, "max") for c in sp}, at_excess=("at_excess", "mean"), ac_over_at_max=("ac_over_at", "max"),
        p5_over_p3c=("p5_over_p3c", "mean"), R_med=("R", "median"), within_R=("within_R", "all")).reset_index()
    agg.to_csv(D / "e2e_seeds_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200):
        show = ["wl", "size", "cfg", "eps", "n_seeds", "sp_P0_mean", "sp_P3c_mean", "sp_P5_mean", "sp_Auto-calls_mean",
                "sp_Auto-time_mean", "sp_Auto-time_min", "sp_Auto-time_max", "sp_post-hoc best_mean", "at_excess",
                "p5_over_p3c", "R_med", "within_R"]
        print(agg[show].round(3).to_string(index=False))
        for wl in ("hc", "llm"):
            x = long[long.wl == wl]
            for eps in (0.05, 0.1):
                y = x[x.eps == eps]
                print(wl, eps, "AT speed-up %.2f-%.2f" % (y["sp_Auto-time"].min(), y["sp_Auto-time"].max()),
                      "| orders P0/P2/P3c/P5 %.2f-%.2f" % (y[["sp_P0", "sp_P2", "sp_P3c", "sp_P5"]].min().min(),
                                                          y[["sp_P0", "sp_P2", "sp_P3c", "sp_P5"]].max().max()),
                      "| AT excess mean %.3f max %.3f" % (y.at_excess.mean(), y.at_excess.max()),
                      "| AC/AT max %.2f" % y.ac_over_at.max(), "| P5/P3c time mean %.3f" % y.p5_over_p3c.mean(),
                      "| AT never slower than ALL:", bool((y["sp_Auto-time"] >= 1).all()),
                      "| within R:", bool(y.within_R.all()), "| R median %.2f" % y.R.median(),
                      "| AT loss max %.3f" % y.at_loss.max())
            print(wl, "Auto-time orders:", x.at_order.value_counts().to_dict(), "Auto-calls:", x.ac_order.value_counts().to_dict())


if __name__ == "__main__":
    main()

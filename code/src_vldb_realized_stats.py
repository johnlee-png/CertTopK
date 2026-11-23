"""Realized utility of cheap-only, certified (P2 + CRC), and full-expensive answers, with paired statistics.

From selective_topk_realized_results.csv (per test query: realized utility sum_{i in T} (y_i R_i - (1 - y_i) C_i)
of the cheap top-k, the certified top-k at eps = 0.05 and 0.10, the expensive top-k, and the label oracle). Per
domain and seed the query mean is taken; differences are paired over the 20 seeds (mean, 95% t-interval, seeds
with a positive difference), and every utility is also expressed as a share of the cheap-to-oracle range, so domains
with different units are comparable. This answers whether the expensive answer the operator certifies is worth more
than the cheap one in what actually happened.
Writes paper/src_extension/selective_topk_realized_stats.csv.
Usage: py src_vldb_realized_stats.py   (cwd: the folder containing paper/src_extension)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path("paper/src_extension")


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def main():
    q = pd.read_csv(R / "selective_topk_realized_results.csv")
    s = q.groupby(["domain", "seed"])[["cheap", "expensive", "oracle", "proposed_eps0.05", "proposed_eps0.1"]].mean()
    rows = []
    for dom, g in s.groupby(level=0):
        span = (g.oracle - g.cheap).mean()
        r = {"domain": dom, "seeds": len(g)}
        for lab, col in (("expensive", "expensive"), ("certified_0.05", "proposed_eps0.05"),
                         ("certified_0.10", "proposed_eps0.1")):
            d = (g[col] - g.cheap) / span                       # gain over cheap, share of the cheap-oracle range
            m, lo, hi = ci(d)
            r.update({f"{lab}_gain": m, f"{lab}_lo": lo, f"{lab}_hi": hi, f"{lab}_seeds_pos": int((d > 0).sum())})
        gap = (g["proposed_eps0.05"] - g.expensive) / span        # certified minus expensive
        m, lo, hi = ci(gap)
        r.update({"cert05_minus_exp": m, "cert05_minus_exp_lo": lo, "cert05_minus_exp_hi": hi,
                  "p_exp_vs_cheap": stats.ttest_rel(g.expensive, g.cheap).pvalue})
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(R / "selective_topk_realized_stats.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print((out.set_index("domain")[["expensive_gain", "expensive_lo", "expensive_hi", "expensive_seeds_pos",
                                        "certified_0.05_gain", "certified_0.05_lo", "certified_0.05_hi",
                                        "cert05_minus_exp", "cert05_minus_exp_lo", "cert05_minus_exp_hi",
                                        "p_exp_vs_cheap"]] * [100, 100, 100, 1, 100, 100, 100, 100, 100, 100, 1]
               ).round(3).to_string())


if __name__ == "__main__":
    main()

"""Fixed orders vs automatic choice vs post-hoc best under the same CRC calibration.

Per (domain, seed, kappa, eps) cell, every candidate order is calibrated with CRC on the same calibration queries; the
automatic rule picks the order with the fewest fit-split plug-in calls (no calibration or test data). We compare its
test calls with P0 (the simplest order), P2, and the post-hoc best fixed order of the cell (chosen on the test data
itself, so not deployable), and report the mean, median and worst extra cost and the failure cells.

Two candidate sets: OLD = {P1, P2, TOPM, EVEREST_CRC, P3, P3CLIP} (the ICDE-era Auto-calls) and
NEW = OLD + {P0, PU}. Reads the reruns in <run_dir>/paper/src_extension.
Writes <run_dir>/paper/src_extension/auto_analysis_{cells,summary,overall,worst}.csv.
Usage: py src_vldb_auto_analysis.py [run_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
R = RUN / "paper/src_extension"
KEY = ["domain", "seed", "k_fraction", "eps"]
OLD = ["P1", "P2", "TOPM", "EVEREST_CRC", "P3", "P3CLIP"]
NEW = OLD + ["P0", "PU"]


def test_table():
    cols = KEY + ["priority", "test_miss", "test_call_fraction"]
    crc_r = pd.read_csv(R / "selective_topk_crc_results.csv")
    evc = pd.read_csv(R / "selective_topk_everest_crc_results.csv")
    la = pd.read_csv(R / "selective_topk_lossaware_results.csv")
    c5 = pd.read_csv(R / "selective_topk_c5_results.csv").rename(columns={"test_calls": "test_call_fraction"})
    c5 = c5[(c5.method == "CRC") & c5.priority.isin(["P3CLIP", "PU"])]
    t = pd.concat([crc_r[crc_r.priority.isin(["P0", "P1", "P2", "TOPM"])][cols], evc[cols], la[la.priority == "P3"][cols],
                   c5[cols]], ignore_index=True)
    return (t.pivot_table(index=KEY, columns="priority", values="test_call_fraction"),
            t.pivot_table(index=KEY, columns="priority", values="test_miss"))


def fit_table():
    a = pd.read_csv(R / "selective_topk_order_select_choices.csv")
    a = a[KEY + [c for c in a.columns if c.startswith("fit_calls_")]]
    b = pd.read_csv(R / "selective_topk_order_select_final_fitcalls.csv").drop_duplicates(KEY)
    c = pd.read_csv(R / "selective_topk_auto_ext_fitcalls.csv").drop_duplicates(KEY)
    return a.merge(b, on=KEY).merge(c, on=KEY).set_index(KEY)


def main():
    calls, miss = test_table()
    fit = fit_table()
    idx = calls.index.intersection(fit.index)
    calls, miss, fit = calls.loc[idx], miss.loc[idx], fit.loc[idx]
    cells = pd.DataFrame(index=idx)
    for lab, cand in (("old", OLD), ("new", NEW)):
        ch = fit[[f"fit_calls_{c}" for c in cand]].idxmin(axis=1).str.replace("fit_calls_", "")
        cells[f"chosen_{lab}"] = ch
        cells[f"calls_{lab}"] = [calls.loc[i, c] for i, c in zip(idx, ch)]
        cells[f"miss_{lab}"] = [miss.loc[i, c] for i, c in zip(idx, ch)]
    for c in NEW:
        cells[f"calls_{c}"] = calls[c]
    cells = cells.reset_index()
    # post-hoc best fixed order per (domain, kappa, eps) cell, averaged over seeds (not deployable)
    g = cells.groupby(["domain", "k_fraction", "eps"])
    mean_calls = g[[f"calls_{c}" for c in NEW]].mean()
    best = mean_calls.idxmin(axis=1).str.replace("calls_", "")
    summ = g.agg(**{f"calls_{lab}": (f"calls_{lab}", "mean") for lab in ("old", "new")},
                 **{f"miss_{lab}": (f"miss_{lab}", "mean") for lab in ("old", "new")},
                 **{f"calls_{c}": (f"calls_{c}", "mean") for c in NEW}).reset_index()
    summ["posthoc_best"] = best.to_numpy()
    summ["posthoc_calls"] = mean_calls.min(axis=1).to_numpy()
    for lab in ("old", "new"):
        for ref in ("P0", "P2"):
            summ[f"{lab}_vs_{ref}"] = summ[f"calls_{lab}"] / summ[f"calls_{ref}"] - 1
        summ[f"{lab}_vs_best"] = summ[f"calls_{lab}"] / summ.posthoc_calls - 1
    summ["P2_vs_P0"] = summ.calls_P2 / summ.calls_P0 - 1
    summ = summ.replace([np.inf, -np.inf], np.nan)
    cells.to_csv(R / "auto_analysis_cells.csv", index=False)
    summ.to_csv(R / "auto_analysis_summary.csv", index=False)
    seven = summ[summ.domain != "lc_llm"]
    rows = []
    for lab, s in (("7 datasets", seven), ("7 without soccer", seven[seven.domain != "soccer"]), ("all 8", summ)):
        for col in ("old_vs_P0", "old_vs_P2", "old_vs_best", "new_vs_P0", "new_vs_P2", "new_vs_best", "P2_vs_P0"):
            x = s[col].dropna()
            rows.append({"set": lab, "comparison": col, "cells": len(x), "mean": x.mean(), "median": x.median(),
                         "worst": x.max(), "share_worse": (x > 0.005).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(R / "auto_analysis_overall.csv", index=False)
    worst = summ.sort_values("new_vs_best", ascending=False).head(10)
    worst.to_csv(R / "auto_analysis_worst.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(out.round(4).to_string(index=False))
        print(summ.groupby("domain")[["new_vs_P0", "new_vs_P2", "new_vs_best", "P2_vs_P0"]].mean().round(4))
        print(worst[["domain", "k_fraction", "eps", "posthoc_best", "new_vs_best", "miss_new"]].round(3).to_string(index=False))
        print("miss <= eps (new):", int((summ.miss_new <= summ.eps + 1e-12).sum()), "of", len(summ))


if __name__ == "__main__":
    main()

"""Profiling for LC-LLM's Auto-time that uses no test query.

* LLM cost per call t_e: throughput of the local LLM server on an independent profiling workload, Lending Club loans
  2007-2014 with a borrower description that are NOT in the LC-LLM data (loans that had not finished, so they were
  excluded); same prompt and server settings (8 parallel slots).
* Per-candidate CPU costs, per seed, on that seed's fit split only: s_hat inference and each order's own priority
  computation (P1, TOPM, P2, P3, P3c). Cheap-model inference is paid by every order and does not affect the choice.
Then Auto-time is re-chosen for every (seed, kappa, eps) from the fit-split plug-in calls
(selective_topk_order_select_final_choices.csv) and these costs, and compared with the earlier choice.
Writes paper/src_extension/selective_topk_llm_profile_fit.{json,csv}.
Usage: py src_extension_selective_topk_llm_profile_fit.py [N_PROFILE] [port]
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
import src_extension_selective_topk_llm_trial as trial
from src_extension_selective_topk_hc_time_v2 import priority

N_PROFILE = int(sys.argv[1]) if len(sys.argv) > 1 else 400
trial.PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8091
OUT = Path("paper/src_extension/selective_topk_llm_profile_fit")
ORDERS = ["P1", "TOPM", "P2", "P3", "P3c"]
CAND = {"P2": "P2", "P3": "P3", "P3CLIP": "P3c", "P1": "P1", "TOPM": "TOPM"}


def llm_throughput():
    used = set(crc.lc_llm_frame()["id"].astype(str))
    parts = []
    for ch in pd.read_csv(trial.RAW, usecols=["id"] + trial.COLS, chunksize=200_000, low_memory=False):
        st = ch.loan_status.astype(str)
        ok = (~st.str.contains("Fully Paid|Charged Off")) & ch.desc.notna() & (ch.desc.astype(str).str.strip().str.len() > 30)
        ok &= ~ch["id"].astype(str).isin(used)
        parts.append(ch[ok])
    pool = pd.concat(parts)
    pool = pool.sample(n=min(N_PROFILE, len(pool)), random_state=0)
    texts = [trial.prompt(r) for r in pool.itertuples()]
    trial.ask(texts[0])
    t0 = time.perf_counter()
    with ThreadPoolExecutor(trial.PARALLEL) as ex:
        list(ex.map(trial.ask, texts))
    wall = time.perf_counter() - t0
    return wall / len(texts), len(pool)


def main():
    t_e, n_prof = llm_throughput()
    print("t_e", t_e, "profiled loans", n_prof, flush=True)
    ch = pd.read_csv("paper/src_extension/selective_topk_order_select_final_choices.csv")
    ch = ch[ch.domain == "lc_llm"]
    rows, costs = [], []
    for seed in sorted(ch.seed.unique()):
        data = crc.DOMAINS["lc_llm"](seed)
        f = data["fit"]
        scale = crc.fit_scale(f)
        n = len(f["pc"])
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            s_f = scale(f)
            ts.append(time.perf_counter() - t0)
        t_scale = float(np.median(ts)) / n
        z_all = (f["pe"] - f["pc"]) / s_f
        U = crc.utility(f["pc"], f["R"], f["C"])
        own = {}
        for o in ORDERS:
            tt = []
            for _ in range(3):
                t0 = time.perf_counter()
                priority(f, U, s_f, z_all, np.random.default_rng(1), o)
                tt.append(time.perf_counter() - t0)
            own[o] = float(np.median(tt)) / n
        costs.append({"seed": seed, "scale": t_scale, **{"own_" + o: v for o, v in own.items()}})
        for _, r in ch[ch.seed == seed].iterrows():
            pred = {c: (t_scale if CAND[c] in ("P2", "P3", "P3c") else 0.0) + own[CAND[c]] + r["fit_calls_" + c] * t_e
                    for c in CAND}
            rows.append({"seed": seed, "k_fraction": r.k_fraction, "eps": r.eps, "auto_time_fitprofiled": min(pred, key=pred.get),
                         "auto_calls": r.chosen})
        print("seed", seed, "done", flush=True)
    t = pd.DataFrame(rows)
    old = pd.read_csv("paper/src_extension/selective_topk_auto_time_lc_llm.csv")[["seed", "k_fraction", "eps", "chosen_time"]]
    t = t.merge(old, on=["seed", "k_fraction", "eps"], how="left")
    t["same_as_before"] = t.auto_time_fitprofiled == t.chosen_time
    t.to_csv(OUT.with_suffix(".csv"), index=False)
    c = pd.DataFrame(costs)
    summary = {"t_e_s_per_call": t_e, "n_profile_loans": n_prof, "cost_medians": c.median(numeric_only=True).to_dict(),
               "same_choice_as_before": int(t.same_as_before.sum()), "n_runs": len(t),
               "seed0_k10_choices": t[(t.seed == 0) & (t.k_fraction == 0.1)][["eps", "auto_time_fitprofiled", "auto_calls"]]
               .to_dict("records")}
    OUT.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print(t.groupby(["k_fraction", "eps"]).auto_time_fitprofiled.agg(lambda s: s.value_counts().to_dict()).to_string())


if __name__ == "__main__":
    main()

"""Measured LC-LLM pipeline per order.

Seed 0, kappa = 10%, eps = 0.05, main protocol (separate fit split, CRC on the calibration queries). For each order
(P0, PU, P2, P3c) we choose lambda_hat on the calibration queries, then run the actual pipeline on the first N_Q test
queries against the local llama.cpp server: priority computation (timed), prompt construction and LLM calls for the
selected loans (timed). Cheap and s_hat inference use the per-loan times measured in selective_topk_lc_llm_cost.json.
The all-LLM path is timed on N_ALL queries. Writes paper/src_extension/selective_topk_llm_orders.csv.
Usage: py src_extension_selective_topk_llm_orders.py [N_Q] [N_ALL] [port]
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
import src_extension_selective_topk_llm_trial as trial
from src_extension_selective_topk_hc_time_breakdown import priorities

N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_ALL = int(sys.argv[2]) if len(sys.argv) > 2 else 2
trial.PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8091
SEED, K_FRAC, EPS = 0, 0.10, 0.05
NAMES = ["P0", "PU", "P2", "P3c"]
OUT = Path("paper/src_extension/selective_topk_llm_orders.csv")


def run_llm(texts):
    t0 = time.perf_counter()
    if texts:
        with ThreadPoolExecutor(trial.PARALLEL) as ex:
            list(ex.map(trial.ask, texts))
    return time.perf_counter() - t0


def main():
    cost = json.loads(Path("paper/src_extension/selective_topk_lc_llm_cost.json").read_text())
    df = crc.lc_llm_frame()
    groups = df["id"].to_numpy(dtype=object)
    m = crc.group_split(groups, SEED)
    qid, m["cal"], m["test"] = crc.batched_cal_test(groups, m["cal"], m["test"], crc.LC_LLM_ROWS, SEED)
    test_ids = df["id"].to_numpy()[m["test"]]
    data = crc.DOMAINS["lc_llm"](SEED)
    f = data["fit"]
    scale = crc.fit_scale(f)
    uf_c, uf_e = crc.utility(f["pc"], f["R"], f["C"]), crc.utility(f["pe"], f["R"], f["C"])
    su_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                 verbosity=-1, random_state=42).fit(np.column_stack([f["xc"], f["pc"], f["R"], f["C"]]),
                                                                    np.abs(uf_e - uf_c))
    su_floor = float(np.quantile(np.abs(uf_e - uf_c), 0.05)) + 1e-9
    z_all = (f["pe"] - f["pc"]) / scale(f)
    pri, U = {}, {}
    for s, stream in (("fit", 0), ("cal", 1), ("test", 2)):
        d = data[s]
        U[s] = crc.utility(d["pc"], d["R"], d["C"])
        su = np.maximum(su_model.predict(np.column_stack([d["xc"], d["pc"], d["R"], d["C"]])), su_floor)
        pri[s] = priorities(d, U[s], scale(d), su, z_all, np.random.default_rng([SEED, stream, 499]), NAMES)
    dc = data["cal"]
    U_ce = crc.utility(dc["pe"], dc["R"], dc["C"])
    dt = data["test"]
    tq = [np.flatnonzero(dt["qid"] == q) for q in np.unique(dt["qid"])][:max(N_Q, N_ALL)]
    need = set(str(i) for ix in tq for i in test_ids[ix])
    parts = [ch[ch["id"].astype(str).isin(need)]
             for ch in pd.read_csv(trial.RAW, usecols=["id"] + trial.COLS, chunksize=200_000, low_memory=False)]
    raw = pd.concat(parts).drop_duplicates("id")
    prompt_of = {str(r.id): trial.prompt(r) for r in raw.itertuples()}
    trial.ask(next(iter(prompt_of.values())))
    rows = []
    t_all = [run_llm([prompt_of[str(i)] for i in test_ids[ix]]) for ix in tq[:N_ALL]]
    rows.append({"order": "ALL", "calls_all_test": 1.0, "llm_s_per_query": float(np.mean(t_all)), "priority_s_per_query": 0.0,
                 "cheap_s_per_query": 0.0, "scale_s_per_query": 0.0, "selected_per_query": 500.0})
    for n in NAMES:
        grid = np.unique(np.concatenate([[-np.inf], np.quantile(pri["fit"][n], np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
        env = []
        for q in np.unique(dc["qid"]):
            ix = np.flatnonzero(dc["qid"] == q)
            k = max(1, int(round(K_FRAC * len(ix))))
            if k < len(ix):
                env.append(crc.envelope(crc.loss_curve({"uc": U["cal"][ix], "ue": U_ce[ix], "k": k,
                                                        "pri": {n: pri["cal"][n][ix]}}, n, grid)[0]))
        env = np.array(env)
        mc = len(env)
        ok = np.flatnonzero(mc / (mc + 1) * env.mean(axis=0) + 1 / (mc + 1) <= EPS)
        lam = grid[ok[0]] if len(ok) else np.inf
        sel_all = pri["test"][n] <= lam
        t_pri, t_llm, n_sel = [], [], []
        for ix in tq[:N_Q]:
            sub = {key: (v[ix] if isinstance(v, np.ndarray) and len(v) == len(dt["pc"]) else v) for key, v in dt.items()}
            t0 = time.perf_counter()
            priorities(sub, U["test"][ix], scale(sub), np.ones(len(ix)), z_all, np.random.default_rng(1), [n])
            t_pri.append(time.perf_counter() - t0)
            sel = ix[sel_all[ix]]
            n_sel.append(len(sel))
            t_llm.append(run_llm([prompt_of[str(i)] for i in test_ids[sel]]))
        rows.append({"order": n, "calls_all_test": float(sel_all.mean()), "llm_s_per_query": float(np.mean(t_llm)),
                     "priority_s_per_query": float(np.mean(t_pri)),
                     "cheap_s_per_query": cost["cheap_seconds_per_loan"] * 500,
                     "scale_s_per_query": 0.0 if n == "P0" else cost["scale_seconds_per_loan"] * 500,
                     "selected_per_query": float(np.mean(n_sel))})
        print(n, rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out["total_s_per_query"] = out[["llm_s_per_query", "priority_s_per_query", "cheap_s_per_query", "scale_s_per_query"]].sum(axis=1)
    out["m_cal"] = len(np.unique(dc["qid"]))
    t_all_q = out.loc[out.order == "ALL", "total_s_per_query"].iloc[0]
    for h in (100, 1000):
        out[f"cum_speedup_H{h}"] = h * t_all_q / (out.m_cal * t_all_q + h * out.total_s_per_query)
    out.to_csv(OUT, index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

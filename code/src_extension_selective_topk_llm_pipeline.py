"""Measured end-to-end pipeline for the LC-LLM workload.

Seed 0, kappa = 10%, eps = 0.05, priority P2 with CRC. For the first N_Q test queries (500 loans each) we run the
actual pipeline against the local llama.cpp server (Qwen2.5-7B-Instruct Q4_K_M, 8 parallel slots, as in the scoring
run): cheap model + s_hat (already computed), selection, prompt construction for the selected loans, and the LLM calls;
we time each query end to end, sequentially across queries. For comparison we time the all-LLM path on N_ALL queries and
single-request latency. Prompt lengths of selected vs. all test loans come from the scoring run.
Writes paper/src_extension/selective_topk_llm_pipeline.json.
Usage: py src_extension_selective_topk_llm_pipeline.py [N_Q] [N_ALL] [port]
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

N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_ALL = int(sys.argv[2]) if len(sys.argv) > 2 else 2
trial.PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8091
SEED, K_FRAC, EPS = 0, 0.10, 0.05
OUT = Path("paper/src_extension/selective_topk_llm_pipeline.json")


def run_llm(texts):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(trial.PARALLEL) as ex:
        list(ex.map(trial.ask, texts))
    return time.perf_counter() - t0


def main():
    df = crc.lc_llm_frame()
    groups = df["id"].to_numpy(dtype=object)
    m = crc.group_split(groups, SEED)
    qid, m["cal"], m["test"] = crc.batched_cal_test(groups, m["cal"], m["test"], crc.LC_LLM_ROWS, SEED)
    test_ids = df["id"].to_numpy()[m["test"]]
    data = crc.DOMAINS["lc_llm"](SEED)
    assert len(test_ids) == len(data["test"]["pc"])
    scale = crc.fit_scale(data["fit"])
    fit = crc.query_arrays(data["fit"], scale, K_FRAC, SEED, stream=0)
    cal = crc.query_arrays(data["cal"], scale, K_FRAC, SEED, stream=1)
    test = crc.query_arrays(data["test"], scale, K_FRAC, SEED, stream=2)
    pooled = np.concatenate([q["pri"]["P2"] for q in fit])
    grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
    env = np.array([crc.envelope(crc.loss_curve(q, "P2", grid)[0]) for q in cal])
    mc = len(cal)
    j = np.flatnonzero(mc / (mc + 1) * env.mean(axis=0) + 1 / (mc + 1) <= EPS)[0]
    lam = grid[j]
    q_rows = [np.flatnonzero(data["test"]["qid"] == q) for q in np.unique(data["test"]["qid"])]
    q_rows = [ix for ix in q_rows if max(1, int(round(K_FRAC * len(ix)))) < len(ix)]
    sel = [ix[q["pri"]["P2"] <= lam] for q, ix in zip(test, q_rows)]
    # prompt lengths from the scoring run
    sc = pd.read_csv(crc.LLM_SCORES).drop_duplicates("id")
    tok = dict(zip(sc["id"].astype(str), sc["prompt_tokens"]))
    all_tok = np.array([tok[str(i)] for i in test_ids], dtype=float)
    sel_rows = np.concatenate(sel)
    # raw rows for the prompts of the timed queries
    need = set(str(i) for ix in q_rows[:max(N_Q, N_ALL)] for i in test_ids[ix])
    parts = []
    for ch in pd.read_csv(trial.RAW, usecols=["id"] + trial.COLS, chunksize=200_000, low_memory=False):
        parts.append(ch[ch["id"].astype(str).isin(need)])
    raw = pd.concat(parts).drop_duplicates("id")
    prompt_of = {str(r.id): trial.prompt(r) for r in raw.itertuples()}
    trial.ask(next(iter(prompt_of.values())))                    # warm-up
    sel_times, sel_counts = [], []
    for ix, s in zip(q_rows[:N_Q], sel[:N_Q]):
        t0 = time.perf_counter()
        texts = [prompt_of[str(i)] for i in test_ids[s]]        # selection already made from cheap scores
        if texts:
            run_llm(texts)
        sel_times.append(time.perf_counter() - t0)
        sel_counts.append(len(texts))
    all_times = [run_llm([prompt_of[str(i)] for i in test_ids[ix]]) for ix in q_rows[:N_ALL]]
    single = []
    for i in test_ids[sel_rows[:20]]:
        t0 = time.perf_counter()
        trial.ask(prompt_of[str(i)]) if str(i) in prompt_of else None
        single.append(time.perf_counter() - t0)
    t_sel, t_all = float(np.mean(sel_times)), float(np.mean(all_times))
    H = np.array([100, 500, 1000, 5000])
    out = {"seed": SEED, "kappa": K_FRAC, "eps": EPS, "m_cal": mc, "n_timed_queries": N_Q,
           "call_fraction_all_test": float(np.mean([len(s) / len(ix) for s, ix in zip(sel, q_rows)])),
           "selected_per_timed_query": sel_counts, "seconds_per_query_selected": sel_times,
           "seconds_per_query_all_llm": all_times, "mean_selected_query_s": t_sel, "mean_all_llm_query_s": t_all,
           "measured_speedup": t_all / t_sel, "single_request_latency_s_median": float(np.median(single)),
           "prompt_tokens_selected_mean": float(all_tok[sel_rows].mean()),
           "prompts_found": len(prompt_of),
           "prompt_tokens_all_mean": float(all_tok.mean()),
           "cumulative_speedup_incl_calibration": {int(h): float(h * t_all / (mc * t_all + h * t_sel)) for h in H}}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

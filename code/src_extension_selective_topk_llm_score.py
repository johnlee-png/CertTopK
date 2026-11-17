"""S1 full run: score every finished Lending Club loan with a description (2007-2014)
with the local LLM (same prompt and Yes/No read-out as the trial script), so the LLM
can serve as the expensive model of an 8th domain. Resumable: rows already in the
output file are skipped, and results are appended every CHUNK loans.
Needs the llama.cpp server on 127.0.0.1:PORT (see the trial script).
Usage: py src_extension_selective_topk_llm_score.py [port]
Output: paper/src_extension/selective_topk_llm_scores.csv (id, p_llm_yes, prompt_tokens)
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import src_extension_selective_topk_llm_trial as trial

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8091
trial.PORT = PORT
CHUNK = 1000
OUT = Path("paper/src_extension/selective_topk_llm_scores.csv")


def load_all():
    parts = []
    for ch in pd.read_csv(trial.RAW, usecols=["id"] + trial.COLS, chunksize=200_000, low_memory=False):
        st = ch.loan_status.astype(str).str.replace("Does not meet the credit policy. Status:", "")
        ok = st.isin(["Fully Paid", "Charged Off"]) & ch.desc.notna() & (ch.desc.astype(str).str.strip().str.len() > 30)
        parts.append(ch[ok])
    return pd.concat(parts, ignore_index=True)


def main():
    df = load_all()
    done = set(pd.read_csv(OUT).id.astype(str)) if OUT.exists() else set()
    todo = df[~df.id.astype(str).isin(done)].reset_index(drop=True)
    print(f"{len(df)} loans, {len(done)} already scored, {len(todo)} to go", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(trial.PARALLEL) as ex:
        for s in range(0, len(todo), CHUNK):
            part = todo.iloc[s:s + CHUNK]
            res = list(ex.map(trial.ask, [trial.prompt(r) for r in part.itertuples()]))
            pd.DataFrame({"id": part.id.to_numpy(), "p_llm_yes": [r[0] for r in res],
                          "prompt_tokens": [r[1] for r in res]}).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            n = s + len(part)
            el = time.perf_counter() - t0
            print(f"{len(done) + n}/{len(df)}  {el / n:.3f}s/loan  eta {el / n * (len(todo) - n) / 60:.0f} min", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

"""Deterministic LLM scores of every LC-LLM loan.

The logged scores (selective_topk_llm_scores.csv) were produced with eight concurrent requests, under which the server
is not deterministic (src_vldb_llm_determinism.py: repeat MAE 0.40 in logit at concurrency 8, 0 at concurrency 1). For
the guarantee experiments the oracle, the reference answer, the fallback and the calibration must use one loss
definition, so every loan is re-scored here one request at a time; the same request at concurrency 1 then reproduces
the stored score exactly. Resumable; the original score file is not modified.
Writes paper/src_extension/selective_topk_llm_scores_c1.csv (id, p_llm_yes, prompt_tokens).
Usage: py src_vldb_llm_rescore_c1.py   (from the artifact root, llama.cpp server on port 8091)
"""
import time
from pathlib import Path

import pandas as pd

import src_extension_selective_topk_crc as crc
import src_extension_selective_topk_llm_trial as trial

trial.PORT = 8091
TEXT = Path("data/processed/lending_club/lc_llm_text_clean.parquet")
OUT = Path("paper/src_extension/selective_topk_llm_scores_c1.csv")
FLUSH = 500


def main():
    ids = crc.lc_llm_frame()["id"].astype(str)
    text = pd.read_parquet(TEXT).set_index("id").loc[ids].reset_index()
    done = set(pd.read_csv(OUT, dtype={"id": str})["id"]) if OUT.exists() else set()
    todo = text[~text["id"].isin(done)]
    print(f"{len(done)} done, {len(todo)} to score", flush=True)
    rows, t0 = [], time.time()
    for n, r in enumerate(todo.itertuples(), 1):
        p, tok = trial.ask(trial.prompt(r))
        rows.append({"id": r.id, "p_llm_yes": p, "prompt_tokens": tok})
        if len(rows) == FLUSH or n == len(todo):
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            rows = []
            rate = n / (time.time() - t0)
            print(f"{len(done) + n}/{len(text)}  {rate:.1f} loans/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()

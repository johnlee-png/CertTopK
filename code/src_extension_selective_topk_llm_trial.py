"""S1 trial: is a local LLM a usable "expensive model" for Lending Club loans
with a free-text description, and what does a call cost?

Takes N finished loans (2007-2014, description longer than 30 characters),
asks the LLM (llama.cpp server, Qwen2.5-7B-Instruct Q4_K_M on the local GPU)
whether the borrower will fully repay, and reads the probability of "Yes"
versus "No" for the first answer token. Reports throughput (s per loan with
PARALLEL concurrent requests), the cheap path time per loan for comparison,
and the AUC of the LLM score alone and combined with listing information.
Writes paper/src_extension/selective_topk_llm_trial.{csv,json}.
Usage: py src_extension_selective_topk_llm_trial.py [N] [port]
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8091
PARALLEL = 8
RAW = Path("data/raw/lending_club/accepted_2007_to_2018Q4.csv.gz")
COLS = ["desc", "title", "loan_status", "issue_d", "loan_amnt", "term", "int_rate", "grade", "purpose",
        "emp_length", "annual_inc", "dti", "home_ownership"]


def load(n, seed=0):
    parts = []
    for ch in pd.read_csv(RAW, usecols=COLS, chunksize=200_000, low_memory=False):
        st = ch.loan_status.astype(str).str.replace("Does not meet the credit policy. Status:", "")
        ok = st.isin(["Fully Paid", "Charged Off"]) & ch.desc.notna() & (ch.desc.astype(str).str.strip().str.len() > 30)
        c = ch[ok].copy()
        c["repaid"] = (st[ok] == "Fully Paid").astype(int)
        parts.append(c)
    df = pd.concat(parts, ignore_index=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def prompt(r):
    desc = str(r.desc).replace("Borrower added on", "").strip()[:1200]
    return ("<|im_start|>system\nYou are a careful credit analyst.<|im_end|>\n<|im_start|>user\n"
            f"Loan request: ${r.loan_amnt:,.0f}, {r.term.strip()}, interest {r.int_rate}%, grade {r.grade}, "
            f"purpose {r.purpose}, title \"{r.title}\", employment {r.emp_length}, annual income ${r.annual_inc:,.0f}, "
            f"DTI {r.dti}, home {r.home_ownership}.\nBorrower's description: \"{desc}\"\n"
            "Will this borrower fully repay the loan? Answer with one word: Yes or No.<|im_end|>\n"
            "<|im_start|>assistant\n")


def ask(text):
    body = json.dumps({"prompt": text, "n_predict": 1, "n_probs": 20, "temperature": 0.0, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body, headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())
    first = out["completion_probabilities"][0]
    # newer llama.cpp servers return top_logprobs/token/logprob, older ones top_probs/tok_str/prob;
    # sum over surface variants (" Yes", "Yes", " yes")
    if "top_logprobs" in first:
        cand = [(p["token"], float(np.exp(p["logprob"]))) for p in first["top_logprobs"]]
    else:
        cand = [(p["tok_str"], p["prob"]) for p in first["top_probs"]]
    y = sum(p for t, p in cand if t.strip().lower() == "yes")
    n = sum(p for t, p in cand if t.strip().lower() == "no")
    return y / (y + n) if y + n > 0 else np.nan, out.get("timings", {}).get("prompt_n", np.nan)


def main():
    df = load(N)
    texts = [prompt(r) for r in df.itertuples()]
    ask(texts[0])                                   # warm-up
    t0 = time.perf_counter()
    with ThreadPoolExecutor(PARALLEL) as ex:
        res = list(ex.map(ask, texts))
    wall = time.perf_counter() - t0
    df["p_llm_yes"] = [r[0] for r in res]
    df["prompt_tokens"] = [r[1] for r in res]
    # cheap path reference: a listing-only logistic model (cost of a tabular prediction)
    X = pd.get_dummies(df[["grade", "purpose"]]).assign(int_rate=df.int_rate, dti=df.dti.fillna(df.dti.median()),
                                                        inc=np.log1p(df.annual_inc.fillna(0)))
    half = np.arange(len(df)) % 2 == 0
    lr = LogisticRegression(max_iter=2000).fit(X[half], df.repaid[half])
    t1 = time.perf_counter()
    for _ in range(100):
        lr.predict_proba(X)
    cheap_per = (time.perf_counter() - t1) / (100 * len(df))
    p_cheap = lr.predict_proba(X[~half])[:, 1]
    Z = np.column_stack([np.log(p_cheap / (1 - p_cheap)), np.log((df.p_llm_yes[~half].clip(1e-4, 1 - 1e-4)) / (1 - df.p_llm_yes[~half].clip(1e-4, 1 - 1e-4)))])
    yb = df.repaid[~half].to_numpy()
    comb = LogisticRegression().fit(Z[: len(Z) // 2], yb[: len(Z) // 2]).predict_proba(Z[len(Z) // 2:])[:, 1]
    summary = {"n": N, "parallel": PARALLEL, "wall_seconds": wall, "seconds_per_loan": wall / N,
               "mean_prompt_tokens": float(np.nanmean(df.prompt_tokens)), "cheap_seconds_per_loan": cheap_per,
               "cost_ratio": (wall / N) / cheap_per, "repaid_rate": float(df.repaid.mean()),
               "auc_llm_alone": float(roc_auc_score(df.repaid, df.p_llm_yes.fillna(0.5))),
               "auc_cheap_holdout": float(roc_auc_score(yb, p_cheap)),
               "auc_cheap_plus_llm_holdout": float(roc_auc_score(yb[len(Z) // 2:], comb)),
               "auc_cheap_same_rows": float(roc_auc_score(yb[len(Z) // 2:], p_cheap[len(Z) // 2:])),
               "corr_llm_cheap": float(np.corrcoef(df.p_llm_yes[~half].fillna(0.5), p_cheap)[0, 1]),
               "nan_answers": int(df.p_llm_yes.isna().sum())}
    out = Path("paper/src_extension")
    df.drop(columns=["desc"]).to_csv(out / "selective_topk_llm_trial.csv", index=False)
    (out / "selective_topk_llm_trial.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

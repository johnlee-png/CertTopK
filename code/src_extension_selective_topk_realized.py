"""M4: realized utility with the actual outcomes.

The reference answer T* is the expensive model's top-k. Is reproducing it
worth anything in terms of what actually happened? For every test query we
compute the realized utility sum_{i in T} (y_i R_i - (1 - y_i) C_i) of
  cheap      the cheap model's top-k
  proposed   the certified top-k (priority P2, CRC lambda_hat, eps in EPSILONS)
  expensive  the expensive model's top-k (T*)
  oracle     the top-k by realized utility (uses the labels; an upper bound)
and report the share of the cheap-to-expensive gap that the proposed answer
recovers. R and C are the planning values each domain uses (for KDD98 the
predicted gift), y is the outcome the probability P refers to (pass success,
30-day readmission, repayment, response, churn, fraud, repayment).
The builders in src_extension_selective_topk_crc.py are not changed: labels
are attached here with the same split and query rules (group_split,
batched_cal_test), and the alignment is checked.
Writes paper/src_extension/selective_topk_realized_{results,summary}.csv.
Usage: py src_extension_selective_topk_realized.py <n_seeds> [domains] [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
import diabetes_new_regret_diagnostic as dia
from src_extension_selective_topk_diagnostic import utility

K_FRAC = 0.10
OUT = Path("paper/src_extension")
_LABELS = {}


def _tabular(groups, y, rows_per_query, seed):
    m = crc.group_split(groups, seed)
    _, m["cal"], m["test"] = crc.batched_cal_test(groups, m["cal"], m["test"], rows_per_query, seed)
    return {s: np.asarray(y)[m[s]] for s in ["fit", "cal", "test"]}


def labels(domain, seed):
    """Outcome y for the fit/cal/test arrays of crc.DOMAINS[domain](seed), in the same row order."""
    if domain == "soccer":
        if "soccer" not in _LABELS:
            d = Path("data/processed")
            _LABELS["soccer"] = (pd.read_parquet(d / "passes_with_features.parquet", columns=["match_id"])["match_id"].to_numpy(),
                                 pd.read_parquet(d / "y_labels.parquet")["pass_success"].to_numpy())
        groups, y = _LABELS["soccer"]
        m = crc.group_split(groups, seed)
        return {s: y[m[s]] for s in ["fit", "cal", "test"]}
    if domain == "diabetes":
        df = pd.read_csv(dia.DATA_DIR / "diabetic_data.csv", usecols=["patient_nbr", "readmitted"])
        return _tabular(df["patient_nbr"].to_numpy(), (df["readmitted"] == "<30").astype(int).to_numpy(), 500, seed)
    if domain == "home_credit":
        app = pd.read_parquet(crc.CLEAN / "application_train_clean.parquet", columns=["SK_ID_CURR", "TARGET"])
        return _tabular(app["SK_ID_CURR"].to_numpy(), 1 - app["TARGET"].to_numpy(), 1000, seed)   # P = repayment
    if domain == "kdd98":
        df = pd.read_csv(crc.KDD_RAW, usecols=["CONTROLN", "TARGET_B"])
        return _tabular(df["CONTROLN"].to_numpy(), df["TARGET_B"].to_numpy(), 250, seed)
    if domain == "telco":
        df = pd.read_csv(crc.TELCO_RAW, usecols=["customerID", "Churn"])
        return _tabular(df["customerID"].to_numpy(), (df["Churn"] == "Yes").astype(int).to_numpy(), 50, seed)
    if domain == "ieee_fraud":
        df = pd.read_parquet(crc.IEEE_CLEAN, columns=["isFraud", "TransactionDT", "D1", "card1", "addr1"])
        day = (np.floor(df["TransactionDT"] / 86400) - df["D1"]).fillna(-1)
        groups = (df["card1"].astype(str) + "_" + df["addr1"].fillna(-1).astype(str) + "_" + day.astype(str)).to_numpy(dtype=object)
        return _tabular(groups, df["isFraud"].to_numpy(), 1000, seed)
    if domain == "lending_club":
        df = pd.read_parquet(crc.LC_CLEAN, columns=["id", "repaid"])
        return _tabular(df["id"].astype(str).to_numpy(dtype=object), df["repaid"].to_numpy(), 5000, seed)
    if domain == "lc_llm":
        df = crc.lc_llm_frame()
        return _tabular(df["id"].astype(str).to_numpy(dtype=object), df["repaid"].to_numpy(), crc.LC_LLM_ROWS, seed)
    raise KeyError(domain)


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    y = labels(domain, seed)
    for s in ["fit", "cal", "test"]:
        assert len(y[s]) == len(data[s]["pc"]), (domain, s, len(y[s]), len(data[s]["pc"]))
    scale = crc.fit_scale(data["fit"])
    d, yt = data["test"], y["test"]
    U_c, U_e = utility(d["pc"], d["R"], d["C"]), utility(d["pe"], d["R"], d["C"])
    u_real = yt * d["R"] - (1 - yt) * d["C"]
    fit = crc.query_arrays(data["fit"], scale, K_FRAC, seed, stream=0)
    cal = crc.query_arrays(data["cal"], scale, K_FRAC, seed, stream=1)
    pooled = np.concatenate([q["pri"]["P2"] for q in fit])
    grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
    Lt_cal = np.array([crc.envelope(crc.loss_curve(q, "P2", grid)[0]) for q in cal])
    m = len(cal)
    lam = {eps: grid[np.flatnonzero(m / (m + 1) * Lt_cal.mean(axis=0) + 1 / (m + 1) <= eps)[0]] for eps in crc.EPSILONS}
    s_test = scale(d)
    rows = []
    for q in np.unique(d["qid"]):
        ix = np.flatnonzero(d["qid"] == q)
        k = max(1, int(round(K_FRAC * len(ix))))
        if k >= len(ix):
            continue
        uc, ue, ur = U_c[ix], U_e[ix], u_real[ix]
        part = -np.partition(-uc, [k - 1, k])
        p2 = np.abs(uc - 0.5 * (part[k - 1] + part[k])) / ((d["R"][ix] + d["C"][ix]) * s_test[ix])
        row = {"domain": domain, "seed": seed, "qid": q, "n": len(ix), "k": k,
               "cheap": ur[crc.topk(uc, k)].sum(), "expensive": ur[crc.topk(ue, k)].sum(),
               "oracle": ur[crc.topk(ur, k)].sum()}
        for eps, l in lam.items():
            mixed = np.where(p2 <= l, ue, uc)
            row[f"proposed_eps{eps}"] = ur[crc.topk(mixed, k)].sum()
        rows.append(row)
    return rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else list(crc.DOMAINS)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    rows = []
    with Pool(workers) as pool:
        for r, msg in pool.imap_unordered(task, [(dm, s) for dm in domains for s in range(n_seeds)]):
            rows += r
            print(msg, flush=True)
    t = pd.concat([crc.kept_rows(OUT / "selective_topk_realized_results.csv", domains), pd.DataFrame(rows)],
                  ignore_index=True)
    t.to_csv(OUT / "selective_topk_realized_results.csv", index=False)
    cols = ["cheap", "expensive", "oracle"] + [f"proposed_eps{e}" for e in crc.EPSILONS]
    per_seed = t.groupby(["domain", "seed"])[cols].mean()   # mean realized utility per query
    summ = per_seed.groupby("domain").mean()
    gap = summ.expensive - summ.cheap
    for e in crc.EPSILONS:
        summ[f"recovered_eps{e}"] = (summ[f"proposed_eps{e}"] - summ.cheap) / gap
    summ["expensive_vs_oracle"] = (summ.expensive - summ.cheap) / (summ.oracle - summ.cheap)
    # seed-level check that expensive > cheap
    diff = (per_seed.expensive - per_seed.cheap).groupby("domain")
    summ["seeds_expensive_better"] = diff.apply(lambda x: int((x > 0).sum()))
    summ["seeds"] = diff.size()
    summ.reset_index().to_csv(OUT / "selective_topk_realized_summary.csv", index=False)
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()

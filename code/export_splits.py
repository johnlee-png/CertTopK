"""Export the train/fit/calibration/test split and the query of every row, per dataset and seed.

The assignment is deterministic given the group ids (crc.group_split, crc.batch_queries, crc.batched_cal_test), so it
can be regenerated from the raw data. We write
  splits/summary.csv          per (dataset, seed): groups and rows per split, number of queries, query sizes, and the
                              SHA-256 of the full assignment (so a regenerated split can be checked byte for byte);
  splits/<dataset>_seed0.csv.gz  the full assignment for seed 0: row position, group id, split, query id.
Usage: py -X utf8 export_splits.py [n_seeds]
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc

OUT = Path("paper/src_extension/splits")
SPLIT_CODE = {"train": 0, "fit": 1, "cal": 2, "test": 3}


def groups_of(domain):
    """(group id per row in the order used by the experiments, rows per query or None for one query per group)."""
    if domain == "soccer":
        return pd.read_parquet("data/processed/passes_with_features.parquet")["match_id"].to_numpy(), None
    if domain == "diabetes":
        return pd.read_csv(crc.dia.DATA_DIR / "diabetic_data.csv")["patient_nbr"].to_numpy(), 500
    if domain == "home_credit":
        return pd.read_parquet(crc.CLEAN / "application_train_clean.parquet", columns=["SK_ID_CURR"])["SK_ID_CURR"].to_numpy(), 1000
    if domain == "kdd98":
        return pd.read_csv(crc.KDD_RAW, usecols=["CONTROLN"])["CONTROLN"].to_numpy(), 250
    if domain == "telco":
        return pd.read_csv(crc.TELCO_RAW)["customerID"].to_numpy(), 50
    if domain == "ieee_fraud":
        df = pd.read_parquet(crc.IEEE_CLEAN, columns=["card1", "addr1", "TransactionDT", "D1"])
        day = (np.floor(df["TransactionDT"] / 86400) - df["D1"]).fillna(-1)
        g = (df["card1"].astype(str) + "_" + df["addr1"].fillna(-1).astype(str) + "_" + day.astype(str)).to_numpy(dtype=object)
        return g, 1000
    if domain == "lending_club":
        return pd.read_parquet(crc.LC_CLEAN, columns=["id"])["id"].astype(str).to_numpy(dtype=object), 5000
    if domain == "lc_llm":
        return crc.lc_llm_frame()["id"].to_numpy(dtype=object), crc.LC_LLM_ROWS
    raise ValueError(domain)


def assignment(groups, rows, seed):
    m = crc.group_split(groups, seed)
    split = np.full(len(groups), -1, dtype=np.int8)
    if rows is None:                                           # soccer: one query per match
        for s, mask in m.items():
            split[mask] = SPLIT_CODE[s]
        _, qid = np.unique(groups, return_inverse=True)
        qid = qid.astype(np.int64)
        qid[split == 0] = -1
        return split, qid
    qid, cal, test = crc.batched_cal_test(groups, m["cal"], m["test"], rows, seed)
    qid = qid.astype(np.int64)
    split[m["train"]], split[m["fit"]], split[cal], split[test] = 0, 1, 2, 3
    qid[m["fit"]] = crc.batch_queries(groups[m["fit"]], rows, seed) + 10_000_000   # fit queries, separate id range
    qid[m["train"]] = -1
    return split, qid


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    OUT.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for domain in ["soccer", "diabetes", "home_credit", "kdd98", "telco", "ieee_fraud", "lending_club", "lc_llm"]:
        groups, rows = groups_of(domain)
        for seed in range(n_seeds):
            split, qid = assignment(groups, rows, seed)
            h = hashlib.sha256(split.tobytes() + qid.tobytes()).hexdigest()
            r = {"dataset": domain, "seed": seed, "rows": len(groups), "sha256": h}
            for s, c in SPLIT_CODE.items():
                sel = split == c
                r[f"{s}_rows"] = int(sel.sum())
                r[f"{s}_groups"] = int(pd.unique(groups[sel]).size)
                if s in ("cal", "test"):
                    sizes = pd.Series(qid[sel]).value_counts()
                    r[f"{s}_queries"], r[f"{s}_query_size_min"], r[f"{s}_query_size_max"] = len(sizes), int(sizes.min()), int(sizes.max())
            rows_out.append(r)
            if seed == 0:
                df = pd.DataFrame({"row": np.arange(len(groups)), "group_id": groups,
                                   "split": pd.Categorical.from_codes(split, list(SPLIT_CODE)), "query_id": qid})
                with gzip.open(OUT / f"{domain}_seed0.csv.gz", "wt", encoding="utf-8") as fh:
                    df.to_csv(fh, index=False)
        print(domain, "done", flush=True)
    pd.DataFrame(rows_out).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame(rows_out).groupby("dataset")[["cal_queries", "test_queries"]].agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()

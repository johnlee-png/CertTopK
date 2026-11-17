"""Access-path timing for the selections made by the current method (CRC + P2).

The first index experiment (src_extension_selective_topk_homecredit_index.py)
timed the refined sets of an earlier, per-candidate interval certification.
Here the refined set is the one the certified top-k operator actually selects:
Home Credit, seed 0, test split, 1,000-applicant queries, priority P2,
lambda_hat from CRC on the calibration queries (exactly as in
src_extension_selective_topk_crc.run). For each (k, eps) we time fetching the
expensive features (aggregates over the six child tables) for the selected
applicants only, and for all test applicants, with
  duckdb_parquet  the semi-join over the parquet files
  duckdb_memory   the same SQL over in-memory DuckDB tables
  offset_index    key-sorted child tables + per-key offsets (OffsetIndex)
Only the feature fetch is timed (the access path); model inference is the same
for every path. Writes paper/src_extension/selective_topk_homecredit_index_crc.{json,csv}.
"""
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_homecredit import CHILD, CLEAN, expensive_features, prepare
from src_extension_selective_topk_homecredit_index import OffsetIndex

SEED = 0
REPEATS = 3


def median_time(fn):
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    data = crc.build_home_credit(SEED)
    app = crc._HC[0]
    groups = app["SK_ID_CURR"].to_numpy()
    m = crc.group_split(groups, SEED)
    _, m["cal"], m["test"] = crc.batched_cal_test(groups, m["cal"], m["test"], 1000, SEED)
    test_ids = groups[m["test"]]
    assert len(test_ids) == len(data["test"]["pc"])
    scale = crc.fit_scale(data["fit"])

    con = duckdb.connect()
    prepare(con)
    mem = duckdb.connect()
    for t in CHILD + ["bureau_balance"]:
        mem.execute(f"create table {t} as select * from '{CLEAN / (t + '_clean.parquet')}'")
    t0 = time.perf_counter()
    idx = OffsetIndex(con)
    t_build = time.perf_counter() - t0

    def timings(ids):
        return {"duckdb_parquet": median_time(lambda: expensive_features(con, ids)),
                "duckdb_memory": median_time(lambda: expensive_features(mem, ids)),
                "offset_index": median_time(lambda: idx.features(ids))}
    full = timings(test_ids)
    rows = []
    for k_frac in crc.K_FRACTIONS:
        fitq = crc.query_arrays(data["fit"], scale, k_frac, SEED)
        calq = crc.query_arrays(data["cal"], scale, k_frac, SEED)
        pooled = np.concatenate([q["pri"]["P2"] for q in fitq])
        grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
        Lt = np.array([crc.envelope(crc.loss_curve(q, "P2", grid)[0]) for q in calq])
        mc = len(calq)
        # P2 for every test row, in row order (query_arrays groups rows by query)
        U_c = crc.utility(data["test"]["pc"], data["test"]["R"], data["test"]["C"])
        s = scale(data["test"])
        p2 = np.empty(len(U_c))
        qid = data["test"]["qid"]
        for q in np.unique(qid):
            ix = np.flatnonzero(qid == q)
            k = max(1, int(round(k_frac * len(ix))))
            part = -np.partition(-U_c[ix], [k - 1, k])
            budget = data["test"]["R"][ix] + data["test"]["C"][ix]
            p2[ix] = np.abs(U_c[ix] - 0.5 * (part[k - 1] + part[k])) / (budget * s[ix])
        for eps in crc.EPSILONS:
            lam = grid[np.flatnonzero(mc / (mc + 1) * Lt.mean(axis=0) + 1 / (mc + 1) <= eps)[0]]
            sel = p2 <= lam
            t = timings(test_ids[sel])
            rows.append({"k_fraction": k_frac, "eps": eps, "call_fraction": float(sel.mean()),
                         **{f"sec_{p}": v for p, v in t.items()},
                         **{f"time_fraction_{p}": t[p] / full[p] for p in t}})
            print(rows[-1], flush=True)
    out = Path("paper/src_extension")
    json.dump({"seed": SEED, "n_test": int(len(test_ids)), "index_build_seconds": t_build,
               "full_test_seconds": full, "configs": rows},
              open(out / "selective_topk_homecredit_index_crc.json", "w"), indent=2)
    pd.DataFrame(rows).to_csv(out / "selective_topk_homecredit_index_crc.csv", index=False)


if __name__ == "__main__":
    main()

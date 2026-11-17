"""Does the cost of selective inference follow the number of refined tuples?

src_extension_selective_topk_homecredit.py found that refining 3.9% of the
applicants still took 38% of the full aggregation time, because the DuckDB
semi-join scans every child table regardless of how many ids are selected.

This script keeps the child tables sorted by SK_ID_CURR (they already are, in
the *_clean.parquet copies) and adds an offset index: for every applicant, the
start row and row count in each child table. Aggregating a selected set then
reads only that set's rows. Compared, for the refined set of every (alpha, k)
configuration and for the full test set:
  duckdb_parquet  the semi-join over the parquet views (as in the first run)
  duckdb_memory   the same SQL over in-memory DuckDB tables
  offset_index    gather the selected rows by offset, aggregate with reduceat
The offset-index features are checked against DuckDB's before timing.
Same split, models and bounds as the first run (build() is reused).
"""
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src_extension_selective_topk_homecredit import (
    ALPHAS, CHILD, CLEAN, build, expensive_features, full_X, numeric_cols,
)
from src_extension_selective_topk_diagnostic import K_FRACTIONS, certify, conformal_halfwidth, utility

REPEATS = 3


class OffsetIndex:
    """Key-sorted columnar copies of the child tables plus per-key offsets."""

    def __init__(self, con):
        self.tables = {}
        for t in CHILD:
            cols = numeric_cols(con, t)
            sel = ", ".join(f'"{c}"' for c in cols)
            df = con.execute(f"select SK_ID_CURR, {sel} from {t} order by SK_ID_CURR").df()
            aggs = [(c, df[c].to_numpy(dtype=float)) for c in cols]
            self.tables[t] = self._index(df["SK_ID_CURR"].to_numpy(), aggs, cols)
        bb = con.execute("""select SK_ID_CURR,
                                   case when STATUS in ('1','2','3','4','5') then 1.0 else 0.0 end as late
                            from bureau_balance order by SK_ID_CURR""").df()
        self.tables["bureau_balance"] = self._index(bb["SK_ID_CURR"].to_numpy(),
                                                    [("late", bb["late"].to_numpy(dtype=float))], [])
        inst = con.execute("""select SK_ID_CURR,
                                     DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT as days_late,
                                     AMT_PAYMENT / nullif(AMT_INSTALMENT, 0) as paid_ratio
                              from installments_payments order by SK_ID_CURR""").df()
        self.tables["inst_derived"] = self._index(
            inst["SK_ID_CURR"].to_numpy(),
            [("days_late", inst["days_late"].to_numpy(dtype=float)),
             ("paid_ratio", inst["paid_ratio"].to_numpy(dtype=float))], [])

    @staticmethod
    def _index(ids, arrays, cols):
        uid, start, cnt = np.unique(ids, return_index=True, return_counts=True)
        return {"uid": uid, "start": start, "cnt": cnt, "cols": cols,
                "M": np.column_stack([a for _, a in arrays])}

    @staticmethod
    def _segments(tab, req):
        pos = np.clip(np.searchsorted(tab["uid"], req), 0, len(tab["uid"]) - 1)
        found = tab["uid"][pos] == req
        starts, counts = tab["start"][pos[found]], tab["cnt"][pos[found]]
        seg = np.cumsum(counts) - counts
        gather = np.arange(counts.sum()) - np.repeat(seg, counts) + np.repeat(starts, counts)
        return found, counts, seg, tab["M"][gather]

    @staticmethod
    def _reduce(rows, seg):
        valid = ~np.isnan(rows)
        s = np.add.reduceat(np.where(valid, rows, 0.0), seg, axis=0)
        n = np.add.reduceat(valid, seg, axis=0)
        mx = np.maximum.reduceat(np.where(valid, rows, -np.inf), seg, axis=0)
        mn = np.minimum.reduceat(np.where(valid, rows, np.inf), seg, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, s / n, np.nan)
        return mean, np.where(n > 0, mx, np.nan), np.where(n > 0, mn, np.nan)

    def features(self, ids):
        req = np.asarray(ids, dtype=np.int64)
        out = {}

        def put(name, found, values):
            col = np.full(len(req), np.nan)
            col[found] = values
            out[name] = col

        rows_read = 0
        for t in CHILD:
            tab = self.tables[t]
            found, counts, seg, rows = self._segments(tab, req)
            rows_read += len(rows)
            put(f"{t}_cnt", found, counts.astype(float))
            if found.any():
                mean, mx, mn = self._reduce(rows, seg)
                for j, c in enumerate(tab["cols"]):
                    put(f"{t}_{c}_mean", found, mean[:, j])
                    put(f"{t}_{c}_max", found, mx[:, j])
                    put(f"{t}_{c}_min", found, mn[:, j])
        for key, names in [("bureau_balance", None), ("inst_derived", None)]:
            tab = self.tables[key]
            found, counts, seg, rows = self._segments(tab, req)
            rows_read += len(rows) if key == "bureau_balance" else 0
            mean = self._reduce(rows, seg)[0] if found.any() else np.empty((0, rows.shape[1]))
            if key == "bureau_balance":
                put("bb_months", found, counts.astype(float))
                put("bb_late_share", found, mean[:, 0])
            else:
                put("inst_days_late_mean", found, mean[:, 0])
                put("inst_paid_ratio_mean", found, mean[:, 1])
        return pd.DataFrame(out, index=pd.Index(req, name="SK_ID_CURR")), rows_read


def median_time(fn):
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    con = duckdb.connect()
    d = build(con)
    te, exp = d["te"], d["exp"]
    te_ids = te["SK_ID_CURR"].to_numpy()

    print("[index] Loading in-memory DuckDB tables...")
    mem = duckdb.connect()
    for t in CHILD + ["bureau_balance"]:
        mem.execute(f"create table {t} as select * from '{CLEAN / (t + '_clean.parquet')}'")

    print("[index] Building offset index...")
    t0 = time.perf_counter()
    idx = OffsetIndex(con)
    t_build = time.perf_counter() - t0
    mem_bytes = sum(tab["M"].nbytes + tab["uid"].nbytes + tab["start"].nbytes + tab["cnt"].nbytes
                    for tab in idx.tables.values())

    print("[index] Checking offset-index features against DuckDB...")
    ref = expensive_features(con, te_ids)
    mine, _ = idx.features(te_ids)
    mine = mine[ref.columns]
    a, b = ref.to_numpy(dtype=float), mine.to_numpy(dtype=float)
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    scale = np.maximum(np.abs(a), 1.0)
    max_rel = float(np.nanmax(np.abs(a - b) / scale))
    p_ref = exp.predict_proba(full_X(te, ref))[:, 1]
    p_mine = exp.predict_proba(full_X(te, mine))[:, 1]
    max_dp = float(np.max(np.abs(p_ref - p_mine)))
    print(f"  same NaN pattern: {same_nan}, max relative feature diff: {max_rel:.2e}, max |dp|: {max_dp:.2e}")

    def timings(ids):
        sub = te.set_index("SK_ID_CURR").loc[ids].reset_index()
        return {
            "duckdb_parquet": median_time(lambda: exp.predict_proba(full_X(sub, expensive_features(con, ids)))),
            "duckdb_memory": median_time(lambda: exp.predict_proba(full_X(sub, expensive_features(mem, ids)))),
            "offset_index": median_time(lambda: exp.predict_proba(full_X(sub, idx.features(ids)[0][ref.columns]))),
        }

    full = timings(te_ids)
    pc, pe, R, C = d["pc"], d["pe"], d["R"], d["C"]
    U_c, U_e = utility(pc, R, C), utility(pe, R, C)
    n = len(te)
    rows = []
    for alpha in ALPHAS:
        w = (R + C) * conformal_halfwidth(d["pc_val"], d["pe_val"], pc, alpha)
        for k_frac in K_FRACTIONS:
            k = int(round(k_frac * n))
            _, refined, _, _ = certify(U_c, U_e, w, k)
            ids = te_ids[refined]
            t = timings(ids)
            rows.append({"alpha": alpha, "k_fraction": k_frac, "k": k, "call_fraction": len(ids) / n,
                         "rows_fraction": float(d["rows_per_id"][refined].sum() / d["rows_per_id"].sum()),
                         **{f"sec_{m}": v for m, v in t.items()},
                         **{f"time_fraction_{m}": t[m] / full[m] for m in t}})

    summary = {"n_test": n, "index_build_seconds": t_build, "index_memory_GB": mem_bytes / 1e9,
               "check_same_nan_pattern": same_nan, "check_max_relative_feature_diff": max_rel,
               "check_max_abs_prob_diff": max_dp, "full_test_seconds": full, "configs": rows}
    out_dir = Path("paper/src_extension")
    with open(out_dir / "selective_topk_homecredit_index_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(rows).to_csv(out_dir / "selective_topk_homecredit_index_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"index build {t_build:.1f}s, {mem_bytes / 1e9:.2f} GB; full test set seconds: "
          + ", ".join(f"{m} {v:.3f}" for m, v in full.items()))
    print(pd.DataFrame(rows)[["alpha", "k_fraction", "call_fraction", "rows_fraction",
                              "time_fraction_duckdb_parquet", "time_fraction_duckdb_memory",
                              "time_fraction_offset_index", "sec_offset_index"]].round(4).to_string(index=False))
    print(f"\nSaved to {out_dir / 'selective_topk_homecredit_index_results.json'} (+ .csv)")


if __name__ == "__main__":
    main()

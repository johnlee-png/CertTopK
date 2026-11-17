"""Third domain (credit, Home Credit Default Risk) for the certified top-k
with selective inference experiment of src_extension_selective_topk_diagnostic.py.

Here the expensive model is expensive for a data-engineering reason: its
features are aggregates over six child tables (about 58M rows) that must be
joined to the selected applicants, while the cheap model reads only the
application row. Cost is reported as wall-clock time of the DuckDB
aggregation (plus model prediction) and as the number of child-table rows the
aggregation touches.

  P = 1 - P(default)     R = 0.10 * AMT_CREDIT   (assumed interest margin)
                         C = 0.45 * AMT_CREDIT   (Basel foundation-IRB LGD for
                                                  senior unsecured exposures)
Amounts are in thousands. Split: 70/15/15 by applicant (one row per
SK_ID_CURR in application_train), seed 42. Raw CSVs are left untouched;
sorted copies are written as *_clean.parquet.
"""
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src_extension_selective_topk_diagnostic import (
    K_FRACTIONS, N_RANDOM, certify, compare, conformal_halfwidth, mixed_topk, topk, utility,
)
import diabetes_new_regret_diagnostic as dia  # noqa: E402  (path registered by the import above)

RAW = Path("data/raw/home_credit")
CLEAN = Path("data/processed/home_credit")
CHILD = ["bureau", "previous_application", "installments_payments", "POS_CASH_balance", "credit_card_balance"]
ALPHAS = [0.10, 0.05, 0.01]
MARGIN, LGD, UNIT = 0.10, 0.45, 1000.0


def prepare(con):
    CLEAN.mkdir(parents=True, exist_ok=True)
    for t in ["application_train"] + CHILD:
        out = CLEAN / f"{t}_clean.parquet"
        if not out.exists():
            con.execute(f"copy (select * from read_csv_auto('{RAW / (t + '.csv')}') order by SK_ID_CURR) "
                        f"to '{out}' (format parquet)")
    out = CLEAN / "bureau_balance_clean.parquet"
    if not out.exists():
        con.execute(f"""copy (select b.SK_ID_CURR, bb.* from read_csv_auto('{RAW / 'bureau_balance.csv'}') bb
                        join read_csv_auto('{RAW / 'bureau.csv'}') b using (SK_ID_BUREAU)
                        order by b.SK_ID_CURR) to '{out}' (format parquet)""")
    for t in CHILD + ["bureau_balance"]:
        con.execute(f"create or replace view {t} as select * from '{CLEAN / (t + '_clean.parquet')}'")


def numeric_cols(con, table):
    cols = con.execute(f"describe {table}").df()
    keep = cols[cols["column_type"].isin(["BIGINT", "INTEGER", "DOUBLE", "FLOAT", "DECIMAL", "SMALLINT"])]
    return [c for c in keep["column_name"] if not c.startswith("SK_ID")]


def expensive_features(con, ids):
    """Aggregate the six child tables for the given applicant ids."""
    con.register("ids_df", pd.DataFrame({"SK_ID_CURR": np.asarray(ids, dtype=np.int64)}))
    con.execute("create or replace temp table ids as select * from ids_df")
    parts = []
    for t in CHILD:
        aggs = [f"count(*) as {t}_cnt"]
        for c in numeric_cols(con, t):
            aggs += [f'avg("{c}") as "{t}_{c}_mean"', f'max("{c}") as "{t}_{c}_max"', f'min("{c}") as "{t}_{c}_min"']
        parts.append(con.execute(f"select SK_ID_CURR, {', '.join(aggs)} from {t} join ids using (SK_ID_CURR) "
                                 f"group by SK_ID_CURR").df().set_index("SK_ID_CURR"))
    parts.append(con.execute("""select SK_ID_CURR, count(*) as bb_months,
                                       avg(case when STATUS in ('1','2','3','4','5') then 1.0 else 0.0 end) as bb_late_share
                                from bureau_balance join ids using (SK_ID_CURR) group by SK_ID_CURR""").df()
                 .set_index("SK_ID_CURR"))
    parts.append(con.execute("""select SK_ID_CURR,
                                       avg(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) as inst_days_late_mean,
                                       avg(AMT_PAYMENT / nullif(AMT_INSTALMENT, 0)) as inst_paid_ratio_mean
                                from installments_payments join ids using (SK_ID_CURR) group by SK_ID_CURR""").df()
                 .set_index("SK_ID_CURR"))
    return pd.DataFrame(index=pd.Index(ids, name="SK_ID_CURR")).join(parts)


def child_rows_per_id(con):
    union = " union all ".join(f"select SK_ID_CURR from {t}" for t in CHILD + ["bureau_balance"])
    return con.execute(f"select SK_ID_CURR, count(*) n from ({union}) group by 1").df().set_index("SK_ID_CURR")["n"]


def cheap_features(app):
    X = app.drop(columns=["SK_ID_CURR", "TARGET"]).copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    return X


def full_X(df, agg):
    return pd.concat([cheap_features(df).reset_index(drop=True),
                      agg.loc[df["SK_ID_CURR"].to_numpy()].reset_index(drop=True)], axis=1)


def build(con, seed=42):
    """Split, train both models and score the test applicants (shared with the index experiment)."""
    print("[home_credit] Preparing sorted parquet copies...")
    prepare(con)
    app = con.execute(f"select * from '{CLEAN / 'application_train_clean.parquet'}'").df()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(app))
    n_tr, n_va = int(0.70 * len(app)), int(0.15 * len(app))
    tr, va, te = app.iloc[perm[:n_tr]], app.iloc[perm[n_tr:n_tr + n_va]], app.iloc[perm[n_tr + n_va:]]

    print("[home_credit] Aggregating child tables for train/valid applicants (training-time cost)...")
    agg_trva = expensive_features(con, pd.concat([tr, va])["SK_ID_CURR"].to_numpy())

    print("[home_credit] Training cheap and expensive default models...")
    cheap = dia.train_lgb(cheap_features(tr), tr["TARGET"], cheap_features(va), va["TARGET"])
    exp = dia.train_lgb(full_X(tr, agg_trva), tr["TARGET"], full_X(va, agg_trva), va["TARGET"])

    pc_val = 1 - cheap.predict_proba(cheap_features(va))[:, 1]
    pe_val = 1 - exp.predict_proba(full_X(va, agg_trva))[:, 1]

    t0 = time.perf_counter()
    pc = 1 - cheap.predict_proba(cheap_features(te))[:, 1]
    t_cheap = time.perf_counter() - t0
    te_ids = te["SK_ID_CURR"].to_numpy()
    t0 = time.perf_counter()
    agg_te = expensive_features(con, te_ids)
    pe = 1 - exp.predict_proba(full_X(te, agg_te))[:, 1]
    t_full = time.perf_counter() - t0

    credit = te["AMT_CREDIT"].to_numpy(dtype=float) / UNIT
    credit_val = va["AMT_CREDIT"].to_numpy(dtype=float) / UNIT
    return {"te": te, "exp": exp, "pc_val": pc_val, "pe_val": pe_val, "pc": pc, "pe": pe,
            "R": MARGIN * credit, "C": LGD * credit,
            "R_val": MARGIN * credit_val, "C_val": LGD * credit_val, "t_cheap": t_cheap, "t_full": t_full,
            "xc_val": cheap_features(va).to_numpy(dtype=float), "xc": cheap_features(te).to_numpy(dtype=float),
            "rows_per_id": child_rows_per_id(con).reindex(te_ids).fillna(0).to_numpy()}


def main():
    con = duckdb.connect()
    d = build(con)
    te, exp, pc_val, pe_val, pc, pe = d["te"], d["exp"], d["pc_val"], d["pe_val"], d["pc"], d["pe"]
    R, C, t_cheap, t_full, rows_per_id = d["R"], d["C"], d["t_cheap"], d["t_full"], d["rows_per_id"]
    U_c, U_e = utility(pc, R, C), utility(pe, R, C)
    n = len(te)

    rows = []
    for alpha in ALPHAS:
        q = conformal_halfwidth(pc_val, pe_val, pc, alpha)
        w = (R + C) * q
        coverage = float(np.mean(np.abs(pe - pc) <= q))
        for k_frac in K_FRACTIONS:
            k = int(round(k_frac * n))
            ref = topk(U_e, k)
            T, refined, certified, rounds = certify(U_c, U_e, w, k)
            b = int(refined.sum())

            t0 = time.perf_counter()
            sel_te = te.iloc[np.flatnonzero(refined)]
            exp.predict_proba(full_X(sel_te, expensive_features(con, sel_te["SK_ID_CURR"].to_numpy())))
            t_sel = time.perf_counter() - t0

            ov_b, rg_b = compare(topk(U_c, k), ref, U_e, k)
            ov_c, rg_c = compare(T, ref, U_e, k)
            rs = np.random.default_rng(0)
            rand = [compare(mixed_topk(U_c, U_e, rs.choice(n, b, replace=False), k), ref, U_e, k)
                    for _ in range(N_RANDOM)]
            cut = np.sort(U_c)[::-1][k - 1]
            closeness = np.abs(U_c - cut) / np.maximum(w, 1e-12)
            ov_e, rg_e = compare(mixed_topk(U_c, U_e, np.argsort(closeness)[:b], k), ref, U_e, k)

            rows.append({
                "domain": "home_credit", "bound": "binned", "alpha": alpha, "bound_coverage": coverage,
                "mean_halfwidth_P": float(q.mean()), "k_fraction": k_frac, "n": n, "k": k,
                "certified": certified, "rounds": rounds,
                "expensive_calls_C": b, "call_fraction_C": b / n,
                "child_rows_fraction_C": float(rows_per_id[refined].sum() / rows_per_id.sum()),
                "seconds_cheap_all": t_cheap, "seconds_expensive_all": t_full,
                "seconds_expensive_refined_once": t_sel,
                "overlap_B": ov_b, "regret_B": rg_b, "overlap_C": ov_c, "regret_C": rg_c,
                "overlap_D_random": float(np.mean([x[0] for x in rand])),
                "regret_D_random": float(np.mean([x[1] for x in rand])),
                "overlap_E_uncertainty": ov_e, "regret_E_uncertainty": rg_e,
            })

    out_dir = Path("paper/src_extension")
    with open(out_dir / "selective_topk_homecredit_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "selective_topk_homecredit_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"n_test={n}  cheap all: {t_cheap:.2f}s  expensive all (aggregate+predict): {t_full:.2f}s  "
          f"ratio {t_full / t_cheap:.1f}x  child rows: {int(rows_per_id.sum()):,}")
    print(table[["alpha", "bound_coverage", "k_fraction", "k", "certified", "call_fraction_C",
                 "child_rows_fraction_C", "seconds_expensive_refined_once", "overlap_B", "overlap_C",
                 "overlap_D_random", "overlap_E_uncertainty", "regret_B", "regret_C"]].round(4).to_string(index=False))
    print(f"\nSaved to {out_dir / 'selective_topk_homecredit_results.json'} (+ .csv)")


if __name__ == "__main__":
    main()

"""Per-component wall-clock time of the certified top-k on Home Credit.

Seed 0, test split (1,000-applicant queries), kappa = 10%, eps in {0.05, 0.10}, same protocol as the main experiments
(separate fit split; CRC on the calibration queries). For each order (P0, PU, P2, P3c) and for running the expensive
path on everyone ("ALL") we time, on the whole test split:
  cheap      cheap-model inference (application features)
  scale      error-size model s_hat (all orders except P0 need it; PU uses its own regressor)
  priority   computing the priority of every candidate (P3c includes its B = 200 scenarios)
  access     fetching the expensive features of the refined applicants (key-sorted offset index over six tables)
  expensive  expensive-model inference on the refined applicants
and the one-time calibration (access + inference for all calibration applicants). Medians of 3 repeats.
Writes paper/src_extension/selective_topk_hc_time_breakdown.csv.
"""
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_baselines import topk_mask
from src_extension_selective_topk_homecredit import prepare
from src_extension_selective_topk_homecredit_index import OffsetIndex

SEED, K_FRAC, B = 0, 0.10, 200
OUT = Path("paper/src_extension/selective_topk_hc_time_breakdown.csv")


def med(fn, rep=3):
    ts, out = [], None
    for _ in range(rep):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def priorities(d, U_c, s, su, z_all, rng, names):
    """Priorities of every row of split d, computed query by query; returns {name: array}."""
    out = {n: np.empty(len(U_c)) for n in names}
    qid = d["qid"]
    for q in np.unique(qid):
        ix = np.flatnonzero(qid == q)
        k = max(1, int(round(K_FRAC * len(ix))))
        uc = U_c[ix]
        part = -np.partition(-uc, [k - 1, k])
        dist = np.abs(uc - 0.5 * (part[k - 1] + part[k]))
        budget = d["R"][ix] + d["C"][ix]
        if "P0" in names:
            out["P0"][ix] = dist
        if "P2" in names or "P3c" in names:
            p2 = dist / (budget * s[ix])
            if "P2" in names:
                out["P2"][ix] = p2
        if "PU" in names:
            out["PU"][ix] = dist / su[ix]
        if "P3c" in names:
            p_sim = d["pc"][ix][None, :] + s[ix][None, :] * rng.choice(z_all, size=(B, len(ix)))
            u_sim = np.clip(p_sim, 0, 1) * budget[None, :] - d["C"][ix][None, :]
            kth = -np.partition(-u_sim, k - 1, axis=1)[:, k - 1:k]
            flip = np.where(topk_mask(uc, k)[None, :], u_sim < kth, u_sim >= kth).mean(axis=0)
            out["P3c"][ix] = 1.0 - flip + 1e-6 * p2 / (p2.max() + 1e-12)
    return out


def main():
    crc.build_home_credit(SEED)                      # fills crc._HC
    app, Xc, Xf = crc._HC
    ids_all = app["SK_ID_CURR"].to_numpy()
    m = crc.group_split(ids_all, SEED)
    qid, m["cal"], m["test"] = crc.batched_cal_test(ids_all, m["cal"], m["test"], 1000, SEED)
    y = app["TARGET"]
    cheap = crc.dia.train_lgb(Xc[m["train"]], y[m["train"]], Xc[m["fit"]], y[m["fit"]])
    exp = crc.dia.train_lgb(Xf[m["train"]], y[m["train"]], Xf[m["fit"]], y[m["fit"]])
    agg_cols = [c for c in Xf.columns if c not in Xc.columns]
    data = {}
    for s in ["fit", "cal", "test"]:
        mask = m[s]
        credit = app.loc[mask, "AMT_CREDIT"].to_numpy(dtype=float) / crc.UNIT
        data[s] = {"pc": 1 - cheap.predict_proba(Xc[mask])[:, 1], "pe": 1 - exp.predict_proba(Xf[mask])[:, 1],
                   "R": crc.MARGIN * credit, "C": crc.LGD * credit, "xc": Xc[mask].to_numpy(dtype=float),
                   "qid": qid[mask] if s != "fit" else crc.batch_queries(ids_all[mask], 1000, SEED),
                   "ids": ids_all[mask], "mask": mask}
    f = data["fit"]
    scale = crc.fit_scale(f)
    uf_c, uf_e = crc.utility(f["pc"], f["R"], f["C"]), crc.utility(f["pe"], f["R"], f["C"])
    su_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                 verbosity=-1, random_state=42).fit(np.column_stack([f["xc"], f["pc"], f["R"], f["C"]]),
                                                                    np.abs(uf_e - uf_c))
    su_floor = float(np.quantile(np.abs(uf_e - uf_c), 0.05)) + 1e-9
    z_all = (f["pe"] - f["pc"]) / scale(f)
    names = ["P0", "PU", "P2", "P3c"]
    # thresholds from the fit grid + CRC on calibration (same rule as the main experiments)
    pri = {}
    for s, stream in (("fit", 0), ("cal", 1), ("test", 2)):
        d = data[s]
        U_c = crc.utility(d["pc"], d["R"], d["C"])
        su = np.maximum(su_model.predict(np.column_stack([d["xc"], d["pc"], d["R"], d["C"]])), su_floor)
        pri[s] = priorities(d, U_c, scale(d), su, z_all, np.random.default_rng([SEED, stream, 499]), names)
    con = duckdb.connect()
    prepare(con)
    idx = OffsetIndex(con)
    dt = data["test"]
    Xc_t, mask_t = Xc[dt["mask"]], dt["mask"]
    t_cheap, _ = med(lambda: cheap.predict_proba(Xc_t))
    t_scale, _ = med(lambda: scale(dt))
    t_su, _ = med(lambda: su_model.predict(np.column_stack([dt["xc"], dt["pc"], dt["R"], dt["C"]])))
    U_t = crc.utility(dt["pc"], dt["R"], dt["C"])
    s_t = scale(dt)
    su_t = np.maximum(su_model.predict(np.column_stack([dt["xc"], dt["pc"], dt["R"], dt["C"]])), su_floor)
    t_pri = {n: med(lambda n=n: priorities(dt, U_t, s_t, su_t, z_all, np.random.default_rng(1), [n]))[0] for n in names}

    Xf_t = Xf[mask_t]

    def expensive_path(ids, rows):
        # access: the offset index fetch (its aggregates equal the DuckDB ones, src_..._homecredit_index.py);
        # inference: the expensive model on the same rows (precomputed features, identical values)
        t_acc = med(lambda: idx.features(ids))[0] if len(ids) else 0.0
        X = Xf_t.iloc[rows]
        t_inf = med(lambda: exp.predict_proba(X))[0] if len(ids) else 0.0
        return t_acc, t_inf

    out = []
    acc_all, inf_all = expensive_path(dt["ids"], np.arange(len(dt["ids"])))
    cal_ids = data["cal"]["ids"]
    acc_cal = med(lambda: idx.features(cal_ids))[0]
    out.append({"order": "ALL", "eps": np.nan, "calls": 1.0, "cheap": 0.0, "scale": 0.0, "priority": 0.0,
                "access": acc_all, "expensive": inf_all, "calibration_access": acc_cal})
    for n in names:
        fitq = pri["fit"][n]
        grid = np.unique(np.concatenate([[-np.inf], np.quantile(fitq, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
        # calibration envelopes
        dc = data["cal"]
        U_cc, U_ce = crc.utility(dc["pc"], dc["R"], dc["C"]), crc.utility(dc["pe"], dc["R"], dc["C"])
        env = []
        for q in np.unique(dc["qid"]):
            ix = np.flatnonzero(dc["qid"] == q)
            k = max(1, int(round(K_FRAC * len(ix))))
            if k >= len(ix):
                continue
            qd = {"uc": U_cc[ix], "ue": U_ce[ix], "k": k, "pri": {n: pri["cal"][n][ix]}}
            env.append(crc.envelope(crc.loss_curve(qd, n, grid)[0]))
        env = np.array(env)
        mc = len(env)
        for eps in crc.EPSILONS:
            ok = np.flatnonzero(mc / (mc + 1) * env.mean(axis=0) + 1 / (mc + 1) <= eps)
            lam = grid[ok[0]] if len(ok) else np.inf
            sel = pri["test"][n] <= lam
            rows = np.flatnonzero(sel)
            t_acc, t_inf = expensive_path(dt["ids"][rows], rows)
            out.append({"order": n, "eps": eps, "calls": float(sel.mean()), "cheap": t_cheap,
                        "scale": 0.0 if n == "P0" else (t_su if n == "PU" else t_scale), "priority": t_pri[n],
                        "access": t_acc, "expensive": t_inf, "calibration_access": acc_cal})
        print(n, "done", flush=True)
    df = pd.DataFrame(out)
    df["total"] = df[["cheap", "scale", "priority", "access", "expensive"]].sum(axis=1)
    df["n_test"] = len(dt["ids"])
    df["n_cal"] = len(cal_ids)
    df.to_csv(OUT, index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

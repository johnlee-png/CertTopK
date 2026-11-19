"""Measured end-to-end time on Home Credit for the orders chosen by Auto-calls and Auto-time .

Seed 0, kappa = 10%, eps in {0.05, 0.10}, main protocol (separate fit split; CRC on the calibration queries).
1. Profiling (fit split only): per-candidate costs of cheap inference, s_hat, each order's priority, and one
   refinement (offset-index access + expensive inference) are measured on the fit applicants; Auto-time uses these
   constants with the fit-split plug-in calls (selective_topk_order_select_final_choices.csv) to pick its order.
2. Test (whole test split, 62,004 applicants): the full pipeline of P0, P1, P2, P3c and of the Auto-calls / Auto-time
   choices is run REP times; we report the median and the min-max of the total, and the median components.
   Refining everyone ("ALL") is timed with both access paths (offset index; DuckDB in-memory semi-join); the faster
   one is the baseline.
3. Set-up costs: offset-index build, expensive outputs of the fit split, and calibration (expensive outputs of the
   calibration split), each with the faster access path.
Writes paper/src_extension/selective_topk_hc_time_v2.{csv,json}.
"""
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_baselines import topk_mask
from src_extension_selective_topk_homecredit import CHILD, CLEAN, expensive_features, prepare
from src_extension_selective_topk_homecredit_index import OffsetIndex
from src_vldb_topk_ties import flip_prob

SEED, K_FRAC, B, REP = 0, 0.10, 200, 7
OUT = Path("paper/src_extension/selective_topk_hc_time_v2")
ORDERS = ["P0", "P1", "TOPM", "P2", "P3", "P3c"]


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def priority(d, U_c, s, z_all, rng, name):
    """One order's priority for every row of split d (computed query by query)."""
    out = np.empty(len(U_c))
    qid = d["qid"]
    for q in np.unique(qid):
        ix = np.flatnonzero(qid == q)
        k = max(1, int(round(K_FRAC * len(ix))))
        uc = U_c[ix]
        part = -np.partition(-uc, [k - 1, k])
        dist = np.abs(uc - 0.5 * (part[k - 1] + part[k]))
        budget = d["R"][ix] + d["C"][ix]
        if name == "P0":
            out[ix] = dist
        elif name == "P1":
            out[ix] = dist / budget
        elif name == "TOPM":
            r = np.empty(len(ix))
            r[np.argsort(-uc, kind="stable")] = np.arange(len(ix))
            out[ix] = r / k
        else:
            p2 = dist / (budget * s[ix])
            if name == "P2":
                out[ix] = p2
            else:
                p_sim = d["pc"][ix][None, :] + s[ix][None, :] * rng.choice(z_all, size=(B, len(ix)))
                if name == "P3c":
                    p_sim = np.clip(p_sim, 0, 1)
                u_sim = p_sim * budget[None, :] - d["C"][ix][None, :]
                flip = flip_prob(u_sim, topk_mask(uc, k), k)
                out[ix] = 1.0 - flip + 1e-6 * p2 / (p2.max() + 1e-12)
    return out


def needs_scale(o):
    return o in ("P2", "P3", "P3c")


def main():
    crc.build_home_credit(SEED)
    app, Xc, Xf = crc._HC
    ids_all = app["SK_ID_CURR"].to_numpy()
    m = crc.group_split(ids_all, SEED)
    qid, m["cal"], m["test"] = crc.batched_cal_test(ids_all, m["cal"], m["test"], 1000, SEED)
    y = app["TARGET"]
    cheap = crc.dia.train_lgb(Xc[m["train"]], y[m["train"]], Xc[m["fit"]], y[m["fit"]])
    exp = crc.dia.train_lgb(Xf[m["train"]], y[m["train"]], Xf[m["fit"]], y[m["fit"]])
    data = {}
    for s in ["fit", "cal", "test"]:
        mask = m[s]
        credit = app.loc[mask, "AMT_CREDIT"].to_numpy(dtype=float) / crc.UNIT
        data[s] = {"pc": 1 - cheap.predict_proba(Xc[mask])[:, 1], "pe": 1 - exp.predict_proba(Xf[mask])[:, 1],
                   "R": crc.MARGIN * credit, "C": crc.LGD * credit, "xc": Xc[mask].to_numpy(dtype=float),
                   "qid": qid[mask] if s != "fit" else crc.batch_queries(ids_all[mask], 1000, SEED),
                   "ids": ids_all[mask], "mask": mask}
    f = data["fit"]
    t_scale_train, scale = timed(lambda: crc.fit_scale(f))
    z_all = (f["pe"] - f["pc"]) / scale(f)
    con = duckdb.connect()
    prepare(con)
    t_index_build, idx = timed(lambda: OffsetIndex(con))
    mem = duckdb.connect()
    for t in CHILD + ["bureau_balance"]:
        mem.execute(f"create table {t} as select * from '{CLEAN / (t + '_clean.parquet')}'")

    def expensive(split, rows, path):
        ids = data[split]["ids"][rows]
        if len(ids) == 0:
            return 0.0, 0.0
        if path == "index":
            t_acc, _ = timed(lambda: idx.features(ids))
        else:
            t_acc, _ = timed(lambda: expensive_features(mem, ids))
        X = Xf[data[split]["mask"]].iloc[rows]
        t_inf, _ = timed(lambda: exp.predict_proba(X))
        return t_acc, t_inf

    # ---- 1. profiling on the fit split
    nf = len(f["ids"])
    Uf = crc.utility(f["pc"], f["R"], f["C"])
    prof = {"cheap": np.median([timed(lambda: cheap.predict_proba(Xc[f["mask"]]))[0] for _ in range(3)]) / nf,
            "scale": np.median([timed(lambda: scale(f))[0] for _ in range(3)]) / nf}
    for o in ORDERS:
        prof["pri_" + o] = np.median([timed(lambda: priority(f, Uf, scale(f) if needs_scale(o) else None, z_all,
                                                             np.random.default_rng(1), o))[0] for _ in range(3)]) / nf
    acc_f = [sum(expensive("fit", np.arange(nf), "index")) for _ in range(3)]
    prof["refine"] = float(np.median(acc_f)) / nf
    ch = pd.read_csv("paper/src_extension/selective_topk_order_select_final_choices.csv")
    ch = ch[(ch.domain == "home_credit") & (ch.seed == SEED) & (ch.k_fraction == K_FRAC)]
    cand = ["P2", "P3", "P3CLIP", "P1", "TOPM"]
    name_of = {"P3CLIP": "P3c"}
    choice = {}
    for _, r in ch.iterrows():
        pred = {o: prof["cheap"] + (prof["scale"] if needs_scale(name_of.get(o, o)) else 0.0)
                + prof["pri_" + name_of.get(o, o)] + r["fit_calls_" + o] * prof["refine"] for o in cand}
        choice[r.eps] = {"auto_calls": name_of.get(r.chosen, r.chosen), "auto_time": name_of.get(min(pred, key=pred.get),
                                                                                                  min(pred, key=pred.get)),
                         "predicted_per_candidate": {name_of.get(o, o): v for o, v in pred.items()}}
    # ---- thresholds (fit grid, CRC on calibration)
    pri = {}
    for s, stream in (("fit", 0), ("cal", 1), ("test", 2)):
        d = data[s]
        U = crc.utility(d["pc"], d["R"], d["C"])
        pri[s] = {o: priority(d, U, scale(d), z_all, np.random.default_rng([SEED, stream, 499]), o) for o in ORDERS}
    dc = data["cal"]
    Ucc, Uce = crc.utility(dc["pc"], dc["R"], dc["C"]), crc.utility(dc["pe"], dc["R"], dc["C"])

    def lam_hat(o, eps):
        grid = np.unique(np.concatenate([[-np.inf], np.quantile(pri["fit"][o], np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
        env = []
        for q in np.unique(dc["qid"]):
            ix = np.flatnonzero(dc["qid"] == q)
            k = max(1, int(round(K_FRAC * len(ix))))
            if k < len(ix):
                env.append(crc.envelope(crc.loss_curve({"uc": Ucc[ix], "ue": Uce[ix], "k": k, "pri": {o: pri["cal"][o][ix]}},
                                                       o, grid)[0]))
        env = np.array(env)
        mc = len(env)
        ok = np.flatnonzero(mc / (mc + 1) * env.mean(axis=0) + 1 / (mc + 1) <= eps)
        return grid[ok[0]] if len(ok) else np.inf

    # ---- 2. test pipelines
    dt = data["test"]
    nt = len(dt["ids"])
    Ut = crc.utility(dt["pc"], dt["R"], dt["C"])
    rows = []
    for path in ("index", "duckdb_memory"):
        tot = []
        for _ in range(REP):
            a, i = expensive("test", np.arange(nt), path)
            tot.append(a + i)
        rows.append({"order": "ALL", "path": path, "eps": np.nan, "calls": 1.0, "total_median": np.median(tot),
                     "total_min": np.min(tot), "total_max": np.max(tot)})
    for eps in crc.EPSILONS:
        measured = sorted(set(["P0", "P1", "P2", "P3c", choice[eps]["auto_calls"], choice[eps]["auto_time"]]))
        for o in measured:
            sel_rows = np.flatnonzero(pri["test"][o] <= lam_hat(o, eps))
            comps = []
            for _ in range(REP):
                t_c, _ = timed(lambda: cheap.predict_proba(Xc[dt["mask"]]))
                t_s, s_t = timed(lambda: scale(dt)) if needs_scale(o) else (0.0, None)
                t_p, _ = timed(lambda: priority(dt, Ut, s_t, z_all, np.random.default_rng(2), o))
                t_a, t_i = expensive("test", sel_rows, "index")
                comps.append((t_c + t_s + t_p, t_a + t_i))
            tot = [a + b for a, b in comps]
            rows.append({"order": o, "path": "index", "eps": eps, "calls": len(sel_rows) / nt,
                         "overhead_median": np.median([c[0] for c in comps]),
                         "expensive_median": np.median([c[1] for c in comps]),
                         "total_median": np.median(tot), "total_min": np.min(tot), "total_max": np.max(tot),
                         "is_auto_calls": o == choice[eps]["auto_calls"], "is_auto_time": o == choice[eps]["auto_time"],
                         "predicted_total": choice[eps]["predicted_per_candidate"].get(o, np.nan) * nt})
            print(rows[-1], flush=True)
    # ---- 3. set-up costs (faster access path for full reads)
    fast = min(("index", "duckdb_memory"), key=lambda p: [r for r in rows if r["order"] == "ALL" and r["path"] == p][0]["total_median"])
    setup = {"index_build_s": t_index_build, "scale_train_s": t_scale_train,
             "fit_outputs_s": float(np.median([sum(expensive("fit", np.arange(nf), fast)) for _ in range(3)])),
             "calibration_s": float(np.median([sum(expensive("cal", np.arange(len(dc["ids"])), fast)) for _ in range(3)])),
             "fast_full_path": fast, "n_fit": nf, "n_cal": len(dc["ids"]), "n_test": nt,
             "profile_per_candidate": prof, "choices": {str(k): v for k, v in choice.items()}}
    pd.DataFrame(rows).to_csv(OUT.with_suffix(".csv"), index=False)
    OUT.with_suffix(".json").write_text(json.dumps(setup, indent=2, default=float))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(json.dumps(setup, indent=2, default=float))


if __name__ == "__main__":
    main()

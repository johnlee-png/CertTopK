"""Home Credit, measured end to end.

Every timed number is a call of src_vldb_e2e.answer on one test query: cheap inference on the query's applicants,
priority, selection, the actual feature access for the selected applicants (offset index in RAM, DuckDB in-memory
tables, or DuckDB over the parquet files), expensive inference on the features just read, and the mixed top-k. The
returned answer is scored against the reference top-k in the same loop. Refining everyone (ALL) goes through the same
function with the same start and end.

Grid: query sizes (default 250, 1000, 2500 rows per query; calibration/test queries re-formed for each size with crc.batched_cal_test,
so CRC is recalibrated per size), access paths (index, duckdb_memory, duckdb_parquet), eps in crc.EPSILONS, kappa = K_FRAC, seed SEED.
Set-up costs are measured separately and split into "reuse" (expensive outputs of the fit and calibration splits are
already logged) and "fresh" (they must be produced through the access path).

Writes paper/src_extension/e2e/hc_{queries.csv,setup.json}.
Usage: py src_vldb_e2e_hc.py [sizes] [paths] [max_test_queries]
"""
import json
import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
import src_vldb_e2e as e2e
from src_extension_selective_topk_homecredit import CHILD, CLEAN, expensive_features, prepare
from src_extension_selective_topk_homecredit_index import OffsetIndex

SEED, K_FRAC = 0, 0.10
OUT = Path("paper/src_extension/e2e")
POLICIES = ["ALL"] + e2e.ORDERS


def tic(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


class HomeCredit:
    seed = SEED

    def __init__(self):
        crc.build_home_credit(SEED)
        self.app, self.Xc, self.Xf = crc._HC
        self.ids = self.app["SK_ID_CURR"].to_numpy()
        self.agg_cols = [c for c in self.Xf.columns if c not in self.Xc.columns]
        self.split = crc.group_split(self.ids, SEED)
        y = self.app["TARGET"]
        tr, fi = self.split["train"], self.split["fit"]
        self.t_models, (self.cheap_m, self.exp_m) = tic(lambda: (
            crc.dia.train_lgb(self.Xc[tr], y[tr], self.Xc[fi], y[fi]),
            crc.dia.train_lgb(self.Xf[tr], y[tr], self.Xf[fi], y[fi])))
        credit = self.app["AMT_CREDIT"].to_numpy(dtype=float) / crc.UNIT
        self.R_all, self.C_all = crc.MARGIN * credit, crc.LGD * credit
        # reference expensive outputs (offline, from the full feature table): the ground truth of the loss only
        self.pe_ref = 1 - self.exp_m.predict_proba(self.Xf)[:, 1]
        self.path = None

    # ---- access paths
    def open_paths(self):
        cost = {}
        cost["index_build_s"], self.idx = tic(lambda: OffsetIndex(self._parquet_con()))
        self.mem = duckdb.connect()

        def load():
            for t in CHILD + ["bureau_balance"]:
                self.mem.execute(f"create table {t} as select * from '{CLEAN / (t + '_clean.parquet')}'")
        cost["duckdb_memory_load_s"], _ = tic(load)
        self.pq = self._parquet_con()
        cost["duckdb_parquet_open_s"] = 0.0
        return cost

    @staticmethod
    def _parquet_con():
        con = duckdb.connect()
        prepare(con)
        return con

    def features(self, ids):
        if self.path == "index":
            F = self.idx.features(ids)[0]
        else:
            F = expensive_features(self.mem if self.path == "duckdb_memory" else self.pq, ids)
        return F.reindex(columns=self.agg_cols)

    # ---- per-query steps used by e2e.answer
    def cheap(self, q):
        X = self.Xc.iloc[q["rows"]]
        return 1 - self.cheap_m.predict_proba(X)[:, 1], X

    def R(self, q):
        return self.R_all[q["rows"]]

    def C(self, q):
        return self.C_all[q["rows"]]

    def k(self, q):
        return max(1, int(round(K_FRAC * len(q["rows"]))))

    def qkey(self, q):
        return q["key"]

    def _side(self, q, pc, X):
        return np.column_stack([X.to_numpy(dtype=float), pc, self.R(q), self.C(q)])

    def scale(self, q, pc, X):
        return np.maximum(self.s_model.predict(self._side(q, pc, X)), self.s_floor)

    def su(self, q, pc, X):
        return np.maximum(self.su_model.predict(self._side(q, pc, X)), self.su_floor)

    def expensive(self, q, sel, X):
        F = self.features(self.ids[q["rows"][sel]])
        Xs = pd.concat([X.iloc[sel].reset_index(drop=True), F.reset_index(drop=True)], axis=1)[self.Xf.columns]
        return 1 - self.exp_m.predict_proba(Xs)[:, 1]

    # ---- offline pieces
    def fit_side_models(self, fit_rows):
        """s_hat (|P^e - P^c|) and su (|U^e - U^c|) on the fit split, from the fit split's expensive outputs."""
        X = self.Xc.iloc[fit_rows]
        pc = 1 - self.cheap_m.predict_proba(X)[:, 1]
        pe = self.pe_ref[fit_rows]
        R, C = self.R_all[fit_rows], self.C_all[fit_rows]
        Z = np.column_stack([X.to_numpy(dtype=float), pc, R, C])
        kw = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbosity=-1, random_state=42)
        self.s_model = lgb.LGBMRegressor(**kw).fit(Z, np.abs(pe - pc))
        self.s_floor = float(np.quantile(np.abs(pe - pc), 0.05)) + 1e-6
        du = np.abs(e2e.utility(pe, R, C) - e2e.utility(pc, R, C))
        self.su_model = lgb.LGBMRegressor(**kw).fit(Z, du)
        self.su_floor = float(np.quantile(du, 0.05)) + 1e-9
        self.z_all = (pe - pc) / np.maximum(self.s_model.predict(Z), self.s_floor)


def make_queries(wl, size):
    """Fit, calibration and test queries of about `size` applicants (cal/test exactly exchangeable)."""
    ids, sp = wl.ids, wl.split
    qid, cal, test = crc.batched_cal_test(ids, sp["cal"], sp["test"], size, SEED)
    fit_rows = np.flatnonzero(sp["fit"])
    fq = crc.batch_queries(ids[fit_rows], size, SEED)
    out = {"fit": [{"rows": fit_rows[fq == q], "key": 10 ** 6 + int(q)} for q in np.unique(fq)]}
    for name, mask, base in (("cal", cal, 2 * 10 ** 6), ("test", test, 3 * 10 ** 6)):
        rows = np.flatnonzero(mask)
        out[name] = [{"rows": rows[qid[rows] == q], "key": base + int(q)} for q in np.unique(qid[rows])]
    return out


def offline_query(wl, q, names):
    """uc, ue, k and the priorities of one fit/cal query, through the same priority code as the online path."""
    pc, X = wl.cheap(q)
    R, C = wl.R(q), wl.C(q)
    uc, ue, k = e2e.utility(pc, R, C), e2e.utility(wl.pe_ref[q["rows"]], R, C), wl.k(q)
    s, su = wl.scale(q, pc, X), wl.su(q, pc, X)
    pri = {}
    for n in names:
        rng = np.random.default_rng([wl.seed, 777, wl.qkey(q)]) if n == "P3c" else None
        pri[n] = e2e.priority(n, uc, k, R, C, pc, s, su, wl.z_all, rng)
    return {"uc": uc, "ue": ue, "k": k, "pri": pri}


def calibrate(wl, qs):
    """Per order: grid (fit), fit envelopes/fractions (for Auto), cal envelopes; and lambda_hat per eps.
    Returns (state, per-order seconds of the offline computation)."""
    state, secs = {}, {}
    for n in e2e.ORDERS:
        t0 = time.perf_counter()
        fit = [offline_query(wl, q, [n]) for q in qs["fit"]]
        cal = [offline_query(wl, q, [n]) for q in qs["cal"]]
        grid = e2e.grid_of(np.concatenate([q["pri"][n] for q in fit]))
        env_f, frac_f = e2e.curves(fit, n, grid)
        env_c, _ = e2e.curves(cal, n, grid)
        state[n] = {"lam": {eps: e2e.crc_lambda(env_c, grid, eps) for eps in crc.EPSILONS},
                    "fit_calls": {eps: e2e.plug_in_calls(env_f, frac_f, eps) for eps in crc.EPSILONS}}
        secs[n] = time.perf_counter() - t0
    return state, secs


def profile(wl, qs, n_q=5):
    """Auto-time cost model on fit queries: per-order overhead per query (cheap + priority + top-k) and the
    expensive step as a + b * (number refined), fitted on refinements of 1, 5%, 20% and 100% of the query."""
    fit = qs["fit"][:n_q]
    over = {}
    for n in e2e.ORDERS:
        ts = []
        for q in fit:
            t = {}
            e2e.answer(wl, q, n, -np.inf, t)            # lambda = -inf: no refinement, overhead only
            ts.append(t["cheap"] + t["priority"] + t["topk"])
        over[n] = float(np.median(ts))
    t = {}
    for q in fit:
        e2e.answer(wl, q, "P0", -np.inf, t)
    over["ALL"] = t["cheap"] / len(fit)
    xs, ys = [], []
    for q in fit:
        n_rows = len(q["rows"])
        pc, X = wl.cheap(q)
        for m in sorted({1, max(1, n_rows // 20), n_rows // 5, n_rows}):
            sel = np.arange(m)
            ts = [tic(lambda: wl.expensive(q, sel, X))[0] for _ in range(2)]
            xs.append(m)
            ys.append(min(ts))
    b, a = np.polyfit(xs, ys, 1)
    return {"overhead": over, "exp_a": float(max(a, 0.0)), "exp_b": float(b)}


def choose(state, prof, size, eps):
    calls = {n: state[n]["fit_calls"][eps] for n in e2e.ORDERS}
    pred = {n: prof["overhead"][n] + (prof["exp_a"] if calls[n] > 0 else 0.0) + prof["exp_b"] * calls[n] * size
            for n in e2e.ORDERS}
    pred["ALL"] = prof["overhead"]["ALL"] + prof["exp_a"] + prof["exp_b"] * size
    return min(calls, key=calls.get), min(pred, key=pred.get), pred


def main():
    sizes = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [250, 1000, 2500]
    paths = sys.argv[2].split(",") if len(sys.argv) > 2 else ["index", "duckdb_memory", "duckdb_parquet"]
    max_test = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    OUT.mkdir(parents=True, exist_ok=True)
    wl = HomeCredit()
    setup = {"models_train_s (workload, not CertTopK)": wl.t_models}
    setup.update(wl.open_paths())
    fit_rows = np.flatnonzero(wl.split["fit"])
    setup["side_models_s"], _ = tic(lambda: wl.fit_side_models(fit_rows))
    rows, per_size = [], {}
    for size in sizes:
        qs = make_queries(wl, size)
        qs["test"] = qs["test"][:max_test]
        state, cal_secs = calibrate(wl, qs)
        info = {"n_fit_q": len(qs["fit"]), "n_cal_q": len(qs["cal"]), "n_test_q": len(qs["test"]),
                "calibration_compute_s": cal_secs,
                "lambda": {n: {str(e): float(v) for e, v in state[n]["lam"].items()} for n in e2e.ORDERS},
                "fit_calls": {n: {str(e): v for e, v in state[n]["fit_calls"].items()} for n in e2e.ORDERS},
                "paths": {}}
        for path in paths:
            wl.path = path
            q0 = qs["test"][0]
            pc0, X0 = wl.cheap(q0)
            gap = float(np.max(np.abs(wl.expensive(q0, np.arange(len(q0["rows"])), X0) - wl.pe_ref[q0["rows"]])))
            print(f"[{path}] max |P^e online - P^e reference| on one query: {gap:.2e}", flush=True)
            # fresh install: expensive outputs of the fit and calibration splits through this access path
            fresh = 0.0
            for split in ("fit", "cal"):
                for q in qs[split]:
                    pc, X = wl.cheap(q)
                    fresh += tic(lambda: wl.expensive(q, np.arange(len(q["rows"])), X))[0]
            t_prof, prof = tic(lambda: profile(wl, qs))
            pinfo = {"fresh_expensive_outputs_s": fresh, "profiling_s": t_prof, "profile": prof, "choice": {}}
            for eps in crc.EPSILONS:
                a_calls, a_time, pred = choose(state, prof, size, eps)
                pinfo["choice"][str(eps)] = {"auto_calls": a_calls, "auto_time": a_time, "predicted_s": pred}
            info["paths"][path] = pinfo
            for pol in POLICIES:
                for eps in ([np.nan] if pol == "ALL" else crc.EPSILONS):
                    lam = np.inf if pol == "ALL" else state[pol]["lam"][eps]
                    e2e.answer(wl, qs["test"][0], pol, lam)          # warm-up
                    for q in qs["test"]:
                        st = {}
                        t0 = time.perf_counter()
                        top, n_calls = e2e.answer(wl, q, pol, lam, st)
                        lat = time.perf_counter() - t0
                        k = wl.k(q)
                        ue = e2e.utility(wl.pe_ref[q["rows"]], wl.R(q), wl.C(q))
                        rows.append({"size": size, "path": path, "eps": eps, "policy": pol, "qkey": q["key"],
                                     "n": len(q["rows"]), "k": k, "latency_s": lat, "calls": n_calls,
                                     "loss": e2e.loss(top, ue, k), **{f"t_{a}": b for a, b in st.items()}})
                    sub = pd.DataFrame(rows[-len(qs["test"]):])
                    print(f"size {size} {path} {pol} eps {eps}: mean {sub.latency_s.mean():.4f}s "
                          f"p95 {sub.latency_s.quantile(.95):.4f}s calls {sub.calls.mean() / size:.3f} "
                          f"loss {sub.loss.mean():.4f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "hc_queries.csv", index=False)
        per_size[str(size)] = info
        setup["per_size"] = per_size
        (OUT / "hc_setup.json").write_text(json.dumps(setup, indent=2, default=float))


if __name__ == "__main__":
    main()

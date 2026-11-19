"""LC-LLM (Lending Club + local Qwen2.5-7B), measured end to end.

Each timed number is a call of src_vldb_e2e.answer on one test query: cheap LightGBM on the listing, priority,
selection, fetching the selected loans' text from the loan-text table (DuckDB), the actual LLM calls (llama.cpp
server, CONC concurrent requests), the expensive LightGBM on listing + the fresh LLM logit, and the mixed top-k.
The answer is scored against the reference top-k (expensive model on the logged LLM scores) in the same loop; the
fresh LLM scores are also compared with the logged ones. ALL goes through the same function.

Grid: query sizes SIZES (cal/test queries re-formed per size, CRC recalibrated per size), concurrency CONCS,
eps in crc.EPSILONS, kappa = K_FRAC, seed SEED. Selective policies run on the first N_TEST test queries; ALL on the
first N_ALL (default N_TEST, the same queries). Set-up: the LLM outputs of the fit and calibration splits are produced
through the server once ("fresh" install; skipped with --no-fresh) and timed. Per size, the loss of every policy on
all test queries is also computed offline with the stored LLM scores in place of the calls (llm{TAG}_offline_loss.csv);
with the deterministic concurrency-1 scores (CERTTOPK_LLM_SCORES=...scores_c1.csv) an online run at concurrency 1
reproduces them exactly, so the oracle, the reference, the ALL fallback and the calibration share one loss.

Needs the server:  llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf -ngl 99 -c 16384 -np 8 --port 8091
Writes paper/src_extension/e2e/llm_{queries.csv,setup.json,fresh_scores.csv}.
Usage: py src_vldb_e2e_llm.py [sizes] [concs] [n_test] [n_all] [--no-fresh] [--tag=suffix]
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc

_argv, sys.argv = sys.argv, sys.argv[:1]     # llm_trial parses N and PORT from sys.argv at import
import src_extension_selective_topk_llm_trial as trial
sys.argv = _argv
import src_vldb_e2e as e2e

trial.PORT = 8091
SEED, K_FRAC = 0, 0.10
args = [a for a in sys.argv[1:] if not a.startswith("--")]
SIZES = [int(x) for x in args[0].split(",")] if len(args) > 0 else [100, 250, 500]
CONCS = [int(x) for x in args[1].split(",")] if len(args) > 1 else [1, 4, 8]
N_TEST = int(args[2]) if len(args) > 2 else 20
N_ALL = int(args[3]) if len(args) > 3 else N_TEST
FRESH = "--no-fresh" not in sys.argv
TAG = next((a[6:] for a in sys.argv[1:] if a.startswith("--tag=")), "")   # output name suffix
OUT = Path("paper/src_extension/e2e")
TEXT = Path("data/processed/lending_club/lc_llm_text_clean.parquet")
POLICIES = ["ALL"] + e2e.ORDERS


def tic(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def build_text_table(ids):
    """One-time copy of the prompt fields of the scored loans (raw gz is read only)."""
    need = set(ids)
    parts = [ch[ch["id"].astype(str).isin(need)]
             for ch in pd.read_csv(trial.RAW, usecols=["id"] + trial.COLS, chunksize=200_000, low_memory=False)]
    t = pd.concat(parts).drop_duplicates("id")
    t["id"] = t["id"].astype(str)
    TEXT.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(TEXT, index=False)


class LendingClubLLM:
    seed = SEED

    def __init__(self):
        df = crc.lc_llm_frame()
        self.df = df
        self.ids = df["id"].to_numpy(dtype=object)
        self.Xc = crc.encode(df, crc.LC_LISTING).astype(np.float32)
        self.Xf = crc.encode(df, crc.LC_LISTING + ["llm_logit"]).astype(np.float32)
        self.split = crc.group_split(self.ids, SEED)
        y = pd.Series(df["repaid"].to_numpy())
        tr, fi = self.split["train"], self.split["fit"]
        self.t_models, (self.cheap_m, self.exp_m) = tic(lambda: (
            crc.dia.train_lgb(self.Xc[tr], y[tr], self.Xc[fi], y[fi]),
            crc.dia.train_lgb(self.Xf[tr], y[tr], self.Xf[fi], y[fi])))
        self.col_c, self.col_e = list(self.cheap_m.classes_).index(1), list(self.exp_m.classes_).index(1)
        amt = df["loan_amnt"].to_numpy(dtype=float)
        off = tr & (df["repaid"].to_numpy() == 0)
        lgd = float(np.clip((amt - df["total_rec_prncp"].to_numpy() - df["recoveries"].to_numpy())[off] / amt[off],
                            0, 1).mean())
        self.R_all = np.maximum(df["installment"].to_numpy() * df["term"].to_numpy() - amt, 0.0)
        self.C_all = lgd * amt
        self.pe_ref = self.exp_m.predict_proba(self.Xf)[:, self.col_e]
        if not TEXT.exists():
            build_text_table(set(map(str, self.ids)))
        self.db = duckdb.connect()
        self.db.execute(f"create table loan_text as select * from '{TEXT}'")
        self.conc = 8
        self.offline = False     # True: stored LLM scores instead of calls (offline loss only, never timed)
        self.fresh_log = []

    # ---- per-query steps
    def cheap(self, q):
        X = self.Xc.iloc[q["rows"]]
        return self.cheap_m.predict_proba(X)[:, self.col_c], X

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

    def llm_logit(self, ids):
        self.db.register("want", pd.DataFrame({"id": list(ids), "pos": np.arange(len(ids))}))
        t = self.db.execute("select l.*, w.pos from want w join loan_text l using (id) order by w.pos").df()
        assert len(t) == len(ids), "loan text missing"
        texts = [trial.prompt(r) for r in t.itertuples()]
        with ThreadPoolExecutor(self.conc) as ex:
            p = np.array([r[0] for r in ex.map(trial.ask, texts)], dtype=float)
        p = np.clip(np.nan_to_num(p, nan=0.5), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def expensive(self, q, sel, X):
        rows = q["rows"][sel]
        logit = (self.df["llm_logit"].to_numpy()[rows] if self.offline else self.llm_logit(self.ids[rows]))
        self.fresh_log.append((rows, logit))
        Xs = X.iloc[sel].copy()
        Xs["llm_logit"] = logit.astype(np.float32)
        return self.exp_m.predict_proba(Xs[self.Xf.columns])[:, self.col_e]

    # ---- offline
    def fit_side_models(self, fit_rows):
        X = self.Xc.iloc[fit_rows]
        pc = self.cheap_m.predict_proba(X)[:, self.col_c]
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


def main():
    import src_vldb_e2e_hc as hc        # shares make_queries / calibrate / profile / choose (workload-agnostic)
    OUT.mkdir(parents=True, exist_ok=True)
    wl = LendingClubLLM()
    trial.ask(trial.prompt(wl.db.execute("select * from loan_text limit 1").df().iloc[0]))     # server warm-up
    setup = {"models_train_s (workload, not CertTopK)": wl.t_models}
    fit_rows = np.flatnonzero(wl.split["fit"])
    setup["side_models_s"], _ = tic(lambda: wl.fit_side_models(fit_rows))
    if FRESH:
        # fresh install: LLM outputs of every fit and calibration loan through the server (concurrency 8)
        rows = np.flatnonzero(wl.split["fit"] | wl.split["cal"])
        wl.conc = 8
        t, logit = tic(lambda: wl.llm_logit(wl.ids[rows]))
        logged = wl.df["llm_logit"].to_numpy()[rows]
        setup["fresh_llm_outputs_s"] = t
        setup["fresh_llm_loans"] = int(len(rows))
        setup["fresh_vs_logged_logit_mae"] = float(np.mean(np.abs(logit - logged)))
        pd.DataFrame({"id": wl.ids[rows], "fresh_logit": logit, "logged_logit": logged}).to_csv(
            OUT / "llm_fresh_scores.csv", index=False)
        print("fresh set-up", {k: v for k, v in setup.items()}, flush=True)
    out_rows, per_size = [], {}
    for size in SIZES:
        qs = hc.make_queries(wl, size)
        state, cal_secs = hc.calibrate(wl, qs)
        info = {"n_fit_q": len(qs["fit"]), "n_cal_q": len(qs["cal"]), "n_test_q": len(qs["test"]),
                "calibration_compute_s": cal_secs,
                "lambda": {n: {str(e): float(v) for e, v in state[n]["lam"].items()} for n in e2e.ORDERS},
                "fit_calls": {n: {str(e): v for e, v in state[n]["fit_calls"].items()} for n in e2e.ORDERS},
                "conc": {}}
        wl.offline, off_rows = True, []
        for pol in POLICIES:
            for eps in ([np.nan] if pol == "ALL" else crc.EPSILONS):
                lam = np.inf if pol == "ALL" else state[pol]["lam"][eps]
                for q in qs["test"]:
                    top, n_calls = e2e.answer(wl, q, pol, lam)
                    k = wl.k(q)
                    ue = e2e.utility(wl.pe_ref[q["rows"]], wl.R(q), wl.C(q))
                    off_rows.append({"size": size, "eps": eps, "policy": pol, "qkey": q["key"], "calls": n_calls,
                                     "loss": e2e.loss(top, ue, k)})
        wl.offline = False
        off_path = OUT / f"llm{TAG}_offline_loss.csv"
        prev = pd.read_csv(off_path) if (off_path.exists() and size != SIZES[0]) else None
        pd.concat([prev, pd.DataFrame(off_rows)]).to_csv(off_path, index=False)
        for conc in CONCS:
            wl.conc = conc
            t_prof, prof = tic(lambda: hc.profile(wl, qs, n_q=2))
            cinfo = {"profiling_s": t_prof, "profile": prof, "choice": {}}
            for eps in crc.EPSILONS:
                a_calls, a_time, pred = hc.choose(state, prof, size, eps)
                cinfo["choice"][str(eps)] = {"auto_calls": a_calls, "auto_time": a_time, "predicted_s": pred}
            info["conc"][str(conc)] = cinfo
            for pol in POLICIES:
                test = qs["test"][:N_ALL if pol == "ALL" else N_TEST]
                for eps in ([np.nan] if pol == "ALL" else crc.EPSILONS):
                    lam = np.inf if pol == "ALL" else state[pol]["lam"][eps]
                    e2e.answer(wl, test[0], pol if pol != "ALL" else "P0", -np.inf)   # warm-up, no LLM call
                    for q in test:
                        wl.fresh_log.clear()
                        st = {}
                        t0 = time.perf_counter()
                        top, n_calls = e2e.answer(wl, q, pol, lam, st)
                        lat = time.perf_counter() - t0
                        k = wl.k(q)
                        ue = e2e.utility(wl.pe_ref[q["rows"]], wl.R(q), wl.C(q))
                        gap = (float(np.mean(np.abs(wl.fresh_log[0][1] - wl.df["llm_logit"].to_numpy()[wl.fresh_log[0][0]])))
                               if wl.fresh_log else np.nan)
                        out_rows.append({"size": size, "conc": conc, "eps": eps, "policy": pol, "qkey": q["key"],
                                         "n": len(q["rows"]), "k": k, "latency_s": lat, "calls": n_calls,
                                         "loss": e2e.loss(top, ue, k), "logit_mae_vs_logged": gap,
                                         **{f"t_{a}": b for a, b in st.items()}})
                    sub = pd.DataFrame(out_rows[-len(test):])
                    print(f"size {size} conc {conc} {pol} eps {eps}: mean {sub.latency_s.mean():.3f}s "
                          f"p95 {sub.latency_s.quantile(.95):.3f}s calls {sub.calls.mean() / size:.3f} "
                          f"loss {sub.loss.mean():.4f}", flush=True)
                    pd.DataFrame(out_rows).to_csv(OUT / f"llm{TAG}_queries.csv", index=False)
        per_size[str(size)] = info
        setup["per_size"] = per_size
        (OUT / f"llm{TAG}_setup.json").write_text(json.dumps(setup, indent=2, default=float))


if __name__ == "__main__":
    main()

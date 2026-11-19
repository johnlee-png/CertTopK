"""Stronger baselines and finer reporting requested in.

For each (domain, seed) and kappa, with the splits, models and queries of src_extension_selective_topk_crc.py:
  LTT     Learn-then-Test (Hoeffding-Bentkus, delta = 0.1) on the envelope ("LTT_ENV") and on the raw, non-monotone
          loss ("LTT_RAW"); fixed-sequence testing is valid without monotonicity, and raw <= envelope.
  JOINT   query-level simultaneous conformal intervals + exact top-k certification: with a_i = (R_i + C_i) s_hat_i,
          M_Q = max_i |U^e_i - U^c_i| / a_i on the calibration queries, q = conformal (1 - alpha) quantile of M_Q,
          intervals U^c_i +- q a_i cover all candidates of a new query with probability >= 1 - alpha; refine one
          undecided candidate at a time (widest interval) until the top-k is determined. With alpha = eps,
          E[mis-selection] <= eps; when coverage holds the answer is exact.
  PU      priority |U^c - theta_c| / s_U with s_U a LightGBM regressor of |U^e - U^c| fitted directly on the fit split
          (no budget factorization), calibrated with CRC like P2.
  P3CLIP  P3 with the simulated probabilities clipped to [0, 1]; also the out-of-range rate of P3's scenarios.
  DISTILL the same offline oracle budget (expensive outputs of the fit and calibration splits) used to train a cheap
          regressor of P^e on cheap information (xc, P^c, R, C); its top-k is returned with no online calls.
  Per-query distribution of the membership loss at the CRC threshold of P2 (median, p90, p95, share > eps, share > 0),
  the normalized utility regret D_Q = (sum_{T*} U^e - sum_T U^e) / (sum_{T*} U^e - sum_{bottom k} U^e) at that
  threshold, CRC on the regret envelope ("CRC_REGRET"), and tail control: CRC on 1{L~_Q > eps0} at level alpha_t
  ("CRC_TAIL", eps0 = 0.1, alpha_t = 0.1).
  Safe pruning from the deterministic bounds -C_i <= U^e_i <= R_i (share of candidates decided without refinement).
Also registers "diabetes_rc": the diabetes workload with R and C's source columns given to the cheap model.

Writes paper/src_extension/selective_topk_c5_{results,perquery}.csv.
Usage: py src_extension_selective_topk_baselines.py <n_seeds> [domains] [workers] | summarize
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_ltt import hb_pvalue
from src_extension_selective_topk_p3_extra import arrays, iid_sampler, p3_priority
from src_vldb_topk_ties import flip_prob

OUT = Path("paper/src_extension")
DELTA = 0.10
EPS0, ALPHA_T = 0.10, 0.10
B = 200

_DIA_RC = None


def build_diabetes_rc(seed):
    global _DIA_RC
    if _DIA_RC is None:
        df = pd.read_csv(crc.dia.DATA_DIR / "diabetic_data.csv")
        df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)
        cols = list(dict.fromkeys(list(crc.dia.DEMOGRAPHIC_COLS) + ["number_inpatient", "time_in_hospital"]))
        _DIA_RC = (df, crc.encode(df, cols), crc.encode(df, crc.dia.ALL_COLS))
    df, Xc, Xf = _DIA_RC
    max_stay = df["time_in_hospital"].max()
    return crc.build_tabular(df["patient_nbr"].to_numpy(), df["readmitted_30"].to_numpy(), Xc, Xf,
                             1.0 + df["number_inpatient"].to_numpy(dtype=float),
                             0.05 * df["time_in_hospital"].to_numpy(dtype=float) / max_stay, 500, seed)


crc.DOMAINS["diabetes_rc"] = build_diabetes_rc


def topk_mask(score, k):
    m = np.zeros(len(score), bool)
    m[crc.topk(score, k)] = True
    return m


def curves(q, name, grid):
    """Membership loss, normalized regret and call fraction on the grid for one query."""
    k, ue, uc = q["k"], q["ue"], q["uc"]
    ref = topk_mask(ue, k)
    best = ue[ref].sum()
    worst = np.sort(ue)[:k].sum()
    den = best - worst
    pri = q["pri"][name]
    L, R, F = np.empty(len(grid)), np.empty(len(grid)), np.empty(len(grid))
    for j, lam in enumerate(grid):
        refined = pri <= lam
        ans = topk_mask(np.where(refined, ue, uc), k)
        L[j] = 1 - (ans & ref).sum() / k
        R[j] = (best - ue[ans].sum()) / den if den > 0 else 0.0
        F[j] = refined.mean()
    return L, R, F


def crc_index(env_cal, eps):
    m = env_cal.shape[0]
    ok = np.flatnonzero(m / (m + 1) * env_cal.mean(axis=0) + 1 / (m + 1) <= eps)
    return ok[0] if len(ok) else env_cal.shape[1] - 1


def ltt_index(cal, eps):
    m = cal.shape[0]
    rhat = cal.mean(axis=0)
    j = cal.shape[1] - 1
    while j > 0 and hb_pvalue(rhat[j - 1], eps, m) <= DELTA:
        j -= 1
    return j


def certify(uc, ue, a, q, k):
    """Exact top-k certification with intervals uc +- q a; returns (refined mask, answer mask)."""
    n = len(uc)
    refined = np.zeros(n, bool)
    if not np.isfinite(q):
        return np.ones(n, bool), topk_mask(ue, k)
    while True:
        lb = np.where(refined, ue, uc - q * a)
        ub = np.where(refined, ue, uc + q * a)
        sub, slb = np.sort(ub), np.sort(lb)
        n_ub_gt = n - np.searchsorted(sub, lb, side="right") - (ub > lb)      # others with ub_j > lb_i
        n_lb_gt = n - np.searchsorted(slb, ub, side="right")                   # others with lb_j > ub_i
        sure_in = n_ub_gt < k
        sure_out = n_lb_gt >= k
        open_ = ~(sure_in | sure_out) & ~refined
        if sure_in.sum() == k:
            return refined, sure_in          # under coverage these k are exactly the reference top-k
        if not open_.any():
            break
        cand = np.flatnonzero(open_)
        refined[cand[np.argmax(a[cand])]] = True
    return refined, topk_mask(np.where(refined, ue, uc), k)


def safe_prune_share(q, R, C):
    """Share of candidates decided by the deterministic bounds -C <= U^e <= R alone."""
    k, n = q["k"], len(q["uc"])
    lo, hi = -C, R
    n_hi_gt = n - np.searchsorted(np.sort(hi), lo, side="right") - (hi > lo)
    n_lo_gt = n - np.searchsorted(np.sort(lo), hi, side="right")
    return float(((n_hi_gt < k) | (n_lo_gt >= k)).mean())


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    f = data["fit"]
    uf_c, uf_e = crc.utility(f["pc"], f["R"], f["C"]), crc.utility(f["pe"], f["R"], f["C"])
    Xf = np.column_stack([f["xc"], f["pc"], f["R"], f["C"]])
    su_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                 verbosity=-1, random_state=42).fit(Xf, np.abs(uf_e - uf_c))
    su_floor = float(np.quantile(np.abs(uf_e - uf_c), 0.05)) + 1e-9
    fc = data["cal"]
    Xd = np.vstack([Xf, np.column_stack([fc["xc"], fc["pc"], fc["R"], fc["C"]])])
    distill = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                verbosity=-1, random_state=42).fit(Xd, np.concatenate([f["pe"], fc["pe"]]))
    z_all = (f["pe"] - f["pc"]) / scale(f)
    rows, pq = [], []
    for k_frac in crc.K_FRACTIONS:
        sets, extra = {}, {}
        for split, stream in (("fit", 0), ("cal", 1), ("test", 2)):
            d = data[split]
            qs = arrays(d, scale, k_frac, seed, stream)
            s_all = scale(d)
            su_all = np.maximum(su_model.predict(np.column_stack([d["xc"], d["pc"], d["R"], d["C"]])), su_floor)
            width = (d["R"] + d["C"]) * s_all
            rng = np.random.default_rng([seed, stream, 99])
            rng_c = np.random.default_rng([seed, stream, 499])
            oor, tot = 0, 0
            for q in qs:
                ix, k = q["_ix"], q["k"]
                dist = q["pri"]["P1"] * (d["R"][ix] + d["C"][ix])          # |U^c - theta_c|
                q["pri"]["PU"] = dist / su_all[ix]
                q["pri"]["P3"] = p3_priority(q["uc"], width[ix], k, q["pri"]["P2"], B, iid_sampler(z_all), rng)
                # P3 with simulated probabilities clipped to [0, 1]: U = P (R + C) - C
                zz = rng_c.choice(z_all, size=(B, len(ix)))
                p_sim = d["pc"][ix][None, :] + s_all[ix][None, :] * zz
                oor += int(((p_sim < 0) | (p_sim > 1)).sum())
                tot += p_sim.size
                u_sim = np.clip(p_sim, 0, 1) * (d["R"][ix] + d["C"][ix])[None, :] - d["C"][ix][None, :]
                flip = flip_prob(u_sim, topk_mask(q["uc"], k), k)
                p2 = q["pri"]["P2"]
                q["pri"]["P3CLIP"] = 1.0 - flip + 1e-6 * p2 / (p2.max() + 1e-12)
                q["_a"] = width[ix]
            sets[split] = qs
            extra[split] = oor / max(tot, 1)
        test, cal = sets["test"], sets["cal"]
        m = len(cal)
        base = {"domain": domain, "seed": seed, "k_fraction": k_frac, "m_cal": m}
        for name in ("P2", "PU", "P3", "P3CLIP"):
            pooled = np.concatenate([q["pri"][name] for q in sets["fit"]])
            grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
            cc = [curves(q, name, grid) for q in cal]
            tc = [curves(q, name, grid) for q in test]
            Lc, Rc = np.array([c[0] for c in cc]), np.array([c[1] for c in cc])
            Lt, Rt, Ft = np.array([c[0] for c in tc]), np.array([c[1] for c in tc]), np.array([c[2] for c in tc])
            envL = np.array([crc.envelope(x) for x in Lc])
            for eps in crc.EPSILONS:
                j = crc_index(envL, eps)
                r = {**base, "eps": eps, "priority": name, "test_miss": Lt[:, j].mean(), "test_calls": Ft[:, j].mean(),
                     "test_regret": Rt[:, j].mean()}
                if name == "P3":
                    r["p3_out_of_range"] = extra["test"]
                rows.append({**r, "method": "CRC"})
                if name != "P2":
                    continue
                lq = Lt[:, j]
                pq.append({**base, "eps": eps, "median": np.median(lq), "p90": np.quantile(lq, 0.9),
                           "p95": np.quantile(lq, 0.95), "share_gt_eps": (lq > eps).mean(), "share_gt_0": (lq > 0).mean(),
                           "regret_p95": np.quantile(Rt[:, j], 0.95), "n_test": len(lq)})
                for meth, cal_curves in (("LTT_ENV", envL), ("LTT_RAW", Lc)):
                    jj = ltt_index(cal_curves, eps)
                    rows.append({**base, "eps": eps, "priority": name, "method": meth, "test_miss": Lt[:, jj].mean(),
                                 "test_calls": Ft[:, jj].mean(), "test_regret": Rt[:, jj].mean()})
                envR = np.array([crc.envelope(x) for x in Rc])
                jr = crc_index(envR, eps)
                rows.append({**base, "eps": eps, "priority": name, "method": "CRC_REGRET", "test_miss": Lt[:, jr].mean(),
                             "test_calls": Ft[:, jr].mean(), "test_regret": Rt[:, jr].mean()})
            if name == "P2":
                tail = (envL > EPS0).astype(float)
                jt = crc_index(tail, ALPHA_T)
                rows.append({**base, "eps": EPS0, "priority": name, "method": "CRC_TAIL", "test_miss": Lt[:, jt].mean(),
                             "test_calls": Ft[:, jt].mean(), "test_regret": Rt[:, jt].mean(),
                             "tail_share": (Lt[:, jt] > EPS0).mean()})
        # JOINT: query-level simultaneous intervals + exact certification
        Mq = np.sort([np.max(np.abs(q["ue"] - q["uc"]) / q["_a"]) for q in cal])
        for eps in crc.EPSILONS:
            r_ = int(np.ceil((m + 1) * (1 - eps)))
            qv = Mq[r_ - 1] if r_ <= m else np.inf
            miss, calls, exact = [], [], []
            for q in test:
                ref_, ans = certify(q["uc"], q["ue"], q["_a"], qv, q["k"])
                l_ = 1 - (ans & topk_mask(q["ue"], q["k"])).sum() / q["k"]
                miss.append(l_)
                calls.append(ref_.mean())
                exact.append(l_ == 0)
            rows.append({**base, "eps": eps, "priority": "JOINT", "method": "JOINT", "test_miss": np.mean(miss),
                         "test_calls": np.mean(calls), "exact_rate": np.mean(exact), "q": qv})
        # DISTILL: same offline oracle outputs, no online calls; cheap-only answer for reference
        dt = data["test"]
        pd_all = distill.predict(np.column_stack([dt["xc"], dt["pc"], dt["R"], dt["C"]]))
        ud_all = np.clip(pd_all, 0, 1) * (dt["R"] + dt["C"]) - dt["C"]
        md, mc, prune = [], [], []
        for q in test:
            ref = topk_mask(q["ue"], q["k"])
            md.append(1 - (topk_mask(ud_all[q["_ix"]], q["k"]) & ref).sum() / q["k"])
            mc.append(1 - (topk_mask(q["uc"], q["k"]) & ref).sum() / q["k"])
            prune.append(safe_prune_share(q, dt["R"][q["_ix"]], dt["C"][q["_ix"]]))
        rows.append({**base, "eps": np.nan, "priority": "DISTILL", "method": "DISTILL", "test_miss": np.mean(md),
                     "test_calls": 0.0})
        rows.append({**base, "eps": np.nan, "priority": "CHEAP", "method": "CHEAP", "test_miss": np.mean(mc),
                     "test_calls": 0.0, "safe_prune_share": np.mean(prune)})
    return rows, pq, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def summarize():
    t = pd.read_csv(OUT / "selective_topk_c5_results.csv")
    t["over"] = (t.test_miss > t.eps).astype(float)
    g = t.groupby(["domain", "k_fraction", "eps", "method", "priority"], dropna=False).agg(
        seeds=("seed", "nunique"), miss=("test_miss", "mean"), calls=("test_calls", "mean"),
        regret=("test_regret", "mean"), over=("over", "sum"), exact=("exact_rate", "mean"),
        oor=("p3_out_of_range", "mean"), tail=("tail_share", "mean"), prune=("safe_prune_share", "mean")).reset_index()
    g.to_csv(OUT / "selective_topk_c5_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 2000):
        print(g.round(4).to_string(index=False))
    p = pd.read_csv(OUT / "selective_topk_c5_perquery.csv")
    pp = p.groupby(["domain", "k_fraction", "eps"])[["median", "p90", "p95", "share_gt_eps", "share_gt_0", "regret_p95"]].mean()
    pp.to_csv(OUT / "selective_topk_c5_perquery_summary.csv")
    print(pp.round(3).to_string())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        return summarize()
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else list(crc.DOMAINS)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    paths = {"r": OUT / "selective_topk_c5_results.csv", "p": OUT / "selective_topk_c5_perquery.csv"}
    acc = {k: (pd.read_csv(p).to_dict("records") if p.exists() else []) for k, p in paths.items()}
    done = {(r["domain"], r["seed"]) for r in acc["r"]}
    todo = [(d, s) for d in domains for s in range(n_seeds) if (d, s) not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    with Pool(workers) as pool:
        for r, p, msg in pool.imap_unordered(task, todo):
            acc["r"] += r
            acc["p"] += p
            pd.DataFrame(acc["r"]).to_csv(paths["r"], index=False)
            pd.DataFrame(acc["p"]).to_csv(paths["p"], index=False)
            print(msg, flush=True)
    summarize()


if __name__ == "__main__":
    main()

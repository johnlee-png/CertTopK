"""Cost-accuracy trade-off at matched accuracy (Pareto curves).

The guarantee experiments compare methods at their own targets (CRC: expected
miss <= eps; Everest: P(exact top-k) >= thres), which are not the same. Here
every method is given the same per-query call budget f (fraction of tuples sent
to the expensive model) and we measure the miss 1 - |T ∩ T*| / k on the test
queries, for f on a grid. This compares the *orders* in which the methods spend
calls, independent of how each decides when to stop.

  P0, P1, P2, RND, TOPM  refine the ceil(f n) tuples of lowest priority, answer
                         = top-k of the mixed scores (as in the CRC script)
  EVEREST                the tuples cleaned by Everest's selection rule after
                         ceil(f n) calls (it starts by cleaning the cheap top-k),
                         answer = top-k of the cleaned tuples (Everest's answer)
Everest is run once per query until its confidence reaches CONF_STOP or every
tuple is cleaned; the trajectory gives its miss for every budget and its stop
point for every threshold. Beyond the stop point the answer is held fixed.

Splits, models and queries are those of src_extension_selective_topk_crc.py.
Writes paper/src_extension/selective_topk_pareto_{curves,thresholds}.csv.
Usage: py src_extension_selective_topk_pareto.py <n_seeds> <domains> [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
import src_extension_selective_topk_everest as ev
from src_extension_selective_topk_diagnostic import utility

F_GRID = np.round(np.linspace(0, 1, 101), 2)
CONF_STOP = 0.999
THRES_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
OUT = Path("paper/src_extension")


def miss_of(answer, ref, k):
    return 1 - len(ref & set(answer.tolist())) / k


def everest_trajectory(uc, ue, width, k, Fz):
    """Everest's selection rule (same computations as ev.everest_query), run to
    CONF_STOP; returns per step (cleaned count, miss of Everest's answer, confidence)."""
    n = len(uc)
    ref = set(crc.topk(ue, k).tolist())
    cleaned = np.zeros(n, dtype=bool)
    cleaned[crc.topk(uc, k)] = True

    def cdf(idx, t):
        return Fz((np.asarray(t)[None, :] - uc[idx, None]) / width[idx, None])
    steps = []
    key, u_idx, Fg, logF = None, None, None, None
    while True:
        c_idx = np.flatnonzero(cleaned)
        order = c_idx[np.argsort(-ue[c_idx], kind="stable")]  # ties: smaller index first, as in crc.topk
        answer = order[:k]
        s_k, s_p = ue[order[k - 1]], ue[order[k - 2]] if k >= 2 else ue[order[k - 1]]
        if (s_k, s_p) != key:
            key = (s_k, s_p)
            u_idx = np.flatnonzero(~cleaned)
            if len(u_idx):
                grid = np.linspace(s_k, s_p, ev.N_INT) if s_p > s_k else np.array([s_k, s_k])
                Fg = np.clip(cdf(u_idx, grid), 1e-300, 1.0)
                logF = np.log(Fg)
        if len(u_idx) == 0:
            steps.append((int(cleaned.sum()), miss_of(answer, ref, k), 1.0))
            return steps
        conf = float(np.exp(logF[:, 0].sum()))
        steps.append((int(cleaned.sum()), miss_of(answer, ref, k), conf))
        if conf >= CONF_STOP:
            return steps
        logG = logF.sum(axis=0)[None, :] - logF
        G = np.exp(logG)
        mass = np.diff(Fg, axis=1)
        mid = 0.5 * (G[:, 1:] + G[:, :-1])
        ex = Fg[:, 0] * G[:, 0] + (mass * mid).sum(axis=1) + (1 - Fg[:, -1]) * G[:, -1]
        pos = int(np.argmax(ex))
        cleaned[u_idx[pos]] = True
        u_idx, Fg, logF = np.delete(u_idx, pos), np.delete(Fg, pos, axis=0), np.delete(logF, pos, axis=0)


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    escale, z = ev.fit_score_model(data["fit"], seed)
    Fz = ev.make_cdf(z)
    d = data["test"]
    U_c, U_e = utility(d["pc"], d["R"], d["C"]), utility(d["pe"], d["R"], d["C"])
    width = (d["R"] + d["C"]) * escale(d)
    curves, thres_rows = [], []
    for k_frac in crc.K_FRACTIONS:
        queries = crc.query_arrays(d, scale, k_frac, seed)
        qids = [q for q in np.unique(d["qid"]) if 1 <= max(1, int(round(k_frac * np.sum(d["qid"] == q)))) < np.sum(d["qid"] == q)]
        acc = {name: np.zeros(len(F_GRID)) for name in crc.PRIORITIES + ["EVEREST"]}
        stops = {t: [] for t in THRES_GRID}
        for qr, q in zip(queries, qids):
            ix = np.flatnonzero(d["qid"] == q)
            n, k = len(ix), qr["k"]
            ref = set(crc.topk(qr["ue"], k).tolist())
            m_of_f = np.ceil(F_GRID * n).astype(int)
            for name in crc.PRIORITIES:
                order = np.argsort(qr["pri"][name], kind="stable")
                for j, m in enumerate(m_of_f):
                    mixed = qr["uc"].copy()
                    mixed[order[:m]] = qr["ue"][order[:m]]
                    acc[name][j] += miss_of(crc.topk(mixed, k), ref, k)
            steps = everest_trajectory(U_c[ix], U_e[ix], width[ix], k, Fz)
            cnt = np.array([s[0] for s in steps])
            mis = np.array([s[1] for s in steps])
            conf = np.array([s[2] for s in steps])
            pos = np.clip(np.searchsorted(cnt, np.maximum(m_of_f, k), side="right") - 1, 0, len(cnt) - 1)
            acc["EVEREST"] += mis[pos]
            for t in THRES_GRID:
                hit = np.flatnonzero(conf >= t)
                i = hit[0] if len(hit) else len(steps) - 1
                stops[t].append((cnt[i] / n, mis[i], float(mis[i] == 0), conf[i]))
        nq = len(queries)
        for name, a in acc.items():
            for f, v in zip(F_GRID, a / nq):
                curves.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "method": name,
                               "call_fraction": f, "miss": v})
        for t, s in stops.items():
            s = np.array(s)
            thres_rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "thres": t,
                               "calls": s[:, 0].mean(), "miss": s[:, 1].mean(), "exact_rate": s[:, 2].mean(),
                               "claimed_conf": s[:, 3].mean(), "n_queries": len(s)})
    return curves, thres_rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def main():
    n_seeds, domains = int(sys.argv[1]), sys.argv[2].split(",")
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    curves, thres_rows = [], []
    done = set()
    if "--resume" in sys.argv:
        # keep the (domain, seed) pairs already present in BOTH output files and run only the rest
        c0 = pd.read_csv(OUT / "selective_topk_pareto_curves.csv")
        t0 = pd.read_csv(OUT / "selective_topk_pareto_thresholds.csv")
        done = set(zip(c0.domain, c0.seed)) & set(zip(t0.domain, t0.seed))
        curves = c0[[p in done for p in zip(c0.domain, c0.seed)]].to_dict("records")
        thres_rows = t0[[p in done for p in zip(t0.domain, t0.seed)]].to_dict("records")
    todo = [(d, s) for d in domains for s in range(n_seeds) if (d, s) not in done]
    print(f"{len(done)} (domain, seed) pairs kept, {len(todo)} to run", flush=True)
    with Pool(workers) as pool:
        for c, t, msg in pool.imap_unordered(task, todo):
            curves += c
            thres_rows += t
            print(msg, flush=True)
            pd.DataFrame(curves).to_csv(OUT / "selective_topk_pareto_curves.csv", index=False)
            pd.DataFrame(thres_rows).to_csv(OUT / "selective_topk_pareto_thresholds.csv", index=False)


if __name__ == "__main__":
    main()

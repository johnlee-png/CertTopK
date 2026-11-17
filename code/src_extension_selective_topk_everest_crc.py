"""Everest's refinement order with the CRC calibration ("certified Everest").

Theorem 1 needs only a nested family of refinement sets indexed by lambda, so it
applies to any order in which tuples are sent to the expensive model, including
Everest's adaptive order. Per query, the order is: Everest's selection rule
(start with the cheap top-k, then argmax E[X_f]) until its confidence reaches
CONF_STOP, then the remaining tuples by P0 (so that refining everything is
reachable and the loss at the last grid point is 0). lambda is the call
fraction on a fixed grid (0, 0.01, ..., 1); the answer is the top-k of the mixed
scores, as for the other priorities. lambda_hat is chosen by CRC on the
envelopes of the calibration queries, exactly as in src_extension_selective_topk_crc.py.

The adaptive order uses expensive scores of tuples already refined; that is
allowed because the refined set after c calls is a function of the query alone
(and of models fit on other splits), and it grows with c.
Writes paper/src_extension/selective_topk_everest_crc_{results,summary}.csv.
Usage: py src_extension_selective_topk_everest_crc.py <n_seeds> <domains> [workers]
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
OUT = Path("paper/src_extension")


def everest_order(uc, ue, width, k, Fz):
    n = len(uc)
    cleaned = np.zeros(n, dtype=bool)
    first = crc.topk(uc, k)
    first = first[np.argsort(-uc[first], kind="stable")]
    cleaned[first] = True
    seq = list(first)

    def cdf(idx, t):
        return Fz((np.asarray(t)[None, :] - uc[idx, None]) / width[idx, None])
    key, u_idx, Fg, logF = None, None, None, None
    while True:
        c_idx = np.flatnonzero(cleaned)
        order = c_idx[np.argsort(-ue[c_idx], kind="stable")]  # ties: smaller index first, as in crc.topk
        s_k, s_p = ue[order[k - 1]], ue[order[k - 2]] if k >= 2 else ue[order[k - 1]]
        if (s_k, s_p) != key:
            key = (s_k, s_p)
            u_idx = np.flatnonzero(~cleaned)
            if len(u_idx):
                grid = np.linspace(s_k, s_p, ev.N_INT) if s_p > s_k else np.array([s_k, s_k])
                Fg = np.clip(cdf(u_idx, grid), 1e-300, 1.0)
                logF = np.log(Fg)
        if len(u_idx) == 0:
            break
        if float(np.exp(logF[:, 0].sum())) >= CONF_STOP:
            break
        logG = logF.sum(axis=0)[None, :] - logF
        G = np.exp(logG)
        mass = np.diff(Fg, axis=1)
        mid = 0.5 * (G[:, 1:] + G[:, :-1])
        ex = Fg[:, 0] * G[:, 0] + (mass * mid).sum(axis=1) + (1 - Fg[:, -1]) * G[:, -1]
        pos = int(np.argmax(ex))
        cleaned[u_idx[pos]] = True
        seq.append(u_idx[pos])
        u_idx, Fg, logF = np.delete(u_idx, pos), np.delete(Fg, pos, axis=0), np.delete(logF, pos, axis=0)
    rest = np.flatnonzero(~cleaned)
    part = -np.partition(-uc, [k - 1, k])
    theta = 0.5 * (part[k - 1] + part[k])
    rest = rest[np.argsort(np.abs(uc[rest] - theta), kind="stable")]
    return np.concatenate([np.asarray(seq, dtype=int), rest])


def curves(d, k_frac, Fz, width):
    U_c, U_e = utility(d["pc"], d["R"], d["C"]), utility(d["pe"], d["R"], d["C"])
    L, F = [], []
    for q in np.unique(d["qid"]):
        ix = np.flatnonzero(d["qid"] == q)
        n = len(ix)
        k = max(1, int(round(k_frac * n)))
        if k >= n:
            continue
        uc, ue = U_c[ix], U_e[ix]
        order = everest_order(uc, ue, width[ix], k, Fz)
        ref = set(crc.topk(ue, k).tolist())
        m_of_f = np.ceil(F_GRID * n).astype(int)
        l = np.empty(len(F_GRID))
        for j, m in enumerate(m_of_f):
            mixed = uc.copy()
            mixed[order[:m]] = ue[order[:m]]
            l[j] = 1 - len(ref & set(crc.topk(mixed, k).tolist())) / k
        L.append(l)
        F.append(m_of_f / n)
    return np.array(L), np.array(F)


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    escale, z = ev.fit_score_model(data["fit"], seed)
    Fz = ev.make_cdf(z)
    rows = []
    for k_frac in crc.K_FRACTIONS:
        Lc, _ = curves(data["cal"], k_frac, Fz, (data["cal"]["R"] + data["cal"]["C"]) * escale(data["cal"]))
        Lt, Ft = curves(data["test"], k_frac, Fz, (data["test"]["R"] + data["test"]["C"]) * escale(data["test"]))
        Lt_cal = np.array([crc.envelope(l) for l in Lc])
        m = len(Lc)
        for eps in crc.EPSILONS:
            ok = np.flatnonzero(m / (m + 1) * Lt_cal.mean(axis=0) + 1 / (m + 1) <= eps)
            j = ok[0] if len(ok) else len(F_GRID) - 1
            rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps, "priority": "EVEREST_CRC",
                         "m_cal": m, "lambda_index": int(j), "test_miss": float(Lt[:, j].mean()),
                         "test_call_fraction": float(Ft[:, j].mean())})
    return rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def main():
    n_seeds, domains = int(sys.argv[1]), sys.argv[2].split(",")
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    # Resume: (domain, seed) pairs already in the results file are kept and not rerun.
    path = OUT / "selective_topk_everest_crc_results.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["domain"], r["seed"]) for r in rows}
    todo = [(d, s) for d in domains for s in range(n_seeds) if (d, s) not in done]
    print(f"resuming: {len(done)} done, {len(todo)} to run", flush=True)
    with Pool(workers) as pool:
        for r, msg in pool.imap_unordered(task, todo):
            rows += r
            print(msg, flush=True)
            pd.DataFrame(rows).to_csv(OUT / "selective_topk_everest_crc_results.csv", index=False)
    t = pd.DataFrame(rows).assign(over=lambda x: (x.test_miss > x.eps).astype(int))
    summ = t.groupby(["domain", "k_fraction", "eps"]).agg(
        seeds=("seed", "nunique"), miss_mean=("test_miss", "mean"), miss_sd=("test_miss", "std"),
        seeds_over_eps=("over", "sum"), calls_mean=("test_call_fraction", "mean")).reset_index()
    summ.to_csv(OUT / "selective_topk_everest_crc_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

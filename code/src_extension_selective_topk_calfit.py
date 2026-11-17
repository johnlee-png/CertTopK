"""No fit split: learn the error-size model s_hat and the grid from the calibration queries.

Reviewer concern: the main protocol refines a separate fit split (10% of the groups) to train s_hat. Here the
calibration queries, which are refined anyway, are split at random into two halves: half A trains s_hat and sets the
grid, half B calibrates (CRC), and the test queries are unchanged. The fit split is not used at all. Given A, the
queries of B and the test queries are still a uniformly random split of the remaining queries, so Theorem 1 holds
with m = |B|. The expensive model itself is treated as given (trained on the training split as before).
Priority P2 only. Writes paper/src_extension/selective_topk_calfit_{results,summary}.csv.
Usage: py src_extension_selective_topk_calfit.py <n_seeds> [domains] [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc

OUT = Path("paper/src_extension")


def subset(d, mask):
    return {key: v[mask] for key, v in d.items()}


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    cal = data["cal"]
    qids = np.unique(cal["qid"])
    perm = np.random.default_rng([seed, 11]).permutation(qids)
    a_ids = perm[: len(perm) // 2]
    in_a = np.isin(cal["qid"], a_ids)
    half_a, half_b = subset(cal, in_a), subset(cal, ~in_a)
    scale = crc.fit_scale(half_a)
    rows = []
    for k_frac in crc.K_FRACTIONS:
        qa = crc.query_arrays(half_a, scale, k_frac, seed, stream=0)
        qb = crc.query_arrays(half_b, scale, k_frac, seed, stream=1)
        qt = crc.query_arrays(data["test"], scale, k_frac, seed, stream=2)
        pooled = np.concatenate([q["pri"]["P2"] for q in qa])
        grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
        lt_b = np.array([crc.envelope(crc.loss_curve(q, "P2", grid)[0]) for q in qb])
        curves = [crc.loss_curve(q, "P2", grid) for q in qt]
        l_test = np.array([c[0] for c in curves])
        f_test = np.array([c[1] for c in curves])
        m = len(qb)
        for eps in crc.EPSILONS:
            ok = np.flatnonzero(m / (m + 1) * lt_b.mean(axis=0) + 1 / (m + 1) <= eps)
            j = ok[0] if len(ok) else len(grid) - 1
            rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps, "priority": "P2",
                         "m_scale": len(qa), "m_cal": m, "test_miss": float(l_test[:, j].mean()),
                         "test_call_fraction": float(f_test[:, j].mean())})
    return rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else list(crc.DOMAINS)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    path = OUT / "selective_topk_calfit_results.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["domain"], r["seed"]) for r in rows}
    todo = [(d, s) for d in domains for s in range(n_seeds) if (d, s) not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    with Pool(workers) as pool:
        for r, msg in pool.imap_unordered(task, todo):
            rows += r
            print(msg, flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
    t = pd.DataFrame(rows).assign(over=lambda x: (x.test_miss > x.eps).astype(int))
    summ = t.groupby(["domain", "k_fraction", "eps"]).agg(
        seeds=("seed", "nunique"), m_scale=("m_scale", "mean"), m_cal=("m_cal", "mean"),
        miss_mean=("test_miss", "mean"), seeds_over_eps=("over", "sum"),
        calls_mean=("test_call_fraction", "mean")).reset_index()
    full = pd.read_csv(OUT / "selective_topk_crc_summary.csv")
    full = full[full.priority == "P2"][["domain", "k_fraction", "eps", "calls_mean"]].rename(columns={"calls_mean": "calls_fit_split"})
    summ = summ.merge(full, on=["domain", "k_fraction", "eps"], how="left")
    summ.to_csv(OUT / "selective_topk_calfit_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

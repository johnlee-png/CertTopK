"""M5: queries defined by predicates instead of random batches.

Each query is "SELECT ... WHERE A = a AND B = b ORDER BY U DESC LIMIT k" over the
calibration + test rows of the group split, for every (a, b) with at least
N_MIN rows:
  lending_club  addr_state, purpose
  ieee_fraud    ProductCD, P_emaildomain, card6 (debit/credit)
The (a, b) cells partition the rows, so the queries are disjoint; their sizes
differ widely (that is the point). The query list is fixed on the pooled
calibration + test rows and a uniformly random half of the queries goes to
calibration; a uniform random permutation of a fixed list is exchangeable, so
Theorem 1 applies as for the batch queries. Models, the scale model and the
lambda grid come from the train/fit splits exactly as in the main runs (the fit
queries keep their batch form). Reports the mean miss over test queries
(the workload average the guarantee is about) and, separately, by query size.
Writes paper/src_extension/selective_topk_predicate_{results,bysize,summary}.csv.
Usage: py src_extension_selective_topk_predicate.py <n_seeds> [domains] [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc

PREDICATES = {"lending_club": (crc.LC_LISTING.index("addr_state"), crc.LC_LISTING.index("purpose")),
              # IEEE cheap columns: TransactionAmt, ProductCD(1), addr1, addr2, dist1, dist2, P_emaildomain(6),
              # R_emaildomain, card1..card6 (card6 = index 13, debit/credit)
              "ieee_fraud": (1, 6, 13)}
N_MIN = 200
PRIORITIES = ["P2", "P1", "TOPM", "RND"]
SIZE_BINS = [0, 500, 2000, 10_000, np.inf]
OUT = Path("paper/src_extension")


def predicate_queries(data, domain, seed):
    """Pool cal + test rows, define one query per predicate cell, split queries at random."""
    cols = PREDICATES[domain]
    pool = {key: np.concatenate([data["cal"][key], data["test"][key]]) for key in ["pc", "pe", "R", "C", "xc"]}
    cell = pd.Series(list(zip(*[pool["xc"][:, c] for c in cols])))
    codes, uniq = pd.factorize(cell)
    sizes = np.bincount(codes)
    keep = np.flatnonzero(sizes >= N_MIN)
    rows = np.isin(codes, keep)
    qid = codes[rows]
    rng = np.random.default_rng([seed, 59])
    cal_q = rng.permutation(keep)[: len(keep) // 2]
    split = {}
    for name, sel in [("cal", np.isin(qid, cal_q)), ("test", ~np.isin(qid, cal_q))]:
        split[name] = {key: v[rows][sel] for key, v in pool.items()}
        split[name]["qid"] = qid[sel]
    return split, len(keep), float(rows.mean())


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    pq, n_queries, coverage = predicate_queries(data, domain, seed)
    rows, size_rows = [], []
    for k_frac in crc.K_FRACTIONS:
        fit = crc.query_arrays(data["fit"], scale, k_frac, seed, stream=0)
        cal = crc.query_arrays(pq["cal"], scale, k_frac, seed, stream=1)
        test = crc.query_arrays(pq["test"], scale, k_frac, seed, stream=2)
        test_sizes = np.array([len(q["uc"]) for q in test])
        m = len(cal)
        for name in PRIORITIES:
            pooled = np.concatenate([q["pri"][name] for q in fit])
            grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
            Lt_cal = np.array([crc.envelope(crc.loss_curve(q, name, grid)[0]) for q in cal])
            curves = [crc.loss_curve(q, name, grid) for q in test]
            L_test = np.array([c[0] for c in curves])
            F_test = np.array([c[1] for c in curves])
            for eps in crc.EPSILONS:
                ok = np.flatnonzero(m / (m + 1) * Lt_cal.mean(axis=0) + 1 / (m + 1) <= eps)
                j = ok[0] if len(ok) else len(grid) - 1
                rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps, "priority": name,
                             "m_cal": m, "n_test_queries": len(test), "n_queries": n_queries, "row_coverage": coverage,
                             "test_miss": float(L_test[:, j].mean()), "test_call_fraction": float(F_test[:, j].mean()),
                             "rows_call_fraction": float((F_test[:, j] * test_sizes).sum() / test_sizes.sum())})
                bins = np.digitize(test_sizes, SIZE_BINS[1:-1])
                for bi in np.unique(bins):
                    sel = bins == bi
                    size_rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps,
                                      "priority": name, "size_bin": f"{SIZE_BINS[bi]}-{SIZE_BINS[bi + 1]}",
                                      "n": int(sel.sum()), "test_miss": float(L_test[sel, j].mean()),
                                      "test_call_fraction": float(F_test[sel, j].mean())})
    return rows, size_rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s ({n_queries} queries)"


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else list(PREDICATES)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    rows, size_rows = [], []
    with Pool(workers) as pool:
        for r, s, msg in pool.imap_unordered(task, [(d, s) for d in domains for s in range(n_seeds)]):
            rows += r
            size_rows += s
            print(msg, flush=True)
            pd.DataFrame(rows).to_csv(OUT / "selective_topk_predicate_results.csv", index=False)
            pd.DataFrame(size_rows).to_csv(OUT / "selective_topk_predicate_bysize.csv", index=False)
    t = pd.DataFrame(rows).assign(over=lambda x: (x.test_miss > x.eps).astype(int))
    summ = t.groupby(["domain", "k_fraction", "eps", "priority"]).agg(
        seeds=("seed", "nunique"), m_cal=("m_cal", "mean"), miss_mean=("test_miss", "mean"),
        miss_sd=("test_miss", "std"), seeds_over_eps=("over", "sum"), calls_mean=("test_call_fraction", "mean"),
        rows_calls_mean=("rows_call_fraction", "mean")).reset_index()
    summ.to_csv(OUT / "selective_topk_predicate_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

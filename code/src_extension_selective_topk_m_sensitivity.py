"""M1: how many calibration queries does the certified top-k need?

The calibration cost is m queries refined in full, once. One calibration
serves every (k, eps, order): the loss curves of the calibration queries are
computed once and lambda_hat for any eps is read off them. Here we ask how
the result changes when only m' of the m calibration queries are used.
For each (domain, seed, k, priority) the envelopes L~ of all calibration
queries and the test curves are computed once (as in
src_extension_selective_topk_crc.run); then, for m' in M_SUB, m' calibration
queries are drawn at random REPS times (without replacement) and lambda_hat
is recomputed with the CRC rule for m'. A subset of an exchangeable sequence
is still exchangeable with the test queries, so Theorem 1 holds for every m'
(if eps < 1/(m'+1) no grid value qualifies and lambda_hat = lambda_G: refine all).
Writes paper/src_extension/selective_topk_m_sensitivity_{results,summary}.csv.
Usage: py src_extension_selective_topk_m_sensitivity.py <n_seeds> [domains] [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc

M_SUB = [10, 20, 30, 45, 60, 90]   # plus "all"
REPS = 50
PRIORITIES = ["P2", "TOPM", "P1"]
OUT = Path("paper/src_extension")


def lam_index(Lt_sub, eps):
    m = Lt_sub.shape[0]
    ok = np.flatnonzero(m / (m + 1) * Lt_sub.mean(axis=0) + 1 / (m + 1) <= eps)
    return ok[0] if len(ok) else Lt_sub.shape[1] - 1


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    rows = []
    for k_frac in crc.K_FRACTIONS:
        fit = crc.query_arrays(data["fit"], scale, k_frac, seed, stream=0)
        cal = crc.query_arrays(data["cal"], scale, k_frac, seed, stream=1)
        test = crc.query_arrays(data["test"], scale, k_frac, seed, stream=2)
        m = len(cal)
        for name in PRIORITIES:
            pooled = np.concatenate([q["pri"][name] for q in fit])
            grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
            Lt_cal = np.array([crc.envelope(crc.loss_curve(q, name, grid)[0]) for q in cal])
            curves = [crc.loss_curve(q, name, grid) for q in test]
            L_test = np.array([c[0] for c in curves]).mean(axis=0)
            F_test = np.array([c[1] for c in curves]).mean(axis=0)
            rng = np.random.default_rng([seed, 31, int(100 * k_frac), PRIORITIES.index(name)])
            for m_sub in [s for s in M_SUB if s < m] + [m]:
                reps = 1 if m_sub == m else REPS
                for eps in crc.EPSILONS:
                    js = [lam_index(Lt_cal if m_sub == m else Lt_cal[rng.choice(m, m_sub, replace=False)], eps)
                          for _ in range(reps)]
                    miss, calls = L_test[js], F_test[js]
                    rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps, "priority": name,
                                 "m_cal": m, "m_sub": m_sub, "is_full": m_sub == m, "reps": reps,
                                 "test_miss": float(miss.mean()), "test_call_fraction": float(calls.mean()),
                                 "frac_all_refined": float(np.mean(np.array(js) == len(grid) - 1))})
    return rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else list(crc.DOMAINS)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    keep = crc.kept_rows(OUT / "selective_topk_m_sensitivity_results.csv", domains)
    rows = []
    with Pool(workers) as pool:
        for r, msg in pool.imap_unordered(task, [(d, s) for d in domains for s in range(n_seeds)]):
            rows += r
            print(msg, flush=True)
            pd.concat([keep, pd.DataFrame(rows)]).to_csv(OUT / "selective_topk_m_sensitivity_results.csv", index=False)
    t = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
    t["m_label"] = np.where(t.is_full, "all", t.m_sub.astype(str))
    t = t.assign(over=(t.test_miss > t.eps).astype(int))
    summ = t.groupby(["domain", "k_fraction", "eps", "priority", "m_label"]).agg(
        m_sub=("m_sub", "mean"), seeds=("seed", "nunique"), miss_mean=("test_miss", "mean"),
        miss_sd=("test_miss", "std"), calls_mean=("test_call_fraction", "mean"),
        frac_all_refined=("frac_all_refined", "mean"), seeds_over_eps=("over", "sum")).reset_index()
    summ.to_csv(OUT / "selective_topk_m_sensitivity_summary.csv", index=False)
    print(summ[(summ.priority == "P2") & (summ.k_fraction == 0.10)].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

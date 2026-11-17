"""A2: high-probability variant of the certified top-k rule (Learn then Test).

CRC (src_extension_selective_topk_crc.py) bounds the expected miss over the
draw of calibration and test queries. Learn then Test (Angelopoulos, Bates,
Candes, Jordan, Lei, Ann. Appl. Stat. 2025; arXiv 2110.01052) instead gives
  P( risk(lambda_hat) > eps ) <= delta,
where risk(lambda) = E_Q[L~_Q(lambda)] over a new query, and the probability is
over the calibration queries. For each grid value we test H_lambda: risk > eps
with the Hoeffding-Bentkus p-value on the m calibration envelopes L~ in [0, 1],
and go down the grid from lambda_G (all refined, risk 0) by fixed-sequence
testing while p <= delta; lambda_hat is the last value rejected. Fixed-sequence
testing controls the family-wise error, so the guarantee holds; the envelope
makes risk monotone, which is what makes the order natural (not needed for
validity).

Same domains, splits, priorities and grid as the CRC script. Records the test
risk (mean over test queries) and call fraction, and whether test risk > eps,
which should happen in at most about delta of the seeds (up to the noise of
estimating the risk on the test queries).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

import src_extension_selective_topk_crc as crc

DELTA = 0.10


def hb_pvalue(rhat, eps, m, bentkus=True):
    """Hoeffding-Bentkus p-value for H: risk > eps. With bentkus=False only the
    Hoeffding (Chernoff/KL) term is used; that term is also valid for sampling
    without replacement from a finite pool (Hoeffding, JASA 58, 1963), which is how
    our random splits draw the calibration queries."""
    if rhat >= eps:
        return 1.0
    a = max(rhat, 1e-12)
    h = a * np.log(a / eps) + (1 - a) * np.log((1 - a) / (1 - eps))
    p = float(np.exp(-m * h))
    return min(p, float(np.e * binom.cdf(np.ceil(m * rhat), m, eps))) if bentkus else p


def ltt_index(Lt_cal, eps, delta, bentkus=True):
    m = Lt_cal.shape[0]
    rhat = Lt_cal.mean(axis=0)
    j = Lt_cal.shape[1] - 1
    while j > 0 and hb_pvalue(rhat[j - 1], eps, m, bentkus) <= delta:
        j -= 1
    return j


def run(domain, seed):
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    rows = []
    for k_frac in crc.K_FRACTIONS:
        fit = crc.query_arrays(data["fit"], scale, k_frac, seed, stream=0)
        cal = crc.query_arrays(data["cal"], scale, k_frac, seed, stream=1)
        test = crc.query_arrays(data["test"], scale, k_frac, seed, stream=2)
        for name in crc.PRIORITIES:
            pooled = np.concatenate([q["pri"][name] for q in fit])
            grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
            Lt_cal = np.array([crc.envelope(crc.loss_curve(q, name, grid)[0]) for q in cal])
            curves = [crc.loss_curve(q, name, grid) for q in test]
            L_test = np.array([c[0] for c in curves])
            F_test = np.array([c[1] for c in curves])
            for eps in crc.EPSILONS:
                for pval, bentkus in [("HB", True), ("H", False)]:
                    j = ltt_index(Lt_cal, eps, DELTA, bentkus)
                    rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps, "delta": DELTA,
                                 "pvalue": pval, "priority": name, "m_cal": len(cal), "lambda_index": int(j),
                                 "test_miss": float(L_test[:, j].mean()),
                                 "test_call_fraction": float(F_test[:, j].mean())})
    return rows


def main():
    seeds = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else list(crc.DOMAINS)
    rows = []
    for domain in domains:
        for seed in seeds:
            rows += run(domain, seed)
            print(f"[{domain}] seed {seed} done", flush=True)
    out = Path("paper/src_extension")
    df = pd.concat([crc.kept_rows(out / "selective_topk_ltt_results.csv", domains), pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(out / "selective_topk_ltt_results.csv", index=False)
    df = df.assign(over=(df["test_miss"] > df["eps"]).astype(int))
    summ = df.groupby(["domain", "k_fraction", "eps", "pvalue", "priority"]).agg(
        miss_mean=("test_miss", "mean"), calls_mean=("test_call_fraction", "mean"),
        seeds=("seed", "count"), seeds_over_eps=("over", "sum")).reset_index()
    summ.to_csv(out / "selective_topk_ltt_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Seed-blocked comparison of P5 and P3c certified calls (one value per seed, not per setting).

Per seed, the relative difference P5/P3c - 1 of the CRC test call fraction is averaged over the six (kappa, eps)
settings of that seed, because the settings of one seed share splits and models and are not independent. Reports the
number of seeds with a reduction, the mean with a t interval and a bootstrap interval over seeds, the seed-level
Wilcoxon signed-rank and sign tests, and per setting the mean and the seeds with a reduction.
Usage: py src_vldb_p5_seedblock.py [domain]   (cwd: the folder containing paper/src_extension)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path("paper/src_extension/selective_topk_p4_results.csv")


def main():
    dom = sys.argv[1] if len(sys.argv) > 1 else "soccer"
    r = pd.read_csv(R)
    s = r[(r.domain == dom) & r.priority.isin(["P3CLIP", "P5"])].pivot_table(
        index=["seed", "k_fraction", "eps"], columns="priority", values="test_call_fraction")
    s["rel"] = s.P5 / s.P3CLIP - 1
    per = s.groupby("seed").rel.mean()
    h = stats.t.ppf(0.975, len(per) - 1) * per.std(ddof=1) / np.sqrt(len(per))
    rng = np.random.default_rng(0)
    boot = [per.to_numpy()[rng.integers(0, len(per), len(per))].mean() for _ in range(10000)]
    print(f"{dom}: settings with a reduction {int((s.rel < 0).sum())}/{len(s)}; seeds {int((per < 0).sum())}/{len(per)}")
    print(f"seed-level mean {per.mean():.4f}, t 95% [{per.mean() - h:.4f}, {per.mean() + h:.4f}], "
          f"bootstrap 95% [{np.quantile(boot, .025):.4f}, {np.quantile(boot, .975):.4f}]")
    print(f"Wilcoxon p = {stats.wilcoxon(per).pvalue:.2g}, sign test p = "
          f"{stats.binomtest(int((per < 0).sum()), len(per)).pvalue:.2g}")
    for (k, e), g in s.groupby(["k_fraction", "eps"]):
        print(f"  kappa {k}, eps {e}: mean {g.rel.mean():.3f}, seeds lower {int((g.rel < 0).sum())}/{len(g)}")


if __name__ == "__main__":
    main()

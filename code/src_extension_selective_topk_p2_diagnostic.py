"""M6: how well does the location-scale error model behind Proposition 1 hold?

Proposition 1 assumes P_e - P_c = s(x) Z with Z independent of the cheap
information and symmetric. Then the probability that candidate i crosses the
cheap cutoff is G(P2_i) with G(t) = P(Z >= t), so the P2 order is the order of
crossing probability. Per domain (seed 0) this script reports
  sym      mean of Z on the fit split (0 for a symmetric Z)
  disp     spread of the |Z| median across deciles of s_hat (max / min); 1 means
           the scale is fully captured by s_hat
  calib    crossing probability predicted from the fit split's Z versus the
           crossing rate observed on the test split, in 10 bins of P2
           (mean absolute gap)
  rank     Spearman correlation between P2 and the observed crossing indicator
           (more negative = P2 orders crossings better)
Writes paper/src_extension/selective_topk_p2_diagnostic.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_diagnostic import utility

SEED, K_FRAC = 0, 0.10


def crossing_and_p2(d, s_hat):
    """Per candidate: P2 and whether U_e lies on the other side of theta_c (inclusive)."""
    U_c, U_e = utility(d["pc"], d["R"], d["C"]), utility(d["pe"], d["R"], d["C"])
    p2, cross = np.empty(len(U_c)), np.zeros(len(U_c), dtype=bool)
    for q in np.unique(d["qid"]):
        ix = np.flatnonzero(d["qid"] == q)
        k = max(1, int(round(K_FRAC * len(ix))))
        if k >= len(ix):
            p2[ix] = np.nan
            continue
        part = -np.partition(-U_c[ix], [k - 1, k])
        theta = 0.5 * (part[k - 1] + part[k])
        budget = d["R"][ix] + d["C"][ix]
        p2[ix] = np.abs(U_c[ix] - theta) / (budget * s_hat[ix])
        cross[ix] = (U_e[ix] - theta) * np.sign(theta - U_c[ix]) >= 0
    return p2, cross


def half(d, mask):
    return {key: v[mask] for key, v in d.items()}


def main():
    rows = []
    for name in crc.DOMAINS:
        data = crc.DOMAINS[name](SEED)
        fit, test = data["fit"], data["test"]
        # Cross-fitting: s_hat is fit on one half of the fit queries and Z is computed on the other
        # half, so the residuals are out of sample for s_hat (in-sample Z would look too well scaled).
        uq = np.unique(fit["qid"])
        a = np.isin(fit["qid"], np.random.default_rng(SEED + 41).permutation(uq)[: len(uq) // 2])
        scale = crc.fit_scale(half(fit, a))
        fb = half(fit, ~a)
        s_b = scale(fb)
        z = (fb["pe"] - fb["pc"]) / s_b
        dec = pd.qcut(s_b, 10, labels=False, duplicates="drop")
        med = pd.Series(np.abs(z)).groupby(dec).median()
        p2, cross = crossing_and_p2(test, scale(test))
        ok = ~np.isnan(p2)
        p2, cross = p2[ok], cross[ok]
        # predicted crossing probability under the symmetric model: P(Z >= t) = P(|Z| >= t) / 2 for t > 0,
        # and 1 for a candidate tied with the cutoff (t = 0, the condition 0 >= 0 always holds)
        az = np.sort(np.abs(z))
        pred = 0.5 * (1 - np.searchsorted(az, p2, side="left") / len(az))
        pred[p2 == 0] = 1.0
        bins = pd.qcut(p2, 10, labels=False, duplicates="drop")
        cal = pd.DataFrame({"b": bins, "pred": pred, "obs": cross.astype(float)}).groupby("b").mean()
        mae = float((cal.pred - cal.obs).abs().mean())
        rows.append({"domain": name, "sym_mean_z": float(z.mean()), "sym_median_z": float(np.median(z)),
                     "disp_ratio": float(med.max() / max(med.min(), 1e-12)),
                     "calib_mae": mae, "cross_rate": float(cross.mean()),
                     "calib_mae_relative": mae / float(cross.mean()),
                     "spearman_p2_cross": float(spearmanr(p2, cross).correlation)})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(Path("paper/src_extension/selective_topk_p2_diagnostic.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

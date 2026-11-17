"""Time-ordered validation on IEEE-CIS.

Instead of a random split, transactions are ordered by TransactionDT: the first 50% train the cheap and expensive
models, the next 10% are the fit split (s_hat, grid), the next 20% are calibration queries and the last 20% (the most
recent ~5 weeks) are test queries; queries are consecutive blocks of 1,000 transactions. Calibration and test are
therefore NOT exchangeable; the run shows how well a calibration made on the past transfers to later queries. We
report the test mis-selection rate and calls of P2 and P3c with CRC, overall and per quarter of the test period.
Writes paper/src_extension/selective_topk_temporal_ieee.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_baselines import topk_mask
from src_extension_selective_topk_p3_extra import iid_sampler

OUT = Path("paper/src_extension/selective_topk_temporal_ieee.csv")
ROWS = 1000
B = 200


def main():
    if crc._IEEE is None:
        crc.build_ieee_fraud(0)                      # loads and encodes the table once
    df, _, Xc, Xf = crc._IEEE
    dt = pd.read_parquet(crc.IEEE_CLEAN, columns=["TransactionDT"])["TransactionDT"].to_numpy()
    order = np.argsort(dt, kind="stable")
    n = len(order)
    cuts = np.cumsum([int(round(crc.FRACS[s] * n)) for s in ["train", "fit", "cal"]])
    parts = dict(zip(["train", "fit", "cal", "test"], np.split(order, cuts)))
    y = pd.Series(df["isFraud"].to_numpy())
    models = {key: crc.dia.train_lgb(X.iloc[parts["train"]], y.iloc[parts["train"]], X.iloc[parts["fit"]],
                                     y.iloc[parts["fit"]]) for key, X in {"cheap": Xc, "exp": Xf}.items()}
    col = {key: list(mod.classes_).index(1) for key, mod in models.items()}
    R_all = df["TransactionAmt"].to_numpy(dtype=float)
    data = {}
    for s, rows in parts.items():
        if s == "train":
            continue
        data[s] = {"pc": models["cheap"].predict_proba(Xc.iloc[rows])[:, col["cheap"]],
                   "pe": models["exp"].predict_proba(Xf.iloc[rows])[:, col["exp"]],
                   "R": R_all[rows], "C": np.full(len(rows), crc.IEEE_REVIEW_COST),
                   "xc": Xc.iloc[rows].to_numpy(dtype=float), "qid": np.arange(len(rows)) // ROWS}
        # merge a short last block into the previous one, as batch_queries does
        q = data[s]["qid"]
        if (q == q.max()).sum() < ROWS and q.max() > 0:
            q[q == q.max()] = q.max() - 1
    scale = crc.fit_scale(data["fit"])
    f = data["fit"]
    z_all = (f["pe"] - f["pc"]) / scale(f)
    out = []
    for k_frac in crc.K_FRACTIONS:
        sets = {}
        for s, stream in (("fit", 0), ("cal", 1), ("test", 2)):
            qs = crc.query_arrays(data[s], scale, k_frac, 0, stream=stream)
            d = data[s]
            s_all = scale(d)
            rng = np.random.default_rng([0, stream, 499])
            qids = [qq for qq in np.unique(d["qid"]) if max(1, int(round(k_frac * (d["qid"] == qq).sum()))) < (d["qid"] == qq).sum()]
            for q, qq in zip(qs, qids):
                ix = np.flatnonzero(d["qid"] == qq)
                k = q["k"]
                p_sim = d["pc"][ix][None, :] + s_all[ix][None, :] * rng.choice(z_all, size=(B, len(ix)))
                u_sim = np.clip(p_sim, 0, 1) * (d["R"][ix] + d["C"][ix])[None, :] - d["C"][ix][None, :]
                kth = -np.partition(-u_sim, k - 1, axis=1)[:, k - 1:k]
                flip = np.where(topk_mask(q["uc"], k)[None, :], u_sim < kth, u_sim >= kth).mean(axis=0)
                p2 = q["pri"]["P2"]
                q["pri"]["P3c"] = 1.0 - flip + 1e-6 * p2 / (p2.max() + 1e-12)
            sets[s] = qs
        for name in ("P2", "P3c"):
            pooled = np.concatenate([q["pri"][name] for q in sets["fit"]])
            grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
            env = np.array([crc.envelope(crc.loss_curve(q, name, grid)[0]) for q in sets["cal"]])
            curves = [crc.loss_curve(q, name, grid) for q in sets["test"]]
            L = np.array([c[0] for c in curves])
            F = np.array([c[1] for c in curves])
            m = len(env)
            quarter = np.minimum(np.arange(len(L)) * 4 // len(L), 3)
            for eps in crc.EPSILONS:
                ok = np.flatnonzero(m / (m + 1) * env.mean(axis=0) + 1 / (m + 1) <= eps)
                j = ok[0] if len(ok) else len(grid) - 1
                r = {"k_fraction": k_frac, "eps": eps, "priority": name, "m_cal": m, "n_test": len(L),
                     "test_miss": L[:, j].mean(), "test_calls": F[:, j].mean(),
                     "cal_env_at_lambda": env[:, j].mean()}
                for qt in range(4):
                    r[f"miss_q{qt + 1}"] = L[quarter == qt, j].mean()
                out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(OUT, index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

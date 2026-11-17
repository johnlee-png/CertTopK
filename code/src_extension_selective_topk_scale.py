"""C2/C3: operator overhead and scalability on Lending Club (1.35M loans).

Times, for n rows (a growing prefix of a random permutation of all loans, cut
into 5,000-row queries as in the CRC experiments):
  cheap        cheap-model inference on all n rows
  scale        error-scale model s on all n rows
  operator     everything the certified top-k operator adds: U_c, the per-query
               cut-off theta_c, priority P2, the selection {P2 <= lambda}, and
               the final top-k of the mixed scores
  expensive_selected  expensive-model inference on the selected rows only
  expensive_all       expensive-model inference on all n rows (the baseline)
lambda is the CRC value calibrated for seed 0, k = 10%, eps = 0.05. Models are
trained as in build_lending_club (seed 0 split). The expensive model here costs
only a bigger LightGBM; in practice its input is a paid bureau pull, so
expensive_* is a lower bound on the real saving and the point of the experiment
is that the operator's own overhead stays small and linear in n.
Writes paper/src_extension/selective_topk_scale.csv.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
import diabetes_new_regret_diagnostic as dia
from src_extension_selective_topk_diagnostic import utility

SIZES = [100_000, 250_000, 500_000, 1_000_000, None]   # None = all rows
REPEATS = 3
K_FRAC, EPS, SEED = 0.10, 0.05, 0


def timed(f, repeats=REPEATS):
    best, out = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    crc.build_lending_club(SEED)            # fills the encoded tables
    df, Xc, Xf = crc._LC
    groups = df["id"].astype(str).to_numpy(dtype=object)
    m = crc.group_split(groups, SEED)
    y = pd.Series(df["repaid"].to_numpy())
    cheap = dia.train_lgb(Xc[m["train"]], y[m["train"]], Xc[m["fit"]], y[m["fit"]])
    exp = dia.train_lgb(Xf[m["train"]], y[m["train"]], Xf[m["fit"]], y[m["fit"]])
    data = crc.DOMAINS["lending_club"](SEED)   # cached predictions for calibration
    scale = crc.fit_scale(data["fit"])
    # lambda_hat for P2 exactly as in the CRC run
    fitq = crc.query_arrays(data["fit"], scale, K_FRAC, SEED)
    calq = crc.query_arrays(data["cal"], scale, K_FRAC, SEED)
    pooled = np.concatenate([q["pri"]["P2"] for q in fitq])
    grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
    Lt = np.array([crc.envelope(crc.loss_curve(q, "P2", grid)[0]) for q in calq])
    mc = len(calq)
    lam = grid[np.flatnonzero(mc / (mc + 1) * Lt.mean(axis=0) + 1 / (mc + 1) <= EPS)[0]]

    amt = df["loan_amnt"].to_numpy(dtype=float)
    off = m["train"] & (df["repaid"].to_numpy() == 0)
    lgd = float(np.clip((amt - df["total_rec_prncp"].to_numpy() - df["recoveries"].to_numpy())[off] / amt[off], 0, 1).mean())
    R_all = np.maximum(df["installment"].to_numpy() * df["term"].to_numpy() - amt, 0.0)
    C_all = lgd * amt
    perm = np.random.default_rng(1).permutation(len(df))
    rows = []
    for size in SIZES:
        ix = np.sort(perm[: size or len(df)])
        n = len(ix)
        xc, xf, R, C = Xc.iloc[ix], Xf.iloc[ix], R_all[ix], C_all[ix]
        qid = np.arange(n) // 5000
        t_cheap, pc = timed(lambda: cheap.predict_proba(xc)[:, 1])
        d = {"xc": xc.to_numpy(dtype=float), "pc": pc, "R": R, "C": C}
        t_scale, s = timed(lambda: scale(d))

        # Rows of each query as contiguous slices of one sort (qid is already sorted here,
        # so this is O(n)); every per-query step below touches only its own slice.
        order = np.argsort(qid, kind="stable")
        starts = np.flatnonzero(np.r_[True, np.diff(qid[order]) != 0])
        bounds = list(zip(starts, np.r_[starts[1:], n]))

        def operator():
            uc = utility(pc, R, C)
            theta = np.empty(n)
            for a, b in bounds:
                q = order[a:b]
                k = max(1, int(round(K_FRAC * len(q))))
                part = -np.partition(-uc[q], [k - 1, k])
                theta[q] = 0.5 * (part[k - 1] + part[k])
            p2 = np.abs(uc - theta) / ((R + C) * s)
            return uc, p2 <= lam
        t_op, (uc, sel) = timed(operator)
        t_sel, pe_sel = timed(lambda: exp.predict_proba(xf[sel])[:, 1])
        t_all, _ = timed(lambda: exp.predict_proba(xf)[:, 1], repeats=1 if n > 500_000 else REPEATS)

        def final_topk():
            mixed = uc.copy()
            mixed[sel] = utility(pe_sel, R[sel], C[sel])
            out = []
            for a, b in bounds:
                j = order[a:b]
                k = max(1, int(round(K_FRAC * len(j))))
                out.append(j[crc.topk(mixed[j], k)])
            return out
        t_topk, _ = timed(final_topk)
        rows.append({"n": n, "selected_fraction": float(sel.mean()), "cheap_s": t_cheap, "scale_s": t_scale,
                     "operator_s": t_op + t_topk, "expensive_selected_s": t_sel, "expensive_all_s": t_all,
                     "certified_total_s": t_cheap + t_scale + t_op + t_topk + t_sel})
        r = rows[-1]
        print(f"n={n:>9,}  selected {100 * r['selected_fraction']:.1f}%  cheap {t_cheap:.2f}s  scale {t_scale:.2f}s  "
              f"operator {r['operator_s']:.2f}s  exp(selected) {t_sel:.2f}s  exp(all) {t_all:.2f}s  "
              f"certified total {r['certified_total_s']:.2f}s", flush=True)
    pd.DataFrame(rows).to_csv(Path("paper/src_extension/selective_topk_scale.csv"), index=False)


if __name__ == "__main__":
    main()

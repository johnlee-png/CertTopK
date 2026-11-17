"""Query-adaptive certified stopping (P2A).

The static rule refines {i : P2_i <= lambda} with one global lambda. Queries differ in difficulty, so P2A learns,
from the fit split only, a query-level scale m(Q) and refines {i : P2_i <= lambda * m(Q)}, i.e. it uses the priority
P2_i / m(Q). For every query the refined sets are nested in lambda and depend only on Q and the training/fit data D0,
so conditions (i)-(iii) and Theorem 1 hold unchanged; CRC then calibrates the single multiplier lambda.

m(Q) is a ridge regression, fitted on the fit queries, of log t0(Q) on cheap query features, where t0(Q) is the
smallest P2 threshold at which the query's envelope reaches zero (the answer is exact). Features use cheap information
only: quantiles of log P2 in the query, k / n, the spread of the budget R + C, and the mean predicted error size.

Writes paper/src_extension/selective_topk_adaptive_{results,bydifficulty,summary}.csv.
Usage: py src_extension_selective_topk_adaptive.py <n_seeds> [domains] [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import src_extension_selective_topk_crc as crc

OUT = Path("paper/src_extension")
EPS_P = 1e-9


def query_features(q, budget, s):
    lp = np.log(q["pri"]["P2"] + EPS_P)
    qs = np.quantile(lp, [0.01, 0.05, 0.10, 0.25, 0.50])
    return np.concatenate([qs, [q["k"] / len(lp), np.log(len(lp)), np.std(np.log(budget)), np.mean(np.log(s))]])


def exact_threshold(q):
    """Smallest P2 threshold whose envelope loss is 0 for this query."""
    grid = np.unique(np.concatenate([[-np.inf], np.sort(q["pri"]["P2"]), [np.inf]]))
    lt = crc.envelope(crc.loss_curve(q, "P2", grid)[0])
    j = int(np.flatnonzero(lt <= 0)[0])
    return grid[j]


def arrays_with_meta(d, scale, k_frac, seed, stream):
    """query_arrays plus each query's budget and predicted error size (same query order)."""
    qs = crc.query_arrays(d, scale, k_frac, seed, stream=stream)
    s_all = scale(d)
    budget_all = d["R"] + d["C"]
    metas = []
    for qid in np.unique(d["qid"]):
        ix = np.flatnonzero(d["qid"] == qid)
        k = max(1, int(round(k_frac * len(ix))))
        if k >= len(ix):
            continue
        metas.append((budget_all[ix], s_all[ix]))
    assert len(metas) == len(qs)
    return qs, metas


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    rows, diff_rows = [], []
    for k_frac in crc.K_FRACTIONS:
        fit, fm = arrays_with_meta(data["fit"], scale, k_frac, seed, 0)
        cal, cm = arrays_with_meta(data["cal"], scale, k_frac, seed, 1)
        test, tm = arrays_with_meta(data["test"], scale, k_frac, seed, 2)
        X_fit = np.array([query_features(q, b, s) for q, (b, s) in zip(fit, fm)])
        t_fit = np.array([exact_threshold(q) for q in fit])
        finite = np.isfinite(t_fit) & (t_fit > 0)
        y_fit = np.log(np.where(finite, t_fit, np.nan))
        ok = np.isfinite(y_fit)
        if ok.sum() >= 5:
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(X_fit[ok], y_fit[ok])
            predict = lambda X: np.exp(model.predict(X))
        else:                      # too few fit queries: fall back to the static rule
            predict = lambda X: np.ones(len(X))
        for qs, ms in ((fit, fm), (cal, cm), (test, tm)):
            X = np.array([query_features(q, b, s) for q, (b, s) in zip(qs, ms)])
            m = predict(X)
            for q, mq in zip(qs, m):
                q["pri"]["P2A"] = q["pri"]["P2"] / mq
                q["m"] = float(mq)
        m_test = np.array([q["m"] for q in test])
        terc = np.quantile(m_test, [1 / 3, 2 / 3]) if len(m_test) >= 3 else [np.inf, np.inf]
        for name in ("P2", "P2A"):
            pooled = np.concatenate([q["pri"][name] for q in fit])
            grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
            lt_cal = np.array([crc.envelope(crc.loss_curve(q, name, grid)[0]) for q in cal])
            curves = [crc.loss_curve(q, name, grid) for q in test]
            l_test = np.array([c[0] for c in curves])
            f_test = np.array([c[1] for c in curves])
            mcal = len(cal)
            for eps in crc.EPSILONS:
                okj = np.flatnonzero(mcal / (mcal + 1) * lt_cal.mean(axis=0) + 1 / (mcal + 1) <= eps)
                j = okj[0] if len(okj) else len(grid) - 1
                rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps, "priority": name,
                             "m_cal": mcal, "n_fit_queries": len(fit), "test_miss": float(l_test[:, j].mean()),
                             "test_call_fraction": float(f_test[:, j].mean())})
                # difficulty terciles by the predicted scale m(Q) (cheap information only)
                band = np.digitize(m_test, terc)
                for b, lab in enumerate(["easy", "medium", "hard"]):
                    sel = band == b
                    if sel.any():
                        diff_rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps,
                                          "priority": name, "difficulty": lab, "n": int(sel.sum()),
                                          "miss": float(l_test[sel, j].mean()), "calls": float(f_test[sel, j].mean())})
    return rows, diff_rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def summarize():
    t = pd.read_csv(OUT / "selective_topk_adaptive_results.csv").assign(over=lambda x: (x.test_miss > x.eps).astype(int))
    summ = t.groupby(["domain", "k_fraction", "eps", "priority"]).agg(
        seeds=("seed", "nunique"), miss_mean=("test_miss", "mean"), seeds_over_eps=("over", "sum"),
        calls_mean=("test_call_fraction", "mean")).reset_index()
    piv = summ.pivot_table(index=["domain", "k_fraction", "eps"], columns="priority", values="calls_mean")
    piv["ratio_P2A_to_P2"] = piv["P2A"] / piv["P2"]
    summ.to_csv(OUT / "selective_topk_adaptive_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))
    print(piv.round(4).to_string())
    print("cells", len(summ), "mean<=eps", int((summ.miss_mean <= summ.eps).sum()))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        return summarize()
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else list(crc.DOMAINS)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    path, dpath = OUT / "selective_topk_adaptive_results.csv", OUT / "selective_topk_adaptive_bydifficulty.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    drows = pd.read_csv(dpath).to_dict("records") if dpath.exists() else []
    done = {(r["domain"], r["seed"]) for r in rows}
    todo = [(d, s) for d in domains for s in range(n_seeds) if (d, s) not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    with Pool(workers) as pool:
        for r, dr, msg in pool.imap_unordered(task, todo):
            rows += r
            drows += dr
            print(msg, flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
            pd.DataFrame(drows).to_csv(dpath, index=False)
    summarize()


if __name__ == "__main__":
    main()

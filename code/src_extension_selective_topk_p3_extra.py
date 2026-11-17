"""Follow-up experiments on the loss-aware priority P3.

For each (domain, seed), with the splits, models and queries of src_extension_selective_topk_crc.py:
  (1) equal-call curve of P3 on the test queries (same F grid and miss as src_extension_selective_topk_pareto.py);
  (2) wall-clock cost of computing P3 (B = 200) on the test queries;
  (3) residual-dependence ablation: P3BLK draws, for every scenario, one fit query at random and resamples all
      candidates' Z from that query's residuals (query-block bootstrap; candidates in a scenario share the query's
      error distribution), instead of resampling Z i.i.d. from the pooled fit residuals;
  (4) number of scenarios: P3 with B in {50, 100, 200, 400};
  (5) cross-fitted order selection: the fit-split plug-in calls of P3 recomputed with 2-fold cross-fitting over fit
      queries (P3 of a fit query uses only residuals of the other fold), so the choice no longer sees P3 scored with
      its own residuals. The deployed P3 on calibration / test queries is unchanged.
Every variant is a function of Q, D0 and independent randomness, so the CRC guarantee applies to each.

Writes paper/src_extension/selective_topk_p3extra_{results,curves,timing,fitcv}.csv.
Usage: py src_extension_selective_topk_p3_extra.py <n_seeds> [domains] [workers] | summarize
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_order_select import plug_in
from src_extension_selective_topk_pareto import F_GRID, miss_of

OUT = Path("paper/src_extension")
VARIANTS = {"P3": (200, "iid"), "P3B50": (50, "iid"), "P3B100": (100, "iid"), "P3B400": (400, "iid"),
            "P3BLK": (200, "block")}


def p3_priority(uc, width, k, p2, B, sampler, rng):
    """1 - P(membership differs from the cheap top-k) under the simulated errors; ties broken by P2."""
    n = len(uc)
    ue_sim = uc[None, :] + width[None, :] * sampler(rng, B, n)
    kth = -np.partition(-ue_sim, k - 1, axis=1)[:, k - 1:k]
    in_sim = ue_sim >= kth
    in_cheap = np.zeros(n, bool)
    in_cheap[crc.topk(uc, k)] = True
    flip = np.where(in_cheap[None, :], ~in_sim, in_sim).mean(axis=0)
    return 1.0 - flip + 1e-6 * p2 / (p2.max() + 1e-12)


def iid_sampler(pool):
    return lambda rng, B, n: rng.choice(pool, size=(B, n))


def block_sampler(groups):
    def s(rng, B, n):
        js = rng.integers(len(groups), size=B)
        return np.stack([rng.choice(groups[j], size=n) for j in js])
    return s


def query_index(d, k_frac):
    """Row indices of the queries kept by crc.query_arrays, in the same order."""
    out = []
    for qid in np.unique(d["qid"]):
        ix = np.flatnonzero(d["qid"] == qid)
        k = max(1, int(round(k_frac * len(ix))))
        if k < len(ix):
            out.append(ix)
    return out


def arrays(d, scale, k_frac, seed, stream):
    qs = crc.query_arrays(d, scale, k_frac, seed, stream=stream)
    for q, ix in zip(qs, query_index(d, k_frac)):
        assert len(ix) == len(q["uc"])
        q["_ix"] = ix
    return qs


def crc_calls(sets, name):
    pooled = np.concatenate([q["pri"][name] for q in sets["fit"]])
    grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
    lt_cal = np.array([crc.envelope(crc.loss_curve(q, name, grid)[0]) for q in sets["cal"]])
    curves = [crc.loss_curve(q, name, grid) for q in sets["test"]]
    l_test = np.array([c[0] for c in curves])
    f_test = np.array([c[1] for c in curves])
    m = len(sets["cal"])
    out = {}
    for eps in crc.EPSILONS:
        ok = np.flatnonzero(m / (m + 1) * lt_cal.mean(axis=0) + 1 / (m + 1) <= eps)
        j = ok[0] if len(ok) else len(grid) - 1
        out[eps] = (float(l_test[:, j].mean()), float(f_test[:, j].mean()), m)
    return out


def fit_plugin_calls(fit_qs, name):
    pooled = np.concatenate([q["pri"][name] for q in fit_qs])
    grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
    cs = [crc.loss_curve(q, name, grid) for q in fit_qs]
    le, fc = np.array([crc.envelope(c[0]) for c in cs]), np.array([c[1] for c in cs])
    return {eps: plug_in(le, fc, eps) for eps in crc.EPSILONS}


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    f = data["fit"]
    z_all = (f["pe"] - f["pc"]) / scale(f)
    groups = [z_all[f["qid"] == q] for q in np.unique(f["qid"])]
    samplers = {"iid": iid_sampler(z_all), "block": block_sampler(groups)}
    rows, curves, timing, fitcv = [], [], [], []
    for k_frac in crc.K_FRACTIONS:
        sets = {split: arrays(data[split], scale, k_frac, seed, stream)
                for split, stream in (("fit", 0), ("cal", 1), ("test", 2))}
        for vi, (name, (B, kind)) in enumerate(VARIANTS.items()):
            for si, (split, stream) in enumerate((("fit", 0), ("cal", 1), ("test", 2))):
                rng = np.random.default_rng([seed, stream, 99] if name == "P3" else [seed, stream, 199, vi])
                d = data[split]
                width_all = (d["R"] + d["C"]) * scale(d)
                tic = time.perf_counter()
                for q in sets[split]:
                    q["pri"][name] = p3_priority(q["uc"], width_all[q["_ix"]], q["k"], q["pri"]["P2"], B,
                                                 samplers[kind], rng)
                sec = time.perf_counter() - tic
                if split == "test":
                    n_cand = sum(len(q["uc"]) for q in sets["test"])
                    timing.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "variant": name, "B": B,
                                   "seconds": sec, "n_queries": len(sets["test"]), "n_candidates": n_cand,
                                   "us_per_candidate": 1e6 * sec / n_cand})
            for eps, (miss, calls, m) in crc_calls(sets, name).items():
                rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps, "priority": name,
                             "m_cal": m, "test_miss": miss, "test_call_fraction": calls})
        # (1) equal-call curve of P3 on the test queries
        acc = np.zeros(len(F_GRID))
        for q in sets["test"]:
            n, k = len(q["uc"]), q["k"]
            ref = set(crc.topk(q["ue"], k).tolist())
            order = np.argsort(q["pri"]["P3"], kind="stable")
            for j, m in enumerate(np.ceil(F_GRID * n).astype(int)):
                mixed = q["uc"].copy()
                mixed[order[:m]] = q["ue"][order[:m]]
                acc[j] += miss_of(crc.topk(mixed, k), ref, k)
        for fv, v in zip(F_GRID, acc / len(sets["test"])):
            curves.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "method": "P3",
                           "call_fraction": fv, "miss": v})
        # (5) cross-fitted fit-split plug-in calls of P3
        fqid = np.unique(f["qid"])
        fold_of_q = dict(zip(fqid, np.random.default_rng([seed, 299]).permutation(len(fqid)) % 2))
        row_fold = np.array([fold_of_q[q] for q in f["qid"]])
        pools = [z_all[row_fold == 1], z_all[row_fold == 0]]   # pool used for queries of fold 0 / fold 1
        rng = np.random.default_rng([seed, 0, 399])
        width_all = (f["R"] + f["C"]) * scale(f)
        for q in sets["fit"]:
            fold = fold_of_q[f["qid"][q["_ix"][0]]]
            q["pri"]["P3CF"] = p3_priority(q["uc"], width_all[q["_ix"]], q["k"], q["pri"]["P2"], 200,
                                           iid_sampler(pools[fold]), rng)
        cf = fit_plugin_calls(sets["fit"], "P3CF")
        for eps in crc.EPSILONS:
            fitcv.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps, "fit_calls_P3": cf[eps]})
    return rows, curves, timing, fitcv, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def summarize():
    t = pd.read_csv(OUT / "selective_topk_p3extra_results.csv").assign(over=lambda x: (x.test_miss > x.eps).astype(int))
    summ = t.groupby(["domain", "k_fraction", "eps", "priority"]).agg(
        seeds=("seed", "nunique"), miss_mean=("test_miss", "mean"), seeds_over_eps=("over", "sum"),
        calls_mean=("test_call_fraction", "mean")).reset_index()
    summ.to_csv(OUT / "selective_topk_p3extra_summary.csv", index=False)
    piv = summ.pivot_table(index=["domain", "k_fraction", "eps"], columns="priority", values="calls_mean")
    print((100 * piv).round(1).to_string())
    print("cells", len(summ), "mean<=eps", int((summ.miss_mean <= summ.eps).sum()),
          "max miss/eps", round((summ.miss_mean / summ.eps).max(), 3))
    tm = pd.read_csv(OUT / "selective_topk_p3extra_timing.csv")
    print(tm.groupby(["domain", "variant"]).us_per_candidate.median().unstack().round(1).to_string())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        return summarize()
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else list(crc.DOMAINS)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    names = ["results", "curves", "timing", "fitcv"]
    paths = {n: OUT / f"selective_topk_p3extra_{n}.csv" for n in names}
    acc = {n: (pd.read_csv(p).to_dict("records") if p.exists() else []) for n, p in paths.items()}
    done = {(r["domain"], r["seed"]) for r in acc["results"]}
    todo = [(d, s) for d in domains for s in range(n_seeds) if (d, s) not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    with Pool(workers) as pool:
        for *parts, msg in pool.imap_unordered(task, todo):
            for n, part in zip(names, parts):
                acc[n] += part
                pd.DataFrame(acc[n]).to_csv(paths[n], index=False)
            print(msg, flush=True)
    summarize()


if __name__ == "__main__":
    main()

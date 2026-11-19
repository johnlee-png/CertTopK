"""Loss-aware priority P3: refine first the candidates most likely to change top-k membership.

P2 ranks candidates by the probability of crossing the cheap cut-off theta_c, a proxy for the top-k loss. P3 targets
the loss itself. Using the error model fitted on the fit split (normalized residuals Z = (P_e - P_c) / s_hat, resampled
from the fit split), it draws NS joint scenarios of the expensive utilities of the whole query,
    U_e^(b) = U_c + (R + C) * s_hat * Z^(b),
computes the top-k in each scenario, and estimates for every candidate the probability that its membership differs from
the cheap top-k:  p_i = P(i in T*) for i outside the cheap top-k, P(i not in T*) for i inside.
Because the reference cut-off theta_e is simulated jointly, P3 accounts for the competition among candidates, which
P2 ignores. Priority P3_i = 1 - p_i (ties broken by P2). It depends only on the query, D0 and independent randomness
drawn per query (fixed seed per split), so Theorem 1 applies unchanged; CRC calibrates the threshold as for P2.

Writes paper/src_extension/selective_topk_lossaware_{results,summary}.csv.
Usage: py src_extension_selective_topk_lossaware.py <n_seeds> [domains] [workers] | summarize
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
from src_vldb_topk_ties import topk_mask_batch

OUT = Path("paper/src_extension")
NS = 200
PRIOS = ("P2", "P3")


def add_p3(qs, d, scale, z_pool, k_frac, rng):
    s_all = scale(d)
    budget = d["R"] + d["C"]
    i = 0
    for qid in np.unique(d["qid"]):
        ix = np.flatnonzero(d["qid"] == qid)
        k = max(1, int(round(k_frac * len(ix))))
        if k >= len(ix):
            continue
        q = qs[i]
        i += 1
        uc = q["uc"]
        n = len(uc)
        ue_sim = uc[None, :] + (budget[ix] * s_all[ix])[None, :] * rng.choice(z_pool, size=(NS, n))
        in_sim = topk_mask_batch(ue_sim, k)
        in_cheap = np.zeros(n, bool)
        in_cheap[crc.topk(uc, k)] = True
        flip = np.where(in_cheap[None, :], ~in_sim, in_sim).mean(axis=0)
        p2 = q["pri"]["P2"]
        q["pri"]["P3"] = 1.0 - flip + 1e-6 * p2 / (p2.max() + 1e-12)
    assert i == len(qs)


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    f = data["fit"]
    z_pool = (f["pe"] - f["pc"]) / scale(f)
    rows = []
    for k_frac in crc.K_FRACTIONS:
        sets = {}
        for split, stream in (("fit", 0), ("cal", 1), ("test", 2)):
            qs = crc.query_arrays(data[split], scale, k_frac, seed, stream=stream)
            add_p3(qs, data[split], scale, z_pool, k_frac, np.random.default_rng([seed, stream, 99]))
            sets[split] = qs
        for name in PRIOS:
            pooled = np.concatenate([q["pri"][name] for q in sets["fit"]])
            grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
            lt_cal = np.array([crc.envelope(crc.loss_curve(q, name, grid)[0]) for q in sets["cal"]])
            curves = [crc.loss_curve(q, name, grid) for q in sets["test"]]
            l_test = np.array([c[0] for c in curves])
            f_test = np.array([c[1] for c in curves])
            m = len(sets["cal"])
            for eps in crc.EPSILONS:
                ok = np.flatnonzero(m / (m + 1) * lt_cal.mean(axis=0) + 1 / (m + 1) <= eps)
                j = ok[0] if len(ok) else len(grid) - 1
                rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps, "priority": name,
                             "m_cal": m, "test_miss": float(l_test[:, j].mean()),
                             "test_call_fraction": float(f_test[:, j].mean())})
    return rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def summarize():
    t = pd.read_csv(OUT / "selective_topk_lossaware_results.csv").assign(over=lambda x: (x.test_miss > x.eps).astype(int))
    summ = t.groupby(["domain", "k_fraction", "eps", "priority"]).agg(
        seeds=("seed", "nunique"), miss_mean=("test_miss", "mean"), miss_sd=("test_miss", "std"),
        seeds_over_eps=("over", "sum"), calls_mean=("test_call_fraction", "mean")).reset_index()
    summ.to_csv(OUT / "selective_topk_lossaware_summary.csv", index=False)
    piv = summ.pivot_table(index=["domain", "k_fraction", "eps"], columns="priority", values="calls_mean")
    piv["ratio_P3_P2"] = piv["P3"] / piv["P2"]
    print(piv.round(4).to_string())
    print("cells", len(summ), "mean<=eps", int((summ.miss_mean <= summ.eps).sum()))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        return summarize()
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else list(crc.DOMAINS)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    path = OUT / "selective_topk_lossaware_results.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["domain"], r["seed"]) for r in rows}
    todo = [(d, s) for d in domains for s in range(n_seeds) if (d, s) not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    with Pool(workers) as pool:
        for r, msg in pool.imap_unordered(task, todo):
            rows += r
            print(msg, flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
    summarize()


if __name__ == "__main__":
    main()

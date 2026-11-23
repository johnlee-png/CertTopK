"""P4: refine by the simulated expected reduction of the top-k loss (loss-aware order).

Theorem 4 of the paper shows that membership-change probabilities (P3/P3c) do not determine the best refinement,
because what matters is the change of the *answer* that a refinement causes, not whether the candidate's own
membership flips. P4 estimates that change directly. For one query, with the cheap top-k T_c (ties to the smaller
index), its weakest member l and the strongest non-member j, refining candidate i alone in scenario b (expensive
utilities u_b simulated as in P3c, from the fit-split residuals of the error-size model, probabilities clipped to
[0, 1]) changes the answer only by a swap:
    i in T_c:     i drops iff u_b,i loses to U^c_j        ->  T_c - {i} + {j}
    i not in T_c: i enters iff u_b,i beats U^c_l      ->  T_c - {l} + {i}
and the loss reduction against the scenario's reference top-k T*_b is ([in] - [out]) / k for the swapped pair.
    P4  = - mean_b Delta_b,i            (expected one-step loss reduction; may be negative)
    P4+ = - mean_b max(Delta_b,i, 0)    (expected benefit when the refinement helps)
Smaller is refined first; ties (e.g. candidates that cannot cross) are broken by P2. Both are functions of the
query's cheap information and of independent randomness, so the CRC guarantee holds unchanged. They are myopic:
by Theorem 3 single-refinement gains do not add up, so neither order is claimed optimal.

Evaluation (same splits, queries, and P3c scenarios as src_extension_selective_topk_baselines.py, so P3c and P4 differ
only in the decision rule): per (domain, seed, kappa, eps), CRC test loss and calls, fit-split plug-in calls (for
Auto-calls), equal-call curves, and the time to compute each priority.
Writes paper/src_extension/selective_topk_p4_{results,curves}.csv (relative to the working directory).
Usage: py src_vldb_p4.py <n_seeds> [domains] [workers] | test
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_p3_extra import arrays, crc_calls, fit_plugin_calls
from src_vldb_topk_ties import flip_prob, topk_mask_batch

OUT = Path("paper/src_extension")
B = 200
F_GRID = np.round(np.linspace(0, 1, 101), 2)
ORDERS = tuple(os.environ.get("P4_ORDERS", "P2,P3CLIP,P4,P4PLUS,P5").split(","))
TAG = os.environ.get("P4_TAG", "p4")                     # output name: selective_topk_<TAG>_{results,curves}.csv


def p5_tail_priority(p5):
    """P5T: refine while the remaining bound sum_{unrefined} w_i is large (Proposition A.8 of proofs.pdf).

    p5 = 1 - w + tiny tie term; along P5's order the tail sums of w decrease, so the sets {tail >= lambda} are
    prefixes of P5's order with a query-dependent length."""
    order = np.argsort(p5, kind="stable")
    w = np.clip(1.0 - p5[order], 0, 1)
    tail = np.cumsum(w[::-1])[::-1]
    pri = np.empty(len(p5))
    pri[order] = -tail + 1e-9 * np.arange(len(p5)) / len(p5)
    return pri


def beats(a, ia, b, ib):
    """a (candidate ia) ranks above b (candidate ib) under the tie rule: larger value, then smaller index."""
    return (a > b) | ((a == b) & (ia < ib))


def p4_priorities(uc, u_sim, k, p2):
    """(P4, P4+) priorities of one query; u_sim: B x n simulated expensive utilities."""
    n = len(uc)
    order = np.lexsort((np.arange(n), -uc))           # descending uc, ties to the smaller index
    in_c = np.zeros(n, bool)
    in_c[order[:k]] = True
    l, j = order[k - 1], order[k]
    ref = topk_mask_batch(u_sim, k)                   # B x n, T*_b
    idx = np.arange(n)
    drop = in_c[None, :] & ~beats(u_sim, idx[None, :], uc[j], j)          # i in T_c falls below j
    enter = ~in_c[None, :] & beats(u_sim, idx[None, :], uc[l], l)        # i outside beats l
    delta = np.zeros_like(u_sim)
    delta = np.where(drop, (ref[:, [j]].astype(float) - ref) / k, delta)
    delta = np.where(enter, (ref - ref[:, [l]].astype(float)) / k, delta)
    tie = 1e-6 * p2 / (p2.max() + 1e-12)
    return -delta.mean(axis=0) + tie, -np.maximum(delta, 0).mean(axis=0) + tie


def wrong_side(uc, u_sim, k):
    """B x n: candidate i's cheap score lies on the wrong side of the scenario's expensive cut-off
    theta_b = midpoint of the k-th and (k+1)-th simulated utilities (members at or below it, non-members at or
    above it count as wrong)."""
    srt = -np.sort(-u_sim, axis=1)
    theta = 0.5 * (srt[:, k - 1:k] + srt[:, k:k + 1])
    ref = topk_mask_batch(u_sim, k)
    return np.where(ref, uc[None, :] <= theta, uc[None, :] >= theta)


def p5_priority(uc, u_sim, k, p2):
    """P5: refine first the candidates whose cheap score is most likely on the wrong side of the expensive cut-off.

    w_i = Pr_b(wrong side). For every scenario and every refined set S, L(S) <= #{i not in S : wrong side}/k
    (paper, Theorem 6), so E[L(S)] <= sum_{i not in S} w_i / k, and refining in decreasing w_i minimizes this bound
    at every budget simultaneously. P2 is the analogue for the cheap cut-off."""
    w = wrong_side(uc, u_sim, k).mean(axis=0)
    return 1.0 - w + 1e-6 * p2 / (p2.max() + 1e-12)


def bound_test(n_cases=20000, seed=1):
    """L(S) <= #{i not in S : wrong side of theta}/k for random scores (continuous and with ties) and random S."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for c in range(n_cases):
        n = int(rng.integers(3, 12))
        k = int(rng.integers(1, n))
        if c % 2:
            uc, ue = rng.normal(size=n), rng.normal(size=n)
        else:
            uc, ue = rng.integers(0, 4, n).astype(float), rng.integers(0, 4, n).astype(float)
        S = rng.random(n) < rng.random()
        ref = np.zeros(n, bool)
        ref[crc.topk(ue, k)] = True
        ans = np.zeros(n, bool)
        ans[crc.topk(np.where(S, ue, uc), k)] = True
        loss = 1 - (ans & ref).sum() / k
        bound = (wrong_side(uc, ue[None, :], k)[0] & ~S).sum() / k
        assert loss <= bound + 1e-12, (uc, ue, k, S, loss, bound)
        worst = max(worst, loss - bound)
    print(f"bound test passed ({n_cases} random cases, max L - bound = {worst:.3f})")


def self_test(n_cases=300, seed=0):
    """Compare the swap formula with explicit top-k of the mixed scores."""
    rng = np.random.default_rng(seed)
    for _ in range(n_cases):
        n = int(rng.integers(4, 12))
        k = int(rng.integers(1, n))
        uc = rng.integers(0, 4, n).astype(float)
        u_sim = rng.integers(0, 4, (5, n)).astype(float)
        p4, _ = p4_priorities(uc, u_sim, k, np.ones(n))
        ref = topk_mask_batch(u_sim, k)
        base = np.zeros(n, bool)
        base[crc.topk(uc, k)] = True
        want = np.zeros(n)
        for i in range(n):
            d = 0.0
            for b in range(len(u_sim)):
                u = uc.copy()
                u[i] = u_sim[b, i]
                ans = np.zeros(n, bool)
                ans[crc.topk(u, k)] = True
                d += ((ans & ref[b]).sum() - (base & ref[b]).sum()) / k
            want[i] = -d / len(u_sim)
        assert np.allclose(p4 - 1e-6 / (1 + 1e-12), want), (uc, u_sim, k, p4, want)
    print(f"self-test passed ({n_cases} random cases)")


def equal_call_curve(qs, name):
    """Mean loss on the test queries when the first ceil(f n_Q) candidates of the order are refined."""
    miss = np.zeros(len(F_GRID))
    for q in qs:
        n, k = len(q["uc"]), q["k"]
        rank = np.argsort(q["pri"][name], kind="stable")
        ref = set(crc.topk(q["ue"], k).tolist())
        for g, f in enumerate(F_GRID):
            u = q["uc"].copy()
            sel = rank[:int(np.ceil(f * n))]
            u[sel] = q["ue"][sel]
            miss[g] += 1 - len(ref & set(crc.topk(u, k).tolist())) / k
    return miss / len(qs)


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    f = data["fit"]
    z_all = (f["pe"] - f["pc"]) / scale(f)
    rows, curves = [], []
    for k_frac in crc.K_FRACTIONS:
        sets, secs = {}, {n: 0.0 for n in ("P2", "P3CLIP", "P4", "P4PLUS", "P5", "P5T")}
        for split, stream in (("fit", 0), ("cal", 1), ("test", 2)):
            d = data[split]
            qs = arrays(d, scale, k_frac, seed, stream)
            s_all = scale(d)
            rng_c = np.random.default_rng([seed, stream, 499])      # the P3c stream of the baselines script
            for q in qs:
                ix, k = q["_ix"], q["k"]
                zz = rng_c.choice(z_all, size=(B, len(ix)))
                t1 = time.perf_counter()
                p_sim = d["pc"][ix][None, :] + s_all[ix][None, :] * zz
                u_sim = np.clip(p_sim, 0, 1) * (d["R"][ix] + d["C"][ix])[None, :] - d["C"][ix][None, :]
                t_sim = time.perf_counter() - t1
                p2 = q["pri"]["P2"]
                t1 = time.perf_counter()
                in_c = np.zeros(len(ix), bool)
                in_c[crc.topk(q["uc"], k)] = True
                q["pri"]["P3CLIP"] = 1.0 - flip_prob(u_sim, in_c, k) + 1e-6 * p2 / (p2.max() + 1e-12)
                t_p3 = time.perf_counter() - t1
                t1 = time.perf_counter()
                q["pri"]["P4"], q["pri"]["P4PLUS"] = p4_priorities(q["uc"], u_sim, k, p2)
                t_p4 = time.perf_counter() - t1
                t1 = time.perf_counter()
                q["pri"]["P5"] = p5_priority(q["uc"], u_sim, k, p2)
                q["pri"]["P5T"] = p5_tail_priority(q["pri"]["P5"])
                t_p5 = time.perf_counter() - t1
                if split == "test":
                    secs["P3CLIP"] += t_sim + t_p3
                    secs["P4"] += t_sim + t_p4
                    secs["P5"] += t_sim + t_p5
                    secs["P5T"] = secs.get("P5T", 0.0) + t_sim + t_p5
            sets[split] = qs
        n_test = len(sets["test"])
        for name in ORDERS:
            res = crc_calls(sets, name)
            fit = fit_plugin_calls(sets["fit"], name)
            for eps in crc.EPSILONS:
                miss, calls, m = res[eps]
                rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps, "priority": name,
                             "m_cal": m, "test_miss": miss, "test_call_fraction": calls, "fit_calls": fit[eps],
                             "priority_ms_per_query": 1e3 * secs[name] / n_test if name != "P2" else np.nan})
            for fg, mi in zip(F_GRID, equal_call_curve(sets["test"], name)):
                curves.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "priority": name,
                               "call_fraction": fg, "miss": mi})
    return rows, curves, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        self_test()
        bound_test()
        return
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else ["soccer", "diabetes", "home_credit", "kdd98", "telco",
                                                                "ieee_fraud", "lending_club", "lc_llm"]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    res_p, cur_p = OUT / f"selective_topk_{TAG}_results.csv", OUT / f"selective_topk_{TAG}_curves.csv"
    rows = pd.read_csv(res_p).to_dict("records") if res_p.exists() else []
    curves = pd.read_csv(cur_p).to_dict("records") if cur_p.exists() else []
    done = {(r["domain"], r["seed"]) for r in rows}
    todo = [(d, s) for d in domains for s in range(n_seeds) if (d, s) not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    with Pool(workers) as pool:
        for r, c, msg in pool.imap_unordered(task, todo):
            rows += r
            curves += c
            pd.DataFrame(rows).to_csv(res_p, index=False)
            pd.DataFrame(curves).to_csv(cur_p, index=False)
            print(msg, flush=True)


if __name__ == "__main__":
    main()

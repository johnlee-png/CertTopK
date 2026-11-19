"""One executable CertTopK query path, shared by profiling, policy selection, calibration and the timed runs.

A workload supplies the per-query steps; `answer` runs them in order and returns the certified-or-fallback top-k:

    cheap inference -> priority -> candidate selection (priority <= lambda) -> actual expensive access
    (feature lookup / LLM call) -> expensive score -> mixed top-k

and the caller times `answer` from the outside. Every priority here is a function of the query alone and of
independent randomness seeded by the query id, so the CRC guarantee of the paper applies to each fixed order.
Top-k always uses crc.topk / topk_mask_batch (exactly k, ties to the smaller index).
"""
import time

import numpy as np

import src_extension_selective_topk_crc as crc
from src_vldb_topk_ties import flip_prob

ORDERS = ["P0", "P1", "PU", "TOPM", "P2", "P3c"]
NEEDS_SCALE = {"P2", "P3c"}
NEEDS_SU = {"PU"}
B = 200


def utility(p, R, C):
    return p * (R + C) - C


def priority(name, uc, k, R, C, pc=None, s=None, su=None, z_all=None, rng=None):
    """Priority of every candidate of one query (smaller = refined first)."""
    part = -np.partition(-uc, [k - 1, k]) if k < len(uc) else -np.sort(-uc)
    dist = np.abs(uc - 0.5 * (part[k - 1] + part[min(k, len(uc) - 1)]))
    budget = R + C
    if name == "P0":
        return dist
    if name == "P1":
        return dist / budget
    if name == "PU":
        return dist / su
    if name == "TOPM":
        r = np.empty(len(uc))
        r[np.argsort(-uc, kind="stable")] = np.arange(len(uc))
        return r / k
    p2 = dist / (budget * s)
    if name == "P2":
        return p2
    if name == "P3c":
        p_sim = np.clip(pc[None, :] + s[None, :] * rng.choice(z_all, size=(B, len(uc))), 0, 1)
        u_sim = p_sim * budget[None, :] - C[None, :]
        in_cheap = np.zeros(len(uc), bool)
        in_cheap[crc.topk(uc, k)] = True
        return 1.0 - flip_prob(u_sim, in_cheap, k) + 1e-6 * p2 / (p2.max() + 1e-12)
    raise ValueError(name)


def answer(wl, q, policy, lam, timings=None):
    """Run one query end to end. wl: workload; q: query handle from wl.queries(); policy: an order name or "ALL".
    Returns (top-k candidate positions within the query, number of expensive calls). If `timings` is a dict, the
    stage times are added to it (they are diagnostics; the reported latency is measured around this call)."""
    tic = time.perf_counter
    t0 = tic()
    pc, aux = wl.cheap(q)                                  # cheap inference on the query's candidates
    R, C = wl.R(q), wl.C(q)
    uc = utility(pc, R, C)
    k = wl.k(q)
    t1 = tic()
    if policy == "ALL":
        sel = np.arange(len(uc))
    else:
        s = wl.scale(q, pc, aux) if policy in NEEDS_SCALE else None
        su = wl.su(q, pc, aux) if policy in NEEDS_SU else None
        rng = np.random.default_rng([wl.seed, 777, wl.qkey(q)]) if policy == "P3c" else None
        pri = priority(policy, uc, k, R, C, pc, s, su, wl.z_all, rng)
        sel = np.flatnonzero(pri <= lam)
    t2 = tic()
    u = uc.copy()
    if len(sel):
        pe = wl.expensive(q, sel, aux)                     # actual feature access / LLM calls + expensive model
        u[sel] = utility(pe, R[sel], C[sel])
    t3 = tic()
    top = crc.topk(u, k)
    t4 = tic()
    if timings is not None:
        for key, v in (("cheap", t1 - t0), ("priority", t2 - t1), ("expensive", t3 - t2), ("topk", t4 - t3)):
            timings[key] = timings.get(key, 0.0) + v
    return top, len(sel)


def loss(top, ue, k):
    ref = crc.topk(ue, k)
    return 1.0 - len(np.intersect1d(top, ref)) / k


# ---------------------------------------------------------------- offline: curves, CRC threshold, plug-in calls

def grid_of(pooled):
    return np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))


def curves(qs, name, grid):
    """Per query: envelope of the membership loss and the refined fraction on the grid.
    qs: list of dicts with uc, ue, k and pri[name]."""
    cs = [crc.loss_curve(q, name, grid) for q in qs]
    return np.array([crc.envelope(c[0]) for c in cs]), np.array([c[1] for c in cs])


def crc_lambda(env, grid, eps):
    """Smallest grid point whose CRC-adjusted calibration risk is <= eps (inf = refine everyone)."""
    m = len(env)
    ok = np.flatnonzero(m / (m + 1) * env.mean(axis=0) + 1 / (m + 1) <= eps)
    return grid[ok[0]] if len(ok) else np.inf


def plug_in_calls(env_fit, frac_fit, eps):
    ok = np.flatnonzero(env_fit.mean(axis=0) <= eps)
    j = ok[0] if len(ok) else env_fit.shape[1] - 1
    return float(frac_fit[:, j].mean())

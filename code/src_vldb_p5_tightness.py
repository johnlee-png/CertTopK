"""How tight is the wrong-side bound, and when do P5 and P3c differ? (bound tightness; where P5 helps)

For every test query (kappa = 10%, seeds 0..N-1, same queries and simulated scenarios as src_vldb_p4.py), along P5's
order at refined fractions f in F:
  L      actual membership loss of the mixed top-k
  B_max  sharper realized bound: max(#unrefined wrong-side members, #unrefined wrong-side non-members) / k
  B_real realized bound: #unrefined wrong-side candidates / k under the actual expensive scores (Theorem 5: L <= B_real)
  B_est  estimated bound: sum of the unrefined estimated w_i / k (what P5 minimizes; equals E[B_real] if w is exact)
Reported per domain: mean L, B_real, B_est, L / B_real, share of (query, f) with L = B_real > 0, and the calibration
of B_est against B_real. To see when P5 and P3c differ: the overlap of their first c candidates (c = the calls P5
needs at eps = 0.05), and the cut-off shift |theta_e - theta_c| between the expensive and the cheap cut-off in units of
the cheap scores' spread around the cut-off, which separates the two reference points of P3 and P5.
Writes paper/src_extension/selective_topk_p5_tightness.csv (relative to the working directory).
Usage: py src_vldb_p5_tightness.py [n_seeds] [domains] [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_p3_extra import arrays
from src_vldb_p4 import B, p5_priority, wrong_side
from src_vldb_topk_ties import flip_prob

OUT = Path("paper/src_extension/selective_topk_p5_tightness.csv")
F = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
K_FRAC = 0.10


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    f = data["fit"]
    z_all = (f["pe"] - f["pc"]) / scale(f)
    d = data["test"]
    qs = arrays(d, scale, K_FRAC, seed, 2)
    s_all = scale(d)
    rng_c = np.random.default_rng([seed, 2, 499])
    rows = []
    for q in qs:
        ix, k, uc, ue = q["_ix"], q["k"], q["uc"], q["ue"]
        n = len(ix)
        zz = rng_c.choice(z_all, size=(B, n))
        p_sim = np.clip(d["pc"][ix][None, :] + s_all[ix][None, :] * zz, 0, 1)
        u_sim = p_sim * (d["R"][ix] + d["C"][ix])[None, :] - d["C"][ix][None, :]
        p2 = q["pri"]["P2"]
        p5 = p5_priority(uc, u_sim, k, p2)
        w_hat = np.clip(1.0 - p5, 0, 1)
        in_c = np.zeros(n, bool)
        in_c[crc.topk(uc, k)] = True
        p3c = 1.0 - flip_prob(u_sim, in_c, k) + 1e-6 * p2 / (p2.max() + 1e-12)
        wrong = wrong_side(uc, ue[None, :], k)[0]
        ref = np.zeros(n, bool)
        ref[crc.topk(ue, k)] = True
        o5, o3 = np.argsort(p5, kind="stable"), np.argsort(p3c, kind="stable")
        su, se = np.sort(uc)[::-1], np.sort(ue)[::-1]
        theta_c, theta_e = 0.5 * (su[k - 1] + su[k]), 0.5 * (se[k - 1] + se[k])
        lo, hi = max(0, k - 1 - n // 10), min(n, k + n // 10)
        spread = np.std(su[lo:hi]) + 1e-12
        for fr in F:
            c = int(np.ceil(fr * n))
            S = np.zeros(n, bool)
            S[o5[:c]] = True
            ans = np.zeros(n, bool)
            ans[crc.topk(np.where(S, ue, uc), k)] = True
            rows.append({"domain": domain, "seed": seed, "qid": len(rows) // len(F), "f": fr, "n": n, "k": k,
                         "L": 1 - (ans & ref).sum() / k, "B_real": (wrong & ~S).sum() / k,
                         "B_max": max((wrong & ~S & ref).sum(), (wrong & ~S & ~ref).sum()) / k,
                         "B_est": w_hat[~S].sum() / k,
                         "overlap_p3c": len(set(o5[:c]) & set(o3[:c])) / c if c else np.nan,
                         "cutoff_shift": abs(theta_e - theta_c) / spread})
    return rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def summarize(t):
    t = t.copy()
    t["tight"] = (t.L == t.B_real) & (t.B_real > 0)
    t["tight_max"] = (t.L == t.B_max) & (t.B_max > 0)
    g = t[t.f > 0].groupby("domain")
    out = g.agg(L=("L", "mean"), B_real=("B_real", "mean"), B_max=("B_max", "mean"), B_est=("B_est", "mean"),
                share_equal=("tight", "mean"), share_equal_max=("tight_max", "mean")).reset_index()
    out["L_over_B_max"] = out.L / out.B_max
    out["L_over_B_real"] = out.L / out.B_real
    out["B_est_over_B_real"] = out.B_est / out.B_real
    q = t.drop_duplicates(["domain", "seed", "qid"])
    out = out.merge(q.groupby("domain").cutoff_shift.median().rename("cutoff_shift_median").reset_index(), on="domain")
    ov = t[t.f.isin([0.05, 0.1])].groupby("domain").overlap_p3c.mean().rename("overlap_p3c_5_10pct").reset_index()
    return out.merge(ov, on="domain")


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else ["soccer", "diabetes", "home_credit", "kdd98", "telco",
                                                                "ieee_fraud", "lending_club", "lc_llm"]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    rows = []
    with Pool(workers) as pool:
        for r, msg in pool.imap_unordered(task, [(d, s) for d in domains for s in range(n_seeds)]):
            rows += r
            print(msg, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)
    s = summarize(t)
    s.to_csv(OUT.with_name("selective_topk_p5_tightness_summary.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(s.round(3).to_string(index=False))
        print(t.groupby(["domain", "f"])[["L", "B_real", "B_est"]].mean().unstack("f").round(3).to_string())


if __name__ == "__main__":
    main()

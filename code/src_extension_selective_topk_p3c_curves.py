"""Equal-call curves of P3c (P3 with simulated probabilities clipped to [0, 1]) on the test queries, for Fig. 5. Same F grid and miss as src_extension_selective_topk_pareto.py, same P3c construction
as src_extension_selective_topk_baselines.py (stream [seed, 2, 499]).
Writes paper/src_extension/selective_topk_p3c_curves.csv.
Usage: py src_extension_selective_topk_p3c_curves.py <n_seeds> [domains] [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_baselines import topk_mask
from src_extension_selective_topk_p3_extra import arrays
from src_extension_selective_topk_pareto import F_GRID, miss_of

OUT = Path("paper/src_extension/selective_topk_p3c_curves.csv")
B = 200


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    f = data["fit"]
    z_all = (f["pe"] - f["pc"]) / scale(f)
    d = data["test"]
    s_all = scale(d)
    rows = []
    for k_frac in [0.10]:                            # Fig. 5 shows kappa = 10%
        qs = arrays(d, scale, k_frac, seed, 2)
        rng = np.random.default_rng([seed, 2, 499])
        acc = np.zeros(len(F_GRID))
        for q in qs:
            ix, k = q["_ix"], q["k"]
            p_sim = d["pc"][ix][None, :] + s_all[ix][None, :] * rng.choice(z_all, size=(B, len(ix)))
            u_sim = np.clip(p_sim, 0, 1) * (d["R"][ix] + d["C"][ix])[None, :] - d["C"][ix][None, :]
            kth = -np.partition(-u_sim, k - 1, axis=1)[:, k - 1:k]
            flip = np.where(topk_mask(q["uc"], k)[None, :], u_sim < kth, u_sim >= kth).mean(axis=0)
            p2 = q["pri"]["P2"]
            pri = 1.0 - flip + 1e-6 * p2 / (p2.max() + 1e-12)
            ref = set(crc.topk(q["ue"], k).tolist())
            order = np.argsort(pri, kind="stable")
            n = len(ix)
            for j, m in enumerate(np.ceil(F_GRID * n).astype(int)):
                mixed = q["uc"].copy()
                mixed[order[:m]] = q["ue"][order[:m]]
                acc[j] += miss_of(crc.topk(mixed, k), ref, k)
        for fv, v in zip(F_GRID, acc / len(qs)):
            rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "method": "P3c", "call_fraction": fv,
                         "miss": v})
    return rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else ["soccer", "diabetes", "home_credit", "kdd98", "telco",
                                                                "ieee_fraud", "lending_club", "lc_llm"]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["domain"], r["seed"]) for r in rows}
    todo = [(dm, s) for dm in domains for s in range(n_seeds) if (dm, s) not in done]
    with Pool(workers) as pool:
        for r, msg in pool.imap_unordered(task, todo):
            rows += r
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(msg, flush=True)


if __name__ == "__main__":
    main()

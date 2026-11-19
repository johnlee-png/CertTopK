"""Fit-split plug-in calls of P0 and PU.

Adds the simple orders P0 and PU to the candidates of Auto-calls (P1, P2, TOPM, EVEREST_CRC, P3, P3c). This script computes their fit-split plug-in calls with the
same protocol as src_extension_selective_topk_order_select_final.py (fit split, fit grid, envelope mean <= eps); PU uses
the |U^e - U^c| regressor of src_extension_selective_topk_baselines.py with the same settings, so its test results in
selective_topk_c5_results.csv are the matching deployed order.
Writes paper/src_extension/selective_topk_auto_ext_fitcalls.csv (relative to the working directory).
Usage: py src_vldb_auto_extended.py <n_seeds> [domains] [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_order_select import plug_in
from src_extension_selective_topk_p3_extra import arrays

OUT = Path("paper/src_extension/selective_topk_auto_ext_fitcalls.csv")


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    scale = crc.fit_scale(data["fit"])
    f = data["fit"]
    uf_c, uf_e = crc.utility(f["pc"], f["R"], f["C"]), crc.utility(f["pe"], f["R"], f["C"])
    Xf = np.column_stack([f["xc"], f["pc"], f["R"], f["C"]])
    su_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                 verbosity=-1, random_state=42).fit(Xf, np.abs(uf_e - uf_c))
    su_floor = float(np.quantile(np.abs(uf_e - uf_c), 0.05)) + 1e-9
    su_all = np.maximum(su_model.predict(Xf), su_floor)
    rows = []
    for k_frac in crc.K_FRACTIONS:
        fit = arrays(f, scale, k_frac, seed, 0)
        for q in fit:
            ix = q["_ix"]
            q["pri"]["PU"] = q["pri"]["P1"] * (f["R"][ix] + f["C"][ix]) / su_all[ix]
        r = {eps: {} for eps in crc.EPSILONS}
        for name in ("P0", "PU"):
            pooled = np.concatenate([q["pri"][name] for q in fit])
            grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, crc.N_GRID + 1)), [np.inf]]))
            cs = [crc.loss_curve(q, name, grid) for q in fit]
            le, fc = np.array([crc.envelope(c[0]) for c in cs]), np.array([c[1] for c in cs])
            for eps in crc.EPSILONS:
                r[eps][f"fit_calls_{name}"] = plug_in(le, fc, eps)
        for eps in crc.EPSILONS:
            rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps, **r[eps]})
    return rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else ["soccer", "diabetes", "home_credit", "kdd98", "telco",
                                                                "ieee_fraud", "lending_club", "lc_llm"]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["domain"], r["seed"]) for r in rows}
    todo = [(d, s) for d in domains for s in range(n_seeds) if (d, s) not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    with Pool(workers) as pool:
        for r, msg in pool.imap_unordered(task, todo):
            rows += r
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(msg, flush=True)


if __name__ == "__main__":
    main()

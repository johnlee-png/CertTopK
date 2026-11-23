"""Summary tables of the end-to-end runs: Home Credit (src_vldb_e2e_hc.py, by access path) and LC-LLM
(src_vldb_e2e_llm.py, by LLM concurrency; "fresh" there recomputes the LLM outputs of all fit/cal loans once).

Per (size, path, eps): mean/p95 latency, throughput, calls and loss of ALL, every fixed order, Auto-calls and Auto-time
(the order each rule chose in that cell, measured in the same run), and the post-hoc fastest order. Initial cost of
CertTopK beyond what ALL also needs (the access path and the workload models are shared): "reuse" = calibration
compute + profiling (expensive outputs of fit/cal taken from logs); "fresh" = reuse + computing those expensive outputs
+ training the side models (s-hat, su). Break-even = initial cost / (ALL mean - policy mean) queries.
Writes paper/src_extension/e2e/<wl>_summary.csv and <wl>_breakeven.csv.
Usage: py src_vldb_e2e_hc_report.py [hc|llm]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("paper/src_extension/e2e")
WL = sys.argv[1] if len(sys.argv) > 1 else "hc"
DIM, SUB = ("path", "paths") if WL.startswith("hc") else ("conc", "conc")


def variability(q, out, n_boot=2000, seed=0):
    """Per configuration and eps: speed-up of Auto-time over ALL and its extra gain over P0, as ratios of mean
    latencies on the same test queries, with 95% bootstrap intervals over queries (paired resampling)."""
    rng = np.random.default_rng(seed)
    rows = []
    for (size, dim), sq in q.groupby(["size", DIM]):
        lat_all = sq[sq.policy == "ALL"].set_index("qkey").latency_s
        for eps in sorted(e for e in sq.eps.unique() if e >= 0):
            at = out[(out["size"] == size) & (out[DIM] == dim) & (out.eps == eps) & (out.policy == "Auto-time")]
            order = at.order.iloc[0]
            se = sq[sq.eps == eps]
            lat_o = se[se.policy == order].set_index("qkey").latency_s
            lat_p0 = se[se.policy == "P0"].set_index("qkey").latency_s
            keys = lat_o.index.intersection(lat_all.index).intersection(lat_p0.index)
            a, o, p0 = lat_all[keys].to_numpy(), lat_o[keys].to_numpy(), lat_p0[keys].to_numpy()
            idx = rng.integers(0, len(keys), (n_boot, len(keys)))
            sp = a[idx].mean(1) / o[idx].mean(1)
            g0 = p0[idx].mean(1) / o[idx].mean(1) - 1
            rows.append({"size": size, DIM: dim, "eps": eps, "auto_time": order, "n_q": len(keys),
                         "speedup": a.mean() / o.mean(), "speedup_lo": np.quantile(sp, .025),
                         "speedup_hi": np.quantile(sp, .975), "cv_latency": o.std() / o.mean(),
                         "gain_vs_P0": p0.mean() / o.mean() - 1, "gain_vs_P0_lo": np.quantile(g0, .025),
                         "gain_vs_P0_hi": np.quantile(g0, .975)})
    return pd.DataFrame(rows)


def main():
    q = pd.read_csv(D / f"{WL}_queries.csv")
    setup = json.loads((D / f"{WL}_setup.json").read_text())
    q["eps"] = q["eps"].fillna(-1)
    g = q.groupby(["size", DIM, "eps", "policy"])
    s = g.agg(n_q=("latency_s", "size"), mean_s=("latency_s", "mean"), p95_s=("latency_s", lambda x: np.quantile(x, .95)),
              calls=("calls", "mean"), call_frac=("calls", lambda x: 0), loss=("loss", "mean"),
              loss_max=("loss", "max")).reset_index()
    s["call_frac"] = (g["calls"].mean() / g["n"].mean()).to_numpy()
    s["qps"] = 1 / s.mean_s
    rows, be = [], []
    for (size, path), sp in s.groupby(["size", DIM]):
        ps = setup["per_size"][str(size)]
        pp = ps[SUB][str(path)]
        all_row = sp[sp.policy == "ALL"].iloc[0]
        reuse = sum(ps["calibration_compute_s"].values()) + pp["profiling_s"]
        fresh = reuse + pp.get("fresh_expensive_outputs_s", setup.get("fresh_llm_outputs_s", 0)) + setup["side_models_s"]
        for eps in sorted(sp.eps.unique()):
            if eps < 0:
                continue
            se = sp[sp.eps == eps].set_index("policy")
            ch = pp["choice"][str(eps)]
            best = se.mean_s.idxmin()
            for lab, pol in [("ALL", None)] + [(o, o) for o in se.index] + \
                            [("Auto-calls", ch["auto_calls"]), ("Auto-time", ch["auto_time"]), ("post-hoc best", best)]:
                r = all_row if pol is None else se.loc[pol]
                rows.append({"size": size, DIM: path, "eps": eps, "policy": lab, "order": pol or "ALL",
                             **{c: r[c] for c in ("mean_s", "p95_s", "qps", "calls", "call_frac", "loss", "loss_max")},
                             "speedup_vs_all": all_row.mean_s / r.mean_s,
                             "pred_s": ch["predicted_s"].get(pol or "ALL", np.nan)})
                if lab in ("Auto-calls", "Auto-time"):
                    gain = all_row.mean_s - r.mean_s
                    be.append({"size": size, DIM: path, "eps": eps, "rule": lab, "order": pol,
                               "gain_per_query_s": gain, "init_reuse_s": reuse, "init_fresh_s": fresh,
                               "breakeven_reuse_q": reuse / gain if gain > 0 else np.inf,
                               "breakeven_fresh_q": fresh / gain if gain > 0 else np.inf})
    out, be = pd.DataFrame(rows), pd.DataFrame(be)
    out.to_csv(D / f"{WL}_summary.csv", index=False)
    be.to_csv(D / f"{WL}_breakeven.csv", index=False)
    variability(q, out).to_csv(D / f"{WL}_variability.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        key = out[out.policy.isin(["ALL", "P0", "P2", "P3c", "Auto-calls", "Auto-time", "post-hoc best"])]
        print(key[["size", DIM, "eps", "policy", "order", "mean_s", "p95_s", "calls", "loss", "speedup_vs_all",
                   "pred_s"]].round(4).to_string(index=False))
        print(be.round(3).to_string(index=False))
        print("setup:", {k: round(v, 2) for k, v in setup.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()

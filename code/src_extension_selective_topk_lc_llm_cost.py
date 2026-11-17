"""Cost of the lc_llm domain's paths, per loan: cheap model (LightGBM on the
listing), scale model s, and the LLM call (from the full scoring run), and the
certified top-k time f * LLM + cheap + scale for the P2 + CRC call fractions.
Seed 0. Writes paper/src_extension/selective_topk_lc_llm_cost.json.
Usage: py src_extension_selective_topk_lc_llm_cost.py
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
import diabetes_new_regret_diagnostic as dia

OUT = Path("paper/src_extension")


def best_of(f, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    df = crc.lc_llm_frame()
    Xc = crc.encode(df, crc.LC_LISTING).astype(np.float32)
    y = pd.Series(df["repaid"].to_numpy())
    m = crc.group_split(df["id"].to_numpy(dtype=object), 0)
    cheap = dia.train_lgb(Xc[m["train"]], y[m["train"]], Xc[m["fit"]], y[m["fit"]])
    Xt = Xc[m["test"]]
    t_cheap = best_of(lambda: cheap.predict_proba(Xt)) / len(Xt)
    data = crc.DOMAINS["lc_llm"](0)
    scale = crc.fit_scale(data["fit"])
    t_scale = best_of(lambda: scale(data["test"])) / len(data["test"]["pc"])
    last = [ln for ln in (OUT / "selective_topk_llm_scores_log.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
            if re.match(r"\d+/\d+\s", ln)][-1]
    t_llm = float(re.search(r"([\d.]+)s/loan", last).group(1))
    crc_s = pd.read_csv(OUT / "selective_topk_crc_summary.csv")
    p2 = crc_s[(crc_s.domain == "lc_llm") & (crc_s.priority == "P2")]
    rows = []
    for r in p2.itertuples():
        total = t_cheap + t_scale + r.calls_mean * t_llm
        rows.append({"k_fraction": r.k_fraction, "eps": r.eps, "calls": r.calls_mean, "m_cal": r.m_cal,
                     "seconds_per_loan": total, "speedup_vs_all_llm": (t_cheap + t_llm) / total,
                     "breakeven_queries": r.m_cal / (1 - r.calls_mean)})
    out = {"cheap_seconds_per_loan": t_cheap, "scale_seconds_per_loan": t_scale, "llm_seconds_per_loan": t_llm,
           "llm_over_cheap": t_llm / t_cheap, "loans": len(df), "rows_per_query": crc.LC_LLM_ROWS,
           "calibration_llm_minutes": float(p2.m_cal.mean()) * crc.LC_LLM_ROWS * t_llm / 60, "configs": rows}
    (OUT / "selective_topk_lc_llm_cost.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

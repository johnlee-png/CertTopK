"""How expensive is the "expensive" soccer model really?

Cheap path per pass: predict with the event-only LightGBM (event features are
available from the event stream). Expensive path per pass: read the 360
freeze-frame of the pass from the raw StatsBomb three-sixty file, compute the
22 geometric context features with the project's extractor
(src/features/context_360_features.py), and predict with the event+360
LightGBM. Timed on all passes of N_MATCHES matches; the per-match file read is
counted for the expensive path because only that path needs it. The extractor
is the project's reference Python implementation (a loop per pass); a
vectorized one would be faster, so the ratio is reported as measured.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.context_360_features import extract_360_features
from src.models.pass_success import train_success_model, predict_success_proba

N_MATCHES = 10
RAW = Path("data/raw/statsbomb-open-data/data/three-sixty")


def main():
    d = Path("data/processed")
    passes = pd.read_parquet(d / "passes_with_features.parquet")
    Xe = pd.read_parquet(d / "X_event.parquet").fillna(0)
    Xf = pd.read_parquet(d / "X_full.parquet").fillna(0)
    y = pd.read_parquet(d / "y_labels.parquet")["pass_success"]
    matches = passes["match_id"].unique()
    rng = np.random.default_rng(0)
    timed = rng.choice(matches, N_MATCHES, replace=False)
    train = ~passes["match_id"].isin(timed)
    cheap = train_success_model("lightgbm", Xe[train], y[train])
    exp = train_success_model("lightgbm", Xf[train], y[train])
    ctx_cols = [c for c in Xf.columns if c not in Xe.columns]

    rows = []
    for mid in timed:
        sub = passes[passes["match_id"] == mid]
        ix = sub.index
        t0 = time.perf_counter()
        predict_success_proba(cheap, Xe.loc[ix])
        t_cheap = time.perf_counter() - t0

        t0 = time.perf_counter()
        frames = {fr["event_uuid"]: fr.get("freeze_frame", []) for fr in json.loads((RAW / f"{mid}.json").read_text(encoding="utf-8"))}
        t_read = time.perf_counter() - t0
        t0 = time.perf_counter()
        df = pd.DataFrame({"start_x": sub["start_x"].to_numpy(), "start_y": sub["start_y"].to_numpy(),
                           "end_x": sub["end_x"].to_numpy(), "end_y": sub["end_y"].to_numpy(),
                           "freeze_frame": [frames.get(e, []) for e in sub["id"]]}, index=ix)
        ctx = extract_360_features(df)
        t_feat = time.perf_counter() - t0
        X = Xe.loc[ix].copy()
        for c in ctx_cols:
            X[c] = ctx[c] if c in ctx.columns else Xf.loc[ix, c]
        t0 = time.perf_counter()
        predict_success_proba(exp, X[Xf.columns])
        t_pred = time.perf_counter() - t0
        rows.append({"match_id": int(mid), "passes": len(ix), "cheap_s": t_cheap, "read_360_s": t_read,
                     "features_360_s": t_feat, "expensive_predict_s": t_pred,
                     "missing_frames": int(sum(e not in frames for e in sub["id"]))})
    t = pd.DataFrame(rows)
    tot = t[["passes", "cheap_s", "read_360_s", "features_360_s", "expensive_predict_s"]].sum()
    exp_total = tot["read_360_s"] + tot["features_360_s"] + tot["expensive_predict_s"]
    summary = {"matches": N_MATCHES, "passes": int(tot["passes"]),
               "cheap_ms_per_pass": 1000 * tot["cheap_s"] / tot["passes"],
               "expensive_ms_per_pass": 1000 * exp_total / tot["passes"],
               "share_read": tot["read_360_s"] / exp_total, "share_features": tot["features_360_s"] / exp_total,
               "share_predict": tot["expensive_predict_s"] / exp_total,
               "cost_ratio": exp_total / tot["cheap_s"], "missing_frames": int(t["missing_frames"].sum())}
    print(t.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2))
    Path("paper/src_extension/selective_topk_soccer_cost.json").write_text(json.dumps({"summary": summary, "per_match": rows}, indent=2))


if __name__ == "__main__":
    main()

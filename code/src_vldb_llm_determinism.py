"""Is the local LLM oracle reproducible? Same 200 loans asked twice at concurrency 1 and twice at concurrency 8;
logit differences between repeats and against the logged scores (selective_topk_llm_scores.csv).
Writes paper/src_extension/e2e/llm_determinism.csv."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc
import src_extension_selective_topk_llm_trial as trial

trial.PORT = 8091
TEXT = Path("data/processed/lending_club/lc_llm_text_clean.parquet")
OUT = Path("paper/src_extension/e2e/llm_determinism.csv")


def logits(texts, conc):
    with ThreadPoolExecutor(conc) as ex:
        p = np.array([r[0] for r in ex.map(trial.ask, texts)], dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    df = crc.lc_llm_frame().sample(200, random_state=0)
    t = pd.read_parquet(TEXT).set_index("id").loc[df["id"].astype(str)]
    texts = [trial.prompt(r) for r in t.reset_index().itertuples()]
    res = {"logged": df["llm_logit"].to_numpy()}
    for conc in (1, 8):
        for rep in (1, 2):
            res[f"c{conc}_r{rep}"] = logits(texts, conc)
    r = pd.DataFrame(res)
    r.to_csv(OUT, index=False)
    pairs = [("c1_r1", "c1_r2"), ("c8_r1", "c8_r2"), ("c1_r1", "c8_r1"), ("c1_r1", "logged"), ("c8_r1", "logged")]
    for a, b in pairs:
        d = np.abs(r[a] - r[b])
        print(f"{a} vs {b}: MAE {d.mean():.4f}  max {d.max():.4f}  identical {(d < 1e-9).mean():.2f}  "
              f"corr {np.corrcoef(r[a], r[b])[0, 1]:.4f}")


if __name__ == "__main__":
    main()

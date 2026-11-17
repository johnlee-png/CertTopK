"""D2: sensitivity of the Home Credit results to the assumed margin and LGD.

R = MARGIN * credit and C = LGD * credit. Scaling both by the same factor does
not change any ranking, so only MARGIN / LGD matters; we vary both over a grid.
The models, splits and queries are those of src_extension_selective_topk_crc.py
(build_home_credit); only R and C are recomputed, which leaves P_c, P_e and the
split untouched. Writes paper/src_extension/selective_topk_crc_hc_sensitivity_*.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import src_extension_selective_topk_crc as crc

MARGINS = [0.05, 0.10, 0.20]
LGDS = [0.30, 0.45, 0.60]


def make_domain(margin, lgd):
    def build(seed):
        data = crc.build_home_credit(seed)
        out = {}
        for s, d in data.items():
            credit = d["R"] / crc.MARGIN
            out[s] = {**d, "R": margin * credit, "C": lgd * credit}
        return out
    return build


def main():
    seeds = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
    rows = []
    for margin in MARGINS:
        for lgd in LGDS:
            name = f"hc_m{margin}_l{lgd}"
            crc.DOMAINS[name] = make_domain(margin, lgd)
            for seed in seeds:
                for r in crc.run(name, seed):
                    rows.append({**r, "margin": margin, "lgd": lgd})
            print(f"margin {margin} lgd {lgd} done", flush=True)
    df = pd.DataFrame(rows)
    out = Path("paper/src_extension")
    df.to_csv(out / "selective_topk_crc_hc_sensitivity_results.csv", index=False)
    summarize(df, out)


def summarize(df, out):
    df = df.assign(over=(df["test_miss"] > df["eps"]).astype(int))
    summ = df.groupby(["margin", "lgd", "k_fraction", "eps", "priority"]).agg(
        miss_mean=("test_miss", "mean"), miss_sd=("test_miss", "std"),
        calls_mean=("test_call_fraction", "mean"), seeds_over_eps=("over", "sum")).reset_index()
    summ.to_csv(out / "selective_topk_crc_hc_sensitivity_summary.csv", index=False)
    print(summ[summ["eps"] == 0.1].pivot_table(index=["margin", "lgd", "k_fraction"], columns="priority",
                                               values="calls_mean").round(3).to_string())
    print("cells with mean miss > eps:", int((summ["miss_mean"] > summ["eps"]).sum()), "of", len(summ))


if __name__ == "__main__":
    main()

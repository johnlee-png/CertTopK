"""Regenerate the numbers of Tables II and III and the headline statistics from the saved result files.

No model is trained and no LLM is called: everything is read from results/.
Usage: py -X utf8 code/make_tables.py   (from the artifact root)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
SEVEN = ["soccer", "diabetes", "home_credit", "kdd98", "telco", "ieee_fraud", "lending_club"]
KEY = ["domain", "k_fraction", "eps"]


def table2():
    c5 = pd.read_csv(RES / "selective_topk_c5_results.csv")
    k = c5[(c5.k_fraction == 0.10) & ((c5.eps == 0.05) | c5.eps.isna())]
    crc = k[k.method == "CRC"].pivot_table(index="domain", columns="priority", values="test_calls") * 100
    other = k[k.method.isin(["LTT_ENV", "JOINT"])].pivot_table(index="domain", columns="method", values="test_calls") * 100
    nocall = k[k.method.isin(["CHEAP", "DISTILL"])].pivot_table(index="domain", columns="method", values="test_miss") * 100
    ev = pd.read_csv(RES / "selective_topk_everest_crc_results.csv")
    ev = ev[(ev.k_fraction == 0.10) & (ev.eps == 0.05)].groupby("domain").test_call_fraction.mean() * 100
    auto = pd.read_csv(RES / "selective_topk_order_select_final_summary.csv")
    auto = auto[(auto.k_fraction == 0.10) & (auto.eps == 0.05)].set_index("domain").chosen_calls * 100
    t = pd.DataFrame({"P2": crc["P2"], "P3": crc["P3"], "P3c": crc["P3CLIP"], "Ev": ev, "Auto": auto,
                      "LTT": other["LTT_ENV"], "Joint": other["JOINT"], "Cheap": nocall["CHEAP"],
                      "Distill": nocall["DISTILL"]})
    print("Table 2 (kappa = 10%, eps = 0.05; calls %, last two columns mis-selection %)")
    print(t.round(1).to_string())


def table3():
    hc = pd.read_csv(RES / "selective_topk_hc_time_v2.csv")
    print("\nTable 3, Home Credit (seconds, whole test split, median [min, max] of 7 runs)")
    print(hc[(hc.eps.isna()) | (hc.eps == 0.05)][["order", "path", "calls", "overhead_median", "expensive_median",
                                                    "total_median", "total_min", "total_max"]].round(4).to_string(index=False))
    llm = pd.read_csv(RES / "selective_topk_llm_orders.csv")
    print("\nTable 3, LC-LLM (seconds per 500-loan query, mean of 10 test queries)")
    print(llm[["order", "selected_per_query", "priority_s_per_query", "llm_s_per_query", "total_s_per_query"]]
          .round(4).to_string(index=False))


def headline():
    crc = pd.read_csv(RES / "selective_topk_crc_summary.csv")
    crc = crc[crc.domain.isin(SEVEN)]
    print("\nObserved mean loss / eps over 210 settings: max", round((crc.miss_mean / crc.eps).max(), 3),
          "| upper end of 95% t-interval / eps: max", round((crc.miss_ci95_hi / crc.eps).max(), 3))
    s = pd.read_csv(RES / "selective_topk_order_select_final_summary.csv")
    s7 = s[s.domain.isin(SEVEN)]
    print("Auto-calls vs P2: mean saving", round(1 - s7.ratio_chosen_to_P2.mean(), 3),
          "median", round(1 - s7.ratio_chosen_to_P2.median(), 3),
          "| ratio to best fixed order: median", round(s7.ratio_chosen_to_best.median(), 3),
          "| settings with observed mean <= eps:", int((s.chosen_miss <= s.eps).sum()), "of", len(s))
    prof = json.loads((RES / "selective_topk_llm_profile_fit.json").read_text())
    print("LC-LLM Auto-time profiled without test queries: t_e =", round(prof["t_e_s_per_call"], 4), "s;",
          prof["same_choice_as_before"], "of", prof["n_runs"], "choices unchanged")


if __name__ == "__main__":
    table2()
    table3()
    headline()

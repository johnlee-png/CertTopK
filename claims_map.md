# CertTopK — claims → evidence map (for co-authors and the artifact)

Every quantitative claim in the abstract, the contributions, and the conclusion is listed with its comparator, scope,
whether it is measured or estimated, whether set-up cost is included, the table or figure that shows it, the raw result
file, and the script that produces it. Result files are in `results/` of the artifact (`paper/src_extension/` locally).
"Splits" are 20 random re-partitions of one fixed dataset (seeds 0–19).

| # | Claim (wording in paper) | Comparator / scope | Measured or estimated | Set-up included | Table / figure | Raw result file | Script |
|---|---|---|---|---|---|---|---|
| 1 | Observed mean mis-selection rate ≤ ε in every setting (max 0.92ε) | 7 datasets × 3 κ × 2 ε × 5 baseline priorities = 210; mean of 20 split means | measured | n/a | Fig. 4(a), Sec. 5.2 | `selective_topk_crc_summary.csv` | `src_extension_selective_topk_crc.py` |
| 2 | Upper end of 95% t-interval of the 20 split means < ε (max 0.98ε) | same 210 settings; variation over random partitions of a fixed dataset | measured | n/a | Sec. 5.2 | `selective_topk_crc_summary.csv` (`miss_ci95_hi`) | same |
| 3 | LTT: observed mean test loss never > ε; HB and Hoeffding-only pick the same λ̂ | 4,200 (setting, split) pairs; 4,800 incl. LC-LLM | measured | n/a | Sec. 5.2 | `selective_topk_ltt_results.csv` | `..._ltt.py` |
| 4 | Rough-score ranking alone changes 12–54% of the top 10% | 7 datasets, κ = 10% | measured | n/a | Table 2 (Cheap) | `selective_topk_c5_results.csv` (method CHEAP) | `..._baselines.py` |
| 5 | Auto-calls: 9.8% fewer calls than P2 on average (median 3.1%); within 1.5% of the best fixed order (median); target satisfied in 48/48 | P2; 7 datasets × 6 settings (48 incl. LC-LLM); test calls of the order chosen on the fit split | measured calls | no | Table 2 (Auto), Sec. 5.6 | `selective_topk_order_select_final_summary.csv` | `..._order_select_final.py` |
| 6 | Soccer: 30% instead of 63% of candidates refined | Auto-calls (P3c) vs P2, κ = 10%, ε = 0.05 | measured calls | no | Table 2 | `selective_topk_c5_results.csv`, `..._order_select_final_summary.csv` | same |
| 7 | Exact certification with simultaneous intervals needs a median 3.9× (1.4–37×) the calls | P2 + CRC; 7 datasets × 6 settings | measured calls | no | Table 2 (Joint), Sec. 5.7 | `selective_topk_c5_results.csv` (method JOINT) | `..._baselines.py` |
| 8 | Distilled cheap model still mis-selects 7.5–53.5% | no refinement; κ = 10%; 7 datasets + LC-LLM + diabetes (RC) | measured | n/a | Table 2 (Distill) | `selective_topk_c5_results.csv` (DISTILL) | same |
| 9 | Everest-style rule: observed exact-match rate > 0.05 below claim in 11 of 63 settings (max gap 0.29) | 7 datasets × 3 κ × 3 thresholds; pooled | measured | n/a | Fig. 4(b), Sec. 5.3 | `selective_topk_everest_results.csv` | `..._everest_parallel.py` |
| 10 | LLM query 1.5–1.6 s instead of 31–32 s | all-LLM vs P2 + CRC; seed 0, κ = 10%, ε = 0.05; two runs (10 queries each) | measured | **excluded** | Table 3, Sec. 5.11 | `selective_topk_llm_pipeline.json`, `selective_topk_llm_orders.csv` | `..._llm_pipeline.py`, `..._llm_orders.py` |
| 11 | Home Credit: Auto-time (P1) 0.256 s vs Auto-calls (P3c) 0.541 s → 2.1×; 2.6× vs refining all (DuckDB in memory, 0.658 s) | seed 0, κ = 10%, ε = 0.05, whole test split, median of 7 runs | measured | **excluded** | Table 3 | `selective_topk_hc_time_v2.csv` | `..._hc_time_v2.py` |
| 12 | Auto-time chose P1 in all 120 Home Credit runs, estimated 2.0× faster than Auto-calls; choice unchanged when each cost is halved or doubled | 20 seeds × 6 settings; cost model × test calls | **estimated** | no | Sec. 5.10 | `selective_topk_auto_time_home_credit.csv`, `selective_topk_hc_time_v2.json` | inline (see README) |
| 13 | LC-LLM Auto-time profiled without test queries: 119/120 same choices; seed 0 = P2 | fit-split CPU costs; LLM cost on 110 loans outside the data | measured costs | n/a | Sec. 3.6 | `selective_topk_llm_profile_fit.{csv,json}` | `..._llm_profile_fit.py` |
| 14 | Set-up pay-off: Home Credit 1.6 workloads (existing resources) / 17.5 (fresh install); LLM ≈53 / ≈80 queries; 9.9× / 7.9× over 1,000 queries | calibration ± index build, fit outputs, ŝ training | measured components, computed ratio | **included** | Sec. 5.10 | `selective_topk_hc_time_v2.json`, `selective_topk_llm_orders.csv` | inline |
| 15 | Time-ordered IEEE-CIS: mean test loss ≤ ε in 12/12 settings; quarter max 0.052 at ε = 0.05 | one time-ordered split, P2 and P3c | measured | n/a | Sec. 5.8 | `selective_topk_temporal_ieee.csv` | `..._temporal.py` |
| 16 | Raw-loss LTT chose the same threshold as envelope LTT in 99.4% of runs | all settings, 9 workloads | measured | n/a | Sec. 5.7 | `selective_topk_c5_results.csv` | `..._baselines.py` |
| 17 | Tail control: share of queries with loss > 0.1 held at 3.2–9.4% | all κ, α = 0.1 | measured | n/a | Sec. 5.7 | `selective_topk_c5_results.csv` (CRC_TAIL) | same |
| 18 | P3c equal-call: soccer 0.05 with 29%, 0.01 with 43% (P2 57%, 86%) | κ = 10%, mean of 20 splits | measured | no | Fig. 5 | `selective_topk_p3c_curves.csv`, `selective_topk_pareto_curves.csv` | `..._p3c_curves.py`, `..._pareto.py` |
| 19 | The two estimates share only 46–88% of the top 10% (48% soccer, 46% KDD98) | 1 − cheap-only mis-selection; κ = 10%, 7 datasets | measured | n/a | Sec. 1, Table 2 (Cheap) | `selective_topk_c5_results.csv` (CHEAP) | `..._baselines.py` |
| 20 | Expensive path ≈46× the cheap path (soccer, 360 frames); ≈9.3×10⁴× (LLM, 60 ms vs 0.65 µs) | per-candidate wall clock | measured | n/a | Secs. I, III-F, V-J, V-K | `selective_topk_soccer_cost.json`, `selective_topk_lc_llm_cost.json` | `..._lc_llm_cost.py`, inline |
| 21 | MDN uncertainty model: gap > 0.05 in 9/9 KDD98 settings, 1/9 soccer, 0 Lending Club and Home Credit | Everest-style rule with a 3-component MDN; pooled over splits | measured | n/a | Sec. 5.3 | `selective_topk_everest_mdn_summary.csv` | `..._everest_mdn.py` |
| 22 | Everest order + CRC: target satisfied in 42/42; fewer calls than P2 on soccer at ε = 0.05, more on HC, IEEE, LC, KDD98 | 7 datasets × 6 settings | measured calls | no | Sec. 5.6, Table 2 (Ev) | `selective_topk_everest_crc_results.csv` | `..._everest_crc.py` |

Note on #13 and Sec. 5.10 ("same order in 13 of 20 seeds; 44 of 120 runs overall"): these counts compare LC-LLM
Auto-time with Auto-calls and come from `selective_topk_llm_profile_fit.csv` (fit-split profile; column comparing the Auto-time choice with the Auto-calls choice).

Definition checks (boundary cases, CRC/LTT rules, P3 streams): `check_definitions.py`, 16/16 pass.

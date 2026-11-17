# CertTopK: Certified Top-k Query Processing with Selective Inference

Code, results, and scripts for the paper submitted to PVLDB Vol. 20 (VLDB 2027).

## Authors

John Heong Lee<sup>1,*</sup>, Dongho Park<sup>1,2</sup>, Ahyun Kim<sup>3</sup>, Charles Cheolgi Lee<sup>4</sup>, Dohhoon Kim<sup>3</sup>

1. Graduate School of Biomedical Science and Engineering, Inha University, Incheon, Republic of Korea
2. Department of Sports Science, Inha University, Incheon, Republic of Korea
3. Department of Industrial and Management Engineering, Inha University, Incheon, Republic of Korea
4. Postal Innovation AI Research Team, Korea Postal Service Agency, Republic of Korea

\* Corresponding author: johnlee@inha.edu

## Overview

This artifact lets a reader regenerate the tables and figures of the paper at three levels of effort.

1. **From saved results (minutes, no data needed).** `code/make_tables.py` prints the numbers of Table 2, the measured columns
   of Table 3, and the headline statistics; `code/make_paper_figs.py` and `code/make_concept_figs.py` redraw Figs. 1–5 into `figures/`.
   The LLM outputs are saved (`results/selective_topk_llm_scores.csv`), so no LLM call is needed.
2. **From the public data (hours).** Download the data sets below and rerun the experiment scripts listed per table.
3. **Checks.** `code/check_definitions.py` runs 16 boundary-case checks of the definitions against the code (all pass);
   `code/export_splits.py` regenerates the splits and their SHA-256 hashes (`results/splits/summary.csv`).

`claims_map.md` maps every quantitative claim of the paper to its table or figure, raw result file, and script, and
states whether the number is measured or estimated and whether set-up cost is included.

## Environment

`pip install -r requirements.txt`. Python 3.14, NumPy 2.5.1, pandas 3.0.3, scikit-learn 1.9.0, LightGBM 4.6.0, DuckDB, SciPy, matplotlib with LaTeX
(figures use `text.usetex` with Linux Libertine, the font of the PVLDB template); 12 logical CPUs; the LLM case uses one 8 GB GPU. The experiment scripts write to
`paper/src_extension`; run them from the artifact root after linking it to `results/`
(`mkdir paper && mklink /J paper\src_extension results` on Windows, `mkdir -p paper && ln -s ../results paper/src_extension` elsewhere).

## Data (not redistributed)

| Data set | Source | Expected location | Preprocessing |
|---|---|---|---|
| Soccer | StatsBomb Open Data (github.com/statsbomb/open-data): all 323 matches with 360 freeze frames — World Cup 2022 (64), Women's World Cup 2023 (64), UEFA Euro 2020 (51) and 2024 (51), Bundesliga 2023/24 (34), UEFA Women's Euro 2025 (31) and 2022 (28) | `data/raw/statsbomb-open-data/` | `code/src/pipeline.py` writes `data/processed/passes_with_features.parquet`, `X_event.parquet`, `X_full.parquet`, `y_labels.parquet` |
| Diabetes 130-US hospitals | UCI | `data/raw/diabetes/diabetic_data.csv` | none |
| Home Credit Default Risk | Kaggle | `data/raw/home_credit/` | `_clean` copies written by `..._homecredit.py` |
| KDD Cup 1998 | UCI (`cup98LRN.txt`) | `data/raw/kddcup98/` | none |
| Telco Customer Churn | IBM / Kaggle | `data/raw/telco_churn/Telco-Customer-Churn.csv` | none |
| IEEE-CIS Fraud Detection | Kaggle | `data/raw/ieee_fraud/` | `train_clean.parquet` (transaction joined with identity) |
| Lending Club 2007–2018Q4 | Kaggle (`accepted_2007_to_2018Q4.csv.gz`) | `data/raw/lending_club/` | `accepted_clean.parquet` (finished loans) |

Raw files are never modified.

## Configuration

All settings are in `code/src_extension_selective_topk_crc.py` (one `build_*` function per data set) unless noted.

| Item | Setting |
|---|---|
| Cheap / expensive model | LightGBM classifier, at most 500 trees, learning rate 0.05, depth 6, min_child_samples 50, early stopping (50 rounds) on the fit split (`code/diabetes_new_regret_diagnostic.py: train_lgb`); features per data set in Table 1 and in the `build_*` functions (`LC_LISTING`, `LC_BUREAU`, `TELCO_ACCOUNT`, `TELCO_SERVICES`, `KDD_HISTORY*`, ...) |
| Error-size model ŝ | LightGBM regressor of \|Pe − Pc\| on (cheap features, Pc, R, C): 300 trees, learning rate 0.05, 31 leaves, min_child_samples 50; predictions floored at the 5% quantile of \|Pe − Pc\| on the fit split + 1e-6 (`fit_scale`) |
| Grid | 201 quantiles (levels 0, 0.005, ..., 1) of the priority over the fit candidates, plus −∞ and +∞ |
| Splits | by group (match, patient, applicant, donor, customer, card holder, loan): 50% train, 10% fit, 20% calibration, 20% test; seeds 0–19 (`group_split`, `batched_cal_test`) |
| Queries | one match (soccer); batches of 500 (diabetes, LC-LLM), 1,000 (Home Credit, IEEE-CIS), 250 (KDD98), 50 (Telco), 5,000 (Lending Club) rows, formed on the pooled calibration+test groups and assigned to calibration or test uniformly at random |
| Targets | κ ∈ {5, 10, 20}%, ε ∈ {0.05, 0.10}; LTT δ = 0.1 |
| P3 / P3c | B = 200 scenarios; residuals resampled i.i.d. from the fit split; one pseudo-random stream per split; P3c clips simulated probabilities to [0, 1] |
| CRC | smallest grid point with (m/(m+1))·mean envelope + 1/(m+1) ≤ ε, else +∞ |
| LTT | p = min{exp(−m·KL(r̂‖ε)), e·P[Bin(m,ε) ≤ ⌈m r̂⌉]}; Hoeffding-only = first term; fixed-sequence testing from +∞ (`..._ltt.py`) |
| Everest re-implementation | `..._everest.py`, `..._everest_parallel.py`, `..._everest_crc.py`, `..._everest_mdn.py` |
| Automatic order choice | `..._order_select_final.py` (Auto-calls); Auto-time costs from `..._hc_time_v2.py` (Home Credit, fit-split profile) and `..._llm_profile_fit.py` (LC-LLM, fit split and 110 loans outside the data) |

## Splits and queries

`results/splits/summary.csv`: per data set and seed, rows and groups per split, number and size of calibration and test
queries, and the SHA-256 of the assignment. `results/splits/<dataset>_seed0.csv.gz`: the full assignment for seed 0
(row, group id, split, query id). `results/soccer_match_query_split.csv`: match → competition → split for all 20 seeds.

## Reproducing the paper

| Paper item | Script (`py -X utf8 code/<script> 20 <datasets> <workers>`) | Result files |
|---|---|---|
| Fig. 4(a), Sec. 5.2 | `src_extension_selective_topk_crc.py`, `..._ltt.py` | `selective_topk_crc_*`, `selective_topk_ltt_*` |
| Fig. 4(b), Sec. 5.3 | `..._everest_parallel.py`, `..._everest_mdn.py` | `selective_topk_everest_*` |
| Fig. 5, Sec. 5.4 | `..._pareto.py`, `..._p3_extra.py`, `..._p3c_curves.py` | `selective_topk_pareto_curves.csv`, `selective_topk_p3extra_curves.csv`, `selective_topk_p3c_curves.csv` |
| Table 2, Secs. V-E–G | `..._everest_crc.py`, `..._lossaware.py`, `..._baselines.py`, `..._order_select_final.py` | `selective_topk_c5_*`, `selective_topk_everest_crc_*`, `selective_topk_order_select_final_*` |
| P3 ablations | `..._p3_extra.py` | `selective_topk_p3extra_*` |
| Sec. 5.8 | `..._m_sensitivity.py`, `..._temporal.py`, `..._calfit.py` | `selective_topk_m_sensitivity_*`, `selective_topk_temporal_ieee.csv`, `selective_topk_calfit_*` |
| Sec. 5.9 | `..._realized.py`, `..._predicate.py`, `..._crc_hc_sensitivity.py` | matching `selective_topk_*` files |
| Table 3, Sec. 5.10 | `..._hc_time_v2.py`, `..._llm_orders.py`, `..._llm_pipeline.py`, `..._llm_profile_fit.py` | `selective_topk_hc_time_v2.*`, `selective_topk_llm_orders.csv`, `selective_topk_llm_pipeline.json`, `selective_topk_llm_profile_fit.*`, `selective_topk_auto_time_*.csv` |
| Table 4, Sec. 5.11 | `..._llm_trial.py`, `..._llm_score.py`, `..._lc_llm_cost.py` | `selective_topk_llm_*`, `selective_topk_lc_llm_cost.json` |
| Figures | `make_paper_figs.py`, `make_concept_figs.py` | `figures/*.pdf` |

In `selective_topk_hc_time_v2.json` the profiled priority costs `pri_P2`, `pri_P3`, `pri_P3c` include one evaluation of
ŝ; the paper's Model column subtracts it (own priority cost = `pri_*` − `scale`).

## LLM case

Qwen2.5-7B-Instruct, Q4_K_M GGUF, served by llama.cpp:
`llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf -ngl 99 -c 16384 -np 8 --port 8091`. The prompt and the read-out (sum
of the probabilities of "Yes"/"No" surface tokens among the top 20 of the first answer token, temperature 0) are
`prompt` and `ask` in `code/src_extension_selective_topk_llm_trial.py`; eight concurrent requests. The scores of all
125,281 loans are saved in `results/selective_topk_llm_scores.csv` (`id, p_llm_yes, prompt_tokens`), so the LC-LLM
workload and its tables can be rebuilt without running the LLM; only the wall-clock measurements of Table 3 need the
server.

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

1. **From saved results (minutes, no data needed).** `code/make_tables.py` prints the numbers of Table 2 and the headline
   statistics; `code/src_vldb_e2e_hc_report.py hc` and `... llm_c1` print Table 3 (end-to-end times) from `results/e2e/`;
   `code/src_vldb_auto_analysis.py` prints the comparison of fixed orders, the automatic choice, and the best order in
   hindsight; `code/make_paper_figs.py` and `code/make_concept_figs.py` redraw the figures into `figures/`.
   The LLM outputs are saved (`results/selective_topk_llm_scores.csv`, `results/selective_topk_llm_scores_c1.csv`), so no
   LLM call is needed.
2. **From the public data (hours).** Download the data sets below and rerun the experiment scripts listed per table.
3. **Checks.** `proofs.pdf` gives the full statements and proofs of the structural results (Secs. A.1–A.3; Results A.1, A.3,
   A.6 are Theorems 3–5 of the paper, the others its supporting results). `code/check_definitions.py` runs 16 boundary-case checks of the definitions against the code (all pass);
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
| P4 / P5 / P5T | same scenarios as P3c (`src_vldb_p4.py`). P5 refines in decreasing probability that a candidate's cheap score lies on the wrong side of the scenario's expensive cut-off (Theorem 5: the loss of every refined set is at most the number of unrefined wrong-side candidates / k); P5T stops each query when the remaining bound is small (Prop. A.8); P4 (one-step expected loss reduction) is a negative control |
| P3 / P3c | B = 200 scenarios; residuals resampled i.i.d. from the fit split; one pseudo-random stream per split; P3c clips simulated probabilities to [0, 1] |
| CRC | smallest grid point with (m/(m+1))·mean envelope + 1/(m+1) ≤ ε, else +∞ |
| LTT | p = min{exp(−m·KL(r̂‖ε)), e·P[Bin(m,ε) ≤ ⌈m r̂⌉]}; Hoeffding-only = first term; fixed-sequence testing from +∞ (`..._ltt.py`) |
| Everest re-implementation | `..._everest.py`, `..._everest_parallel.py`, `..._everest_crc.py`, `..._everest_mdn.py` |
| Automatic order choice | Auto-calls: smallest fit-split plug-in calls among P0, PU, P1, TOPM, P2, P3, P3c, P5, and Everest's order (`..._order_select_final.py`, `src_vldb_auto_extended.py`). Auto-time: smallest predicted time `overhead + a + b·calls` among P0, PU, P1, TOPM, P2, P3c, P5, and refining everyone, with the costs profiled on fit queries of each configuration (`src_vldb_e2e_hc.py: profile, choose`) |
| Ties | every top-k, including P3's simulated scenarios, returns exactly k candidates; ties at the cut-off go to the smaller index (`crc.topk`, `src_vldb_topk_ties.topk_mask_batch`) |

## Splits and queries

`results/splits/summary.csv`: per data set and seed, rows and groups per split, number and size of calibration and test
queries, and the SHA-256 of the assignment. `results/splits/<dataset>_seed0.csv.gz`: the full assignment for seed 0
(row, group id, split, query id). `results/soccer_match_query_split.csv`: match → competition → split for all 20 seeds.

## Reproducing the paper

| Paper item | Script (`py -X utf8 code/<script> 20 <datasets> <workers>`) | Result files |
|---|---|---|
| Fig. 3(a), Sec. 5.2 | `src_extension_selective_topk_crc.py`, `..._ltt.py` | `selective_topk_crc_*`, `selective_topk_ltt_*` |
| Fig. 3(b), Sec. 5.3 | `..._everest_parallel.py`, `..._everest_mdn.py` | `selective_topk_everest_*` |
| Equal-call curves (`figures/fig_pareto.pdf`, artifact only), Sec. 5.4 | `..._pareto.py`, `..._p3_extra.py`, `..._p3c_curves.py`, `src_vldb_p4.py` | `selective_topk_pareto_curves.csv`, `selective_topk_p3extra_curves.csv`, `selective_topk_p3c_curves.csv` |
| Table 2, Secs. 5.5–5.7 | `..._everest_crc.py`, `..._lossaware.py`, `..._baselines.py`, `..._order_select_final.py` | `selective_topk_c5_*`, `selective_topk_everest_crc_*`, `selective_topk_order_select_final_*` |
| P5, P4, P5T vs P3c (Table 2, Sec. 5.6) | `src_vldb_p4.py 20 <datasets> <workers>`; P5T: `P4_ORDERS=P3CLIP,P5,P5T P4_TAG=p5t` | `selective_topk_p4_*`, `selective_topk_p5t_*` |
| Automatic choice with P0, PU, and P5, Sec. 5.6 | `src_vldb_auto_extended.py`, then `src_vldb_auto_analysis.py` | `selective_topk_auto_ext_fitcalls.csv`, `auto_analysis_{cells,summary,overall,worst}.csv` |
| Ties, Sec. 5.1 | reruns with `CERTTOPK_TIE_LOG` set, then `src_vldb_summarize_ties.py` | `tie_rates.csv` |
| P3 ablations | `..._p3_extra.py` | `selective_topk_p3extra_*` |
| Realized utility, Sec. 5.8 | `..._realized.py`, then `src_vldb_realized_stats.py` (paired over seeds) | `selective_topk_realized_*` |
| Sec. 5.8 | `..._m_sensitivity.py`, `..._temporal.py`, `..._calfit.py`, `..._realized.py`, `..._predicate.py`, `..._crc_hc_sensitivity.py` | matching `selective_topk_*` files |
| Table 3, Sec. 5.9 (seeds via `CERTTOPK_SEED`; across-seed summary `src_vldb_e2e_seeds.py` → `e2e/e2e_seeds_*.csv`) | `src_vldb_e2e_hc.py` (Home Credit: sizes 250/1,000/2,500 × index / DuckDB in memory / DuckDB over Parquet), `src_vldb_e2e_llm.py` (LC-LLM: sizes 100/250/500 × 1/4/8 concurrent requests), `src_vldb_e2e_hc_report.py` | `results/e2e/hc_*`, `results/e2e/llm_c1_*` |
| LLM determinism, Sec. 5.9 | `src_vldb_llm_determinism.py`, `src_vldb_llm_rescore_c1.py` | `results/e2e/llm_determinism.csv`, `selective_topk_llm_scores_c1.csv` |
| Sec. 5.10 | `..._llm_trial.py`, `..._llm_score.py`, `..._lc_llm_cost.py` | `selective_topk_llm_*`, `selective_topk_lc_llm_cost.json` |
| Figures | `make_paper_figs.py`, `make_concept_figs.py` | `figures/*.pdf` |

`src_vldb_e2e.py` holds the single per-query function `answer` used for every timed number (cheap model → priority →
selection → data access → expensive model → mixed top-k); refining everyone (ALL) runs through the same function and
access path but skips cheap inference, which it does not need (it reads the inputs and runs only the expensive path),
and the loss of each timed answer is recorded in the same loop. Earlier timing scripts (`..._hc_time_v2.py`,
`..._hc_time_breakdown.py`, `..._llm_orders.py`, `..._llm_pipeline.py`, `..._llm_profile_fit.py`, and their result
files, including `selective_topk_auto_time_*.csv`) are superseded by this protocol and kept for reference only.
`results/e2e/llm_queries.csv`, `llm_summary.csv`, and `llm_moretest_queries.csv` are an earlier run against the logged
concurrency-8 scores, superseded by `llm_c1_*`.

## LLM case

Qwen2.5-7B-Instruct, Q4_K_M GGUF, served by llama.cpp:
`llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf -ngl 99 -c 16384 -np 8 --port 8091`. The prompt and the read-out (sum
of the probabilities of "Yes"/"No" surface tokens among the top 20 of the first answer token, temperature 0) are
`prompt` and `ask` in `code/src_extension_selective_topk_llm_trial.py`. The scores of all 125,281 loans are saved
(`id, p_llm_yes, prompt_tokens`), so the LC-LLM workload and its tables can be rebuilt without running the LLM; only
the wall-clock measurements of Table 3 need the server.

* `results/selective_topk_llm_scores.csv`: one scoring pass with eight concurrent requests, the oracle of the 20-seed
  experiments (Sec. 5.10; refining reads these stored scores, so full refinement reproduces the reference).
* `results/selective_topk_llm_scores_c1.csv`: one request at a time. The server is deterministic only then
  (`results/e2e/llm_determinism.csv`: repeated runs differ by 0.40 in mean absolute logit at concurrency 8, by 0 at
  concurrency 1), so the end-to-end runs train, calibrate, and define the reference with these scores
  (`CERTTOPK_LLM_SCORES=paper/src_extension/selective_topk_llm_scores_c1.csv`); at concurrency 1 the online outputs
  equal the stored ones and refining everyone has loss 0. Runs at concurrency 4 and 8 measure speed and robustness
  against the same reference.

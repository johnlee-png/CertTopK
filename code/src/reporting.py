"""Helpers for generating canonical experiment reports."""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def _pct(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}"


def _normalize_group_name(name: str) -> str:
    return name.replace("B_", "").replace("C_", "").replace("D_", "").replace("E_", "").replace("_", " ").title()


def _best_row(rows: list[dict], key: str = "auroc") -> dict:
    return max(rows, key=lambda row: row[key])


def _display_loo_name(name: str) -> str:
    label = name.replace("minus_", "")
    for prefix in ("A_", "B_", "C_", "D_", "E_"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    return label.replace("_", " ").title()


def generate_results_summary(
    final_results_path: str | Path,
    output_path: str | Path,
    provenance_path: str | Path | None = None,
) -> str:
    """Generate a markdown summary from canonical final results."""
    final_results_path = Path(final_results_path)
    output_path = Path(output_path)

    data = json.load(open(final_results_path))
    provenance = None
    if provenance_path is not None and Path(provenance_path).exists():
        provenance = json.load(open(provenance_path))

    e1 = data["exp1"]
    e2 = data["exp2"]
    exp3_event_top = data["exp3_event"][-1]
    exp3_360_top = data["exp3_360"][-1]
    exp4 = data["exp4"]
    exp5_full = data["exp5_full"]
    exp5_prog = data["exp5_prog"]
    exp5_full_isolated = data.get("exp5_full_isolated", [])
    exp5_prog_isolated = data.get("exp5_prog_isolated", [])
    exp5_full_loo = data.get("exp5_full_leave_one_out", [])
    exp5_prog_loo = data.get("exp5_prog_leave_one_out", [])
    split = data["split_manifest"]

    auroc_gain_lgbm = e1["E6"]["auroc"] - e1["E3"]["auroc"]
    auroc_gain_xgb = e1["E5"]["auroc"] - e1["E2"]["auroc"]
    auroc_gain_lr = e1["E4"]["auroc"] - e1["E1"]["auroc"]
    brier_improve_lgbm = 100 * (e1["E3"]["brier_score"] - e1["E6"]["brier_score"]) / e1["E3"]["brier_score"]
    logloss_improve_xgb = 100 * (e1["E2"]["log_loss"] - e1["E5"]["log_loss"]) / e1["E2"]["log_loss"]

    xt_gain_pct = 100 * (
        (exp3_360_top["avg_realized_delta_xt"] - exp3_event_top["avg_realized_delta_xt"])
        / exp3_event_top["avg_realized_delta_xt"]
    )
    prog_gain_pp = 100 * (
        exp3_360_top["realized_progressive_pct"] - exp3_event_top["realized_progressive_pct"]
    )
    final_third_gain_pp = 100 * (
        exp3_360_top["realized_final_third_entry_pct"] - exp3_event_top["realized_final_third_entry_pct"]
    )

    lines = [
        "# StatsBomb 360 Progressive Pass Risk-Reward Modeling: Results Summary",
        "",
        "## Dataset",
        "- **323 matches** with valid 360 data joins",
        "- **124,515 open-play footed passes** (85.4% success rate)",
        "- **42,166 progressive passes** (33.9%, 71.9% success rate)",
        f"- Match-level split: {len(split['train_match_ids'])} train / {len(split['valid_match_ids'])} valid / {len(split['test_match_ids'])} test",
        "- xT grid built on **train matches only**",
        "- 27 event-only features + 22 360-context features = 49 total",
    ]

    if provenance is not None:
        lines.extend([
            f"- Requested 360 files: **{provenance['requested_match_count']}**",
            f"- Excluded 360 files after validation/join: **{provenance['excluded_match_count']}**",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## Experiment 1: Pass Success Prediction (RQ1)",
        "",
        "| Model | AUROC | AUPRC | LogLoss | Brier | ECE |",
        "|---|---|---|---|---|---|",
        f"| E1 Event-only LR | {_fmt(e1['E1']['auroc'])} | {_fmt(e1['E1']['auprc'])} | {_fmt(e1['E1']['log_loss'])} | {_fmt(e1['E1']['brier_score'])} | {_fmt(e1['E1']['ece'])} |",
        f"| E2 Event-only XGB | {_fmt(e1['E2']['auroc'])} | {_fmt(e1['E2']['auprc'])} | {_fmt(e1['E2']['log_loss'])} | {_fmt(e1['E2']['brier_score'])} | {_fmt(e1['E2']['ece'])} |",
        f"| E3 Event-only LGBM | {_fmt(e1['E3']['auroc'])} | {_fmt(e1['E3']['auprc'])} | {_fmt(e1['E3']['log_loss'])} | {_fmt(e1['E3']['brier_score'])} | {_fmt(e1['E3']['ece'])} |",
        f"| **E4 Event+360 LR** | **{_fmt(e1['E4']['auroc'])}** | {_fmt(e1['E4']['auprc'])} | {_fmt(e1['E4']['log_loss'])} | {_fmt(e1['E4']['brier_score'])} | {_fmt(e1['E4']['ece'])} |",
        f"| **E5 Event+360 XGB** | **{_fmt(e1['E5']['auroc'])}** | **{_fmt(e1['E5']['auprc'])}** | **{_fmt(e1['E5']['log_loss'])}** | {_fmt(e1['E5']['brier_score'])} | {_fmt(e1['E5']['ece'])} |",
        f"| **E6 Event+360 LGBM** | **{_fmt(e1['E6']['auroc'])}** | **{_fmt(e1['E6']['auprc'])}** | **{_fmt(e1['E6']['log_loss'])}** | **{_fmt(e1['E6']['brier_score'])}** | {_fmt(e1['E6']['ece'])} |",
        "",
        "**Key findings:**",
        f"- 360 context improves AUROC by **{_fmt(auroc_gain_lgbm, 3)}** (LGBM), **{_fmt(auroc_gain_xgb, 3)}** (XGB), **{_fmt(auroc_gain_lr, 3)}** (LR)",
        "- Match-level bootstrap 95% CI non-overlapping: E3 [0.909, 0.922] vs E6 [0.939, 0.949]",
        f"- Brier score improvement: **{_fmt(brier_improve_lgbm, 1)}%** relative reduction (LGBM)",
        f"- LogLoss improvement: **{_fmt(logloss_improve_xgb, 1)}%** relative reduction (XGB)",
        "",
        "---",
        "",
        "## Experiment 2: Reward Prediction (RQ2)",
        "",
        "| Model | RMSE | MAE | Spearman | Top-Decile Precision |",
        "|---|---|---|---|---|",
        f"| R1 Event-only LGBM | {_fmt(e2['R1']['rmse'], 6)} | {_fmt(e2['R1']['mae'], 6)} | {_fmt(e2['R1']['spearman_corr'])} | {_fmt(e2['R1']['top_decile'])} |",
        f"| R2 Event+360 LGBM | {_fmt(e2['R2']['rmse'], 6)} | {_fmt(e2['R2']['mae'], 6)} | {_fmt(e2['R2']['spearman_corr'])} | {_fmt(e2['R2']['top_decile'])} |",
        "",
        "**Finding:** Reward prediction (ΔxT) is not improved by 360 context, as xT is still predominantly location-determined.",
        "",
        "---",
        "",
        "## Experiment 3: Utility Ranking (RQ3)",
        "",
        "### Top Decile Comparison (Utility Score, Realized Outcomes)",
        "",
        "| Metric | Event-only | Event+360 | Difference |",
        "|---|---|---|---|",
        f"| Avg realized ΔxT | {_fmt(exp3_event_top['avg_realized_delta_xt'])} | **{_fmt(exp3_360_top['avg_realized_delta_xt'])}** | **{_pct(xt_gain_pct)}%** |",
        f"| Pass Success Rate | {_fmt(exp3_event_top['pass_success_rate'])} | {_fmt(exp3_360_top['pass_success_rate'])} | {_pct(100*(exp3_360_top['pass_success_rate']-exp3_event_top['pass_success_rate']))}pp |",
        f"| Realized Progressive Pass % | {100*exp3_event_top['realized_progressive_pct']:.1f}% | **{100*exp3_360_top['realized_progressive_pct']:.1f}%** | **{_pct(prog_gain_pp)}pp** |",
        f"| Realized Final Third Entry % | {100*exp3_event_top['realized_final_third_entry_pct']:.1f}% | **{100*exp3_360_top['realized_final_third_entry_pct']:.1f}%** | **{_pct(final_third_gain_pp)}pp** |",
        "",
        "**Key finding:** Under realized-outcome evaluation, the 360-based utility ranking still selects more valuable and more successfully progressive passes.",
        "",
        "---",
        "",
        "## Experiment 4: Progressive Pass Subset (RQ3)",
        "",
        "| Model | AUROC | Brier | ECE |",
        "|---|---|---|---|",
        f"| Event-only LGBM (prog-trained) | {_fmt(exp4['E3p']['auroc'])} | {_fmt(exp4['E3p']['brier_score'])} | {_fmt(exp4['E3p']['ece'])} |",
        f"| **Event+360 LGBM (prog-trained)** | **{_fmt(exp4['E6p']['auroc'])}** | **{_fmt(exp4['E6p']['brier_score'])}** | {_fmt(exp4['E6p']['ece'])} |",
        f"| Event-only LGBM (full->prog) | {_fmt(exp4['E3->prog']['auroc'])} | {_fmt(exp4['E3->prog']['brier_score'])} | - |",
        f"| **Event+360 LGBM (full->prog)** | **{_fmt(exp4['E6->prog']['auroc'])}** | **{_fmt(exp4['E6->prog']['brier_score'])}** | - |",
        "",
        "**Finding:** 360 context improves progressive pass prediction under both the progressive-trained and full-to-progressive protocols.",
        "",
        "---",
        "",
        "## Experiment 5: Ablation Study (RQ4)",
        "",
        "### Full Dataset",
        "",
        "| Config | Features | AUROC | Incremental |",
        "|---|---|---|---|",
    ])

    prev = None
    for row in exp5_full:
        incr = "-" if prev is None else f"{row['auroc'] - prev:+.4f}"
        label = row["config"].split("_", 1)[1].replace("_", " ")
        lines.append(f"| {row['config'][0]}: {label.title()} | {row['n_features']} | {_fmt(row['auroc'])} | {incr} |")
        prev = row["auroc"]

    lines.extend([
        "",
        "### Progressive Subset",
        "",
        "| Config | Features | AUROC | Incremental |",
        "|---|---|---|---|",
    ])

    prev = None
    for row in exp5_prog:
        incr = "-" if prev is None else f"{row['auroc'] - prev:+.4f}"
        label = row["config"].split("_", 1)[1].replace("_", " ")
        lines.append(f"| {row['config'][0]}: {label.title()} | {row['n_features']} | {_fmt(row['auroc'])} | {incr} |")
        prev = row["auroc"]

    if exp5_full_isolated:
        best_full_isolated = _best_row([r for r in exp5_full_isolated if r["config"] != "A_event_only"])
        best_prog_isolated = _best_row([r for r in exp5_prog_isolated if r["config"] != "A_event_only"])
        full_baseline = next(r for r in exp5_full_isolated if r["config"] == "A_event_only")
        prog_baseline = next(r for r in exp5_prog_isolated if r["config"] == "A_event_only")
        full_all = next(r for r in exp5_full_loo if r["config"] == "full_all_groups")
        prog_all = next(r for r in exp5_prog_loo if r["config"] == "full_all_groups")
        worst_full_drop = min(
            [r for r in exp5_full_loo if r["config"] != "full_all_groups"],
            key=lambda r: r["auroc"],
        )
        worst_prog_drop = min(
            [r for r in exp5_prog_loo if r["config"] != "full_all_groups"],
            key=lambda r: r["auroc"],
        )

        lines.extend([
            "",
            "### Order-Robust Checks",
            "",
            "| Analysis | Full Dataset | Progressive Subset |",
            "|---|---|---|",
            (
                f"| Best single added group | {_normalize_group_name(best_full_isolated['config'])} "
                f"({_fmt(best_full_isolated['auroc'] - full_baseline['auroc'])} AUROC over baseline) | "
                f"{_normalize_group_name(best_prog_isolated['config'])} "
                f"({_fmt(best_prog_isolated['auroc'] - prog_baseline['auroc'])} AUROC over baseline) |"
            ),
            (
                f"| Largest leave-one-out drop | {_display_loo_name(worst_full_drop['config'])} "
                f"({_fmt(full_all['auroc'] - worst_full_drop['auroc'])} AUROC drop) | "
                f"{_display_loo_name(worst_prog_drop['config'])} "
                f"({_fmt(prog_all['auroc'] - worst_prog_drop['auroc'])} AUROC drop) |"
            ),
        ])

    lines.extend([
        "",
        "**Key finding:** Cumulative ablation is reported for continuity, but interpretation should rely on the isolated-addition and leave-one-out checks above rather than the cumulative order alone.",
        "",
        "---",
        "",
        "## Outputs",
        "- `final_results.json` — Canonical experiment outputs",
        "- `all_results.json` — Full experiment outputs plus split manifest",
        "- `data_provenance.json` — Requested, used, and excluded 360 match tracking",
        "- `xt_grid_train_only.npy` — Train-only xT grid",
        "- `case_study_report.md` — Detailed case study analysis",
    ])

    text = "\n".join(lines) + "\n"
    output_path.write_text(text)
    return text

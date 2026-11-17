"""Experiment 6: Qualitative tactical case study analysis."""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def select_case_study_passes(
    df: pd.DataFrame,
    utility_event: np.ndarray,
    utility_360: np.ndarray,
    n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Select passes for qualitative case study analysis.

    Returns dict with four categories:
    - top_utility: Highest utility (360 model) passes
    - bottom_utility: Lowest utility (360 model) passes
    - event_high_360_low: Event model rates high, 360 model rates low
    - event_low_360_high: Event model rates low, 360 model rates high
    """
    df = df.copy()
    df["utility_event"] = utility_event
    df["utility_360"] = utility_360
    df["utility_diff"] = utility_360 - utility_event

    # Percentile ranks
    df["event_pctile"] = df["utility_event"].rank(pct=True)
    df["f360_pctile"] = df["utility_360"].rank(pct=True)

    cases = {}

    # Top utility (360 model)
    cases["top_utility"] = (
        df.nlargest(n, "utility_360")
        [["start_x", "start_y", "end_x", "end_y",
          "utility_event", "utility_360", "utility_diff",
          "is_progressive", "pass_outcome_name", "delta_xt"]]
    )

    # Bottom utility (360 model)
    cases["bottom_utility"] = (
        df.nsmallest(n, "utility_360")
        [["start_x", "start_y", "end_x", "end_y",
          "utility_event", "utility_360", "utility_diff",
          "is_progressive", "pass_outcome_name", "delta_xt"]]
    )

    # Event high but 360 low (360 context reveals hidden risk)
    event_high_360_low = df[
        (df["event_pctile"] > 0.8) & (df["f360_pctile"] < 0.3)
    ].nsmallest(n, "utility_diff")
    cases["event_high_360_low"] = event_high_360_low[
        ["start_x", "start_y", "end_x", "end_y",
         "utility_event", "utility_360", "utility_diff",
         "is_progressive", "pass_outcome_name", "delta_xt"]
    ]

    # Event low but 360 high (360 context reveals hidden value)
    event_low_360_high = df[
        (df["event_pctile"] < 0.3) & (df["f360_pctile"] > 0.8)
    ].nlargest(n, "utility_diff")
    cases["event_low_360_high"] = event_low_360_high[
        ["start_x", "start_y", "end_x", "end_y",
         "utility_event", "utility_360", "utility_diff",
         "is_progressive", "pass_outcome_name", "delta_xt"]
    ]

    return cases


def generate_case_study_report(
    cases: dict[str, pd.DataFrame],
    output_path: str,
) -> str:
    """Generate a markdown case study report."""
    lines = ["# Tactical Case Study Analysis\n"]

    titles = {
        "top_utility": "Top Utility Passes (360 Model)",
        "bottom_utility": "Bottom Utility Passes (360 Model)",
        "event_high_360_low": "Event-High but 360-Low: Hidden Risk Cases",
        "event_low_360_high": "Event-Low but 360-High: Hidden Value Cases",
    }

    for key, title in titles.items():
        lines.append(f"\n## {title}\n")
        if key in cases and len(cases[key]) > 0:
            df = cases[key]
            lines.append(f"Number of cases: {len(df)}\n")
            lines.append(df.to_markdown(index=False))
            lines.append("")

            # Summary statistics
            lines.append(f"\n**Summary:**")
            lines.append(f"- Mean utility (event): {df['utility_event'].mean():.4f}")
            lines.append(f"- Mean utility (360): {df['utility_360'].mean():.4f}")
            lines.append(f"- Progressive pass %: {df['is_progressive'].mean()*100:.1f}%")
            if "delta_xt" in df.columns:
                lines.append(f"- Mean delta xT: {df['delta_xt'].mean():.4f}")
            lines.append("")
        else:
            lines.append("No cases found matching criteria.\n")

    report = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    return report

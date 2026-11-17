"""Visualization utilities for pass analysis."""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import seaborn as sns


def draw_pitch(ax=None, pitch_length=120, pitch_width=80, color="white", linecolor="black"):
    """Draw a football pitch on the given axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_xlim(-2, pitch_length + 2)
    ax.set_ylim(-2, pitch_width + 2)
    ax.set_aspect("equal")
    ax.set_facecolor(color)

    # Pitch outline
    ax.plot([0, pitch_length], [0, 0], color=linecolor)
    ax.plot([0, pitch_length], [pitch_width, pitch_width], color=linecolor)
    ax.plot([0, 0], [0, pitch_width], color=linecolor)
    ax.plot([pitch_length, pitch_length], [0, pitch_width], color=linecolor)

    # Halfway line
    ax.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width], color=linecolor)

    # Center circle
    circle = plt.Circle((pitch_length / 2, pitch_width / 2), 9.15, fill=False, color=linecolor)
    ax.add_patch(circle)

    # Penalty areas
    for x_start in [0, pitch_length - 18]:
        ax.plot([x_start, x_start + 18], [18, 18], color=linecolor)
        ax.plot([x_start, x_start + 18], [62, 62], color=linecolor)
        if x_start == 0:
            ax.plot([18, 18], [18, 62], color=linecolor)
        else:
            ax.plot([x_start, x_start], [18, 62], color=linecolor)

    # Goal areas
    for x_start in [0, pitch_length - 6]:
        ax.plot([x_start, x_start + 6], [30, 30], color=linecolor)
        ax.plot([x_start, x_start + 6], [50, 50], color=linecolor)
        if x_start == 0:
            ax.plot([6, 6], [30, 50], color=linecolor)
        else:
            ax.plot([x_start, x_start], [30, 50], color=linecolor)

    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_pass_map(df, ax=None, title="Pass Map", color_by="utility_score", cmap="RdYlGn"):
    """Plot passes on a pitch, colored by a value column."""
    ax = draw_pitch(ax=ax)

    if color_by in df.columns:
        norm = Normalize(vmin=df[color_by].quantile(0.05), vmax=df[color_by].quantile(0.95))
        colors = matplotlib.colormaps[cmap](norm(df[color_by].values))
    else:
        colors = "steelblue"

    for i, (_, row) in enumerate(df.iterrows()):
        c = colors[i] if isinstance(colors, np.ndarray) else colors
        ax.annotate(
            "",
            xy=(row["end_x"], row["end_y"]),
            xytext=(row["start_x"], row["start_y"]),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.2, alpha=0.7),
        )

    ax.set_title(title, fontsize=14)
    return ax


def plot_xt_grid(xt_grid, ax=None, title="Expected Threat Grid"):
    """Visualize the xT grid as a heatmap on the pitch."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(
        xt_grid, extent=[0, 120, 0, 80], origin="lower",
        cmap="YlOrRd", aspect="auto", alpha=0.8,
    )
    plt.colorbar(im, ax=ax, label="xT value")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_calibration_curve(y_true, y_prob, n_bins=10, ax=None, label="Model"):
    """Plot calibration curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_trues = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask |= (y_prob == bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means.append(y_prob[mask].mean())
            bin_trues.append(y_true[mask].mean())

    ax.plot(bin_means, bin_trues, "o-", label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def plot_utility_decile_comparison(
    decile_event: pd.DataFrame,
    decile_360: pd.DataFrame,
    metric: str = "avg_delta_xt",
    ax=None,
):
    """Compare utility decile metrics between event-only and event+360 models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(decile_event))
    width = 0.35

    ax.bar(x - width / 2, decile_event[metric], width, label="Event-only", alpha=0.8)
    ax.bar(x + width / 2, decile_360[metric], width, label="Event+360", alpha=0.8)

    ax.set_xlabel("Utility Decile")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by Utility Decile")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i}" for i in range(len(x))])
    ax.legend()
    return ax


def plot_feature_importance(importances: dict, top_n: int = 20, ax=None, title="Feature Importance"):
    """Plot feature importances as horizontal bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))

    sorted_feats = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names = [f[0] for f in sorted_feats][::-1]
    values = [f[1] for f in sorted_feats][::-1]

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]
    ax.barh(names, values, color=colors, alpha=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Importance")
    return ax


def plot_freeze_frame(
    start, end, freeze_frame, ax=None, title="Freeze Frame",
):
    """Visualize a single pass with its 360 freeze-frame context."""
    ax = draw_pitch(ax=ax)

    for p in freeze_frame:
        loc = p.get("location")
        if loc is None:
            continue
        color = "#3498db" if p.get("teammate", False) else "#e74c3c"
        marker = "o" if p.get("teammate", False) else "^"
        ax.plot(loc[0], loc[1], marker, color=color, markersize=10, alpha=0.8)

    # Draw pass arrow
    ax.annotate(
        "",
        xy=(end[0], end[1]),
        xytext=(start[0], start[1]),
        arrowprops=dict(arrowstyle="->", color="black", lw=2.5),
    )
    ax.plot(start[0], start[1], "s", color="gold", markersize=12, zorder=5, label="Passer")

    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper left")
    return ax

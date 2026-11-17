"""Evaluation metrics for pass models."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
)


# === Classification Metrics ===

def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute all classification metrics for pass success prediction."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (n_bin / total) * abs(avg_accuracy - avg_confidence)

    return ece


# === Regression Metrics ===

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics for reward prediction."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "spearman_corr": stats.spearmanr(y_true, y_pred).correlation,
    }


def top_decile_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_pct: float = 0.9,
) -> float:
    """Fraction of predicted top-decile passes that are actually high-value."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pred_threshold = np.percentile(y_pred, threshold_pct * 100)
    true_threshold = np.percentile(y_true, threshold_pct * 100)

    pred_top = y_pred >= pred_threshold
    true_top = y_true >= true_threshold

    if pred_top.sum() == 0:
        return 0.0
    return (pred_top & true_top).sum() / pred_top.sum()


# === Bootstrap Confidence Intervals ===

def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
    match_ids: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    If match_ids is provided, performs match-level block bootstrap
    to account for within-match correlation.

    Returns (point_estimate, lower_bound, upper_bound).
    """
    import warnings
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    match_ids = None if match_ids is None else np.asarray(match_ids)
    rng = np.random.default_rng(random_state)
    point = metric_fn(y_true, y_pred)

    scores = []
    n_failed = 0

    if match_ids is not None:
        unique_matches = np.unique(match_ids)
        n_matches = len(unique_matches)
        for _ in range(n_bootstrap):
            sampled = rng.choice(unique_matches, size=n_matches, replace=True)
            idx = np.concatenate([np.where(match_ids == m)[0] for m in sampled])
            try:
                s = metric_fn(y_true[idx], y_pred[idx])
                scores.append(s)
            except (ValueError, ZeroDivisionError):
                n_failed += 1
    else:
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            try:
                s = metric_fn(y_true[idx], y_pred[idx])
                scores.append(s)
            except (ValueError, ZeroDivisionError):
                n_failed += 1

    if n_failed > n_bootstrap * 0.1:
        warnings.warn(f"Bootstrap: {n_failed}/{n_bootstrap} samples failed")

    if not scores:
        return point, point, point

    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    return point, lower, upper


# === Utility Ranking Evaluation ===

def evaluate_utility_deciles(
    df: pd.DataFrame,
    utility_col: str = "utility_score",
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Evaluate realized quality metrics per utility decile.

    Decile 9 is the highest-utility group and decile 0 the lowest.
    """
    df = df.copy()
    df["decile"] = pd.qcut(df[utility_col], q=n_deciles, labels=False, duplicates="drop")

    agg = df.groupby("decile").agg(
        n_passes=("decile", "size"),
        avg_realized_delta_xt=("realized_delta_xt", "mean"),
        pass_success_rate=("pass_success", "mean"),
        realized_progressive_pct=("realized_progressive", "mean"),
        realized_final_third_entry_pct=("realized_final_third_entry", "mean"),
    ).reset_index()

    return agg

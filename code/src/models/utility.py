"""Utility score computation: U(p|s) = P_succ * R_succ - (1 - P_succ) * C_fail."""

import numpy as np
import pandas as pd


def compute_failure_cost(
    df: pd.DataFrame,
    method: str = "location_based",
    pitch_length: float = 120.0,
    learned_cost: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate pass failure cost.

    Methods:
    - "location_based": Cost proportional to how advanced the turnover position is.
      Losing the ball in your own half is less costly than in the final third.
    - "negative_xt": Use negative xT at start location (losing possession value).
    - "constant": Fixed cost for all failures.
    - "learned": Use predictions from a trained failure_cost_model (pass via
      `learned_cost`, aligned to `df`'s row order).
    """
    if method == "learned":
        if learned_cost is None:
            raise ValueError("method='learned' requires `learned_cost` (see src.models.failure_cost_model)")
        return np.asarray(learned_cost)

    elif method == "location_based":
        # Normalize start_x to [0, 1], then scale
        # Turning over near own goal = low cost (opponent far from goal)
        # Turning over in advanced position = higher cost (counter-attack risk)
        advancement = df["start_x"].values / pitch_length
        # Cost increases with advancement: losing ball at x=100 is worse than x=20
        cost = 0.5 * advancement + 0.1
        return cost

    elif method == "negative_xt":
        if "start_xt" in df.columns:
            return df["start_xt"].values.clip(min=0.001)
        # Fallback: use location-based
        return compute_failure_cost(df, method="location_based", pitch_length=pitch_length)

    elif method == "constant":
        return np.full(len(df), 0.3)

    else:
        raise ValueError(f"Unknown cost method: {method}")


def compute_utility(
    p_success: np.ndarray,
    reward: np.ndarray,
    cost: np.ndarray,
) -> np.ndarray:
    """Compute utility score for each pass.

    U(p|s) = P_succ * R_succ - (1 - P_succ) * C_fail
    """
    return p_success * reward - (1 - p_success) * cost


def rank_passes_by_utility(
    df: pd.DataFrame,
    utility_scores: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Rank passes into deciles by utility score."""
    df = df.copy()
    df["utility_score"] = utility_scores
    df["utility_decile"] = pd.qcut(
        df["utility_score"], q=n_deciles, labels=False, duplicates="drop",
    )
    return df

"""Learned failure-cost (C_fail) model.

Predicts realized opponent xT gain shortly after a turnover (see
src.data.turnover_labels.compute_realized_turnover_cost), as a data-driven
alternative to the location_based heuristic in src.models.utility.compute_failure_cost.
"""

import numpy as np
import pandas as pd

from src.models.reward_model import train_reward_model, predict_reward


def train_failure_cost_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
):
    """Train a C_fail regression model on realized turnover-cost labels."""
    return train_reward_model(model_name, X_train, y_train, X_valid, y_valid)


def predict_failure_cost(model, X: pd.DataFrame) -> np.ndarray:
    """Predict C_fail for any pass (successful or not). Cost cannot be negative."""
    return np.clip(predict_reward(model, X), 0.0, None)

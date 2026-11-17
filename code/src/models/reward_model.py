"""Pass reward (delta xT) prediction models."""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


REWARD_MODEL_REGISTRY = {
    "linear_regression": lambda: ElasticNet(
        alpha=0.001, l1_ratio=0.5, max_iter=2000, random_state=42,
    ),
    "xgboost_regressor": lambda: XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        random_state=42, verbosity=0,
    ),
    "lightgbm_regressor": lambda: LGBMRegressor(
        n_estimators=500, num_leaves=63, learning_rate=0.05,
        random_state=42, verbose=-1,
    ),
}


def get_reward_model(name: str):
    """Get a reward model instance by name."""
    if name not in REWARD_MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(REWARD_MODEL_REGISTRY.keys())}")
    return REWARD_MODEL_REGISTRY[name]()


def train_reward_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
):
    """Train a reward prediction model."""
    model = get_reward_model(model_name)

    if model_name in ("xgboost_regressor", "lightgbm_regressor") and X_valid is not None:
        if model_name == "xgboost_regressor":
            model.set_params(early_stopping_rounds=50)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
        else:
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
            )
    else:
        model.fit(X_train, y_train)

    return model


def predict_reward(model, X: pd.DataFrame) -> np.ndarray:
    """Predict pass reward (delta xT)."""
    return model.predict(X)

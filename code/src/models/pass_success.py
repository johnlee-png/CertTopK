"""Pass success probability prediction models."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


MODEL_REGISTRY = {
    "logistic_regression": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)),
    ]),
    "xgboost": lambda: XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        eval_metric="logloss",
        random_state=42, verbosity=0,
    ),
    "lightgbm": lambda: LGBMClassifier(
        n_estimators=500, num_leaves=63, learning_rate=0.05,
        random_state=42, verbose=-1,
    ),
    "mlp": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4,
            learning_rate_init=1e-3, max_iter=300, early_stopping=True,
            n_iter_no_change=15, random_state=42,
        )),
    ]),
}


def get_model(name: str):
    """Get a model instance by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()


def train_success_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
):
    """Train a pass success prediction model."""
    model = get_model(model_name)

    if model_name in ("xgboost", "lightgbm") and X_valid is not None:
        if model_name == "xgboost":
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


def predict_success_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Predict pass success probability."""
    return model.predict_proba(X)[:, 1]

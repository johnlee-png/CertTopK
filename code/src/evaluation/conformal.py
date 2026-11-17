"""Split conformal prediction for the binary pass-success classifier.

Score-based (LAC) split conformal: calibrate a nonconformity threshold on a
held-out calibration (validation) set, then build prediction sets on test data
with guaranteed marginal coverage of at least 1 - alpha.
"""

import numpy as np


def calibrate_conformal_threshold(
    y_valid: np.ndarray,
    p_valid: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Compute the conformal threshold q_hat from calibration (validation) data.

    Nonconformity score for true label y and predicted P(success)=p is
    1 - p_hat(true class): (1 - p) if y == 1 else p.
    """
    y_valid = np.asarray(y_valid)
    p_valid = np.asarray(p_valid)
    p_true_class = np.where(y_valid == 1, p_valid, 1 - p_valid)
    scores = 1 - p_true_class

    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def predict_conformal_sets(p_test: np.ndarray, q_hat: float) -> np.ndarray:
    """Build prediction sets. Returns (n, 2) bool array: columns [includes_0, includes_1]."""
    p_test = np.asarray(p_test)
    includes_0 = p_test <= q_hat
    includes_1 = (1 - p_test) <= q_hat
    return np.stack([includes_0, includes_1], axis=1)


def set_size(pred_sets: np.ndarray) -> np.ndarray:
    return pred_sets.sum(axis=1)


def empirical_coverage(y_true: np.ndarray, pred_sets: np.ndarray) -> float:
    """Fraction of examples whose true label is contained in the prediction set.

    Signature matches src.evaluation.metrics.bootstrap_metric's metric_fn contract.
    """
    y_true = np.asarray(y_true).astype(int)
    covered = pred_sets[np.arange(len(y_true)), y_true]
    return float(covered.mean())


def average_set_size(y_true: np.ndarray, pred_sets: np.ndarray) -> float:
    """Mean prediction-set size (efficiency). y_true is unused but kept for the
    bootstrap_metric(y_true, y_pred, metric_fn) contract."""
    return float(set_size(pred_sets).mean())


def singleton_accuracy(y_true: np.ndarray, pred_sets: np.ndarray) -> float:
    """Among examples with a singleton prediction set, fraction where that
    single predicted label matches the true label."""
    y_true = np.asarray(y_true).astype(int)
    sizes = set_size(pred_sets)
    singleton = sizes == 1
    if not singleton.any():
        return float("nan")
    predicted_label = pred_sets[:, 1].astype(int)  # includes_1 -> label 1 if singleton is {1}
    correct = predicted_label[singleton] == y_true[singleton]
    return float(correct.mean())

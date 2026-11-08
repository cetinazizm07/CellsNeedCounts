"""Regression metrics shared by every model, so all comparisons use the same code."""
import numpy as np


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(err)),  # mean signed error (+ = over-counting)
    }


def grouped_metrics(y_true, y_pred, groups) -> dict:
    """Metrics per group value (e.g. per blur level), as {group: metrics}."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    return {g.item(): regression_metrics(y_true[groups == g], y_pred[groups == g])
            for g in np.unique(groups)}

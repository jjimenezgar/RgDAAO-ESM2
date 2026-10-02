from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr, spearmanr


def regression_metrics(y_true, y_pred) -> dict[str, float | None]:
    """Undefined correlations are null, never fabricated zeros or JSON NaNs."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError('y_true and y_pred must be one-dimensional arrays of equal shape')
    if len(y_true) < 2 or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError('At least two finite observations are required')
    constant = np.ptp(y_true) == 0 or np.ptp(y_pred) == 0
    return {
        'spearman': None if constant else float(spearmanr(y_true, y_pred).statistic),
        'pearson': None if constant else float(pearsonr(y_true, y_pred).statistic),
        'mae': float(np.mean(np.abs(y_true - y_pred))),
        'rmse': float(np.sqrt(np.mean((y_true - y_pred)**2))),
    }

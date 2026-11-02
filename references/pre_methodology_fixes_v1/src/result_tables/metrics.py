"""Problem-specific metrics from rp4lp.py (lines 1061–1065, 1261–1267)."""

import numpy as np
from .config import OBJECTIVE_ZERO_TOL

TABLE_COLUMNS = ["epsilon", "f_projected / f_original", "f_retrieved / f_original",
                 "avgin", "avgeq", "k / m", "t_projected / t_original"]


def ratio(numerator, denominator, zero_tolerance=OBJECTIVE_ZERO_TOL):
    if numerator is None or denominator is None:
        return float("nan")
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= zero_tolerance:
        return float("nan")
    return float(numerator / denominator)


def quantile_metrics(instance, x, lp):
    m, p = instance.m, instance.n
    positive_violation = np.maximum(-x[p:p + m], 0)
    negative_violation = np.maximum(-x[p + m:], 0)
    residual = lp.A @ x - lp.b
    return {
        "avgin": float((positive_violation.sum() + negative_violation.sum()) / m),
        "avgeq": float(np.abs(residual).mean()),
        # Upstream maxinq adds the maxima of the two blocks. Also retain the
        # actual maximum violation, whose meaning differs from that convention.
        "maxin": float(positive_violation.max() + negative_violation.max()),
        "max_bound_violation": float(max(positive_violation.max(), negative_violation.max())),
        "maxeq": float(np.abs(residual).max()),
        "original_residual_l2": float(np.linalg.norm(residual)),
    }


def diet_metrics(instance, x, lp=None):
    q = x[:instance.n]
    bounds = np.maximum(-q, 0)
    nutrients = np.maximum(instance.b - instance.D @ q, 0)
    return {
        "avgin": float(bounds.mean()), "maxin": float(bounds.max()),
        "max_bound_violation": float(bounds.max()),
        # The publication's avgeq label measures nutrient INEQUALITY shortfalls.
        "avgeq": float(nutrients.mean()), "maxeq": float(nutrients.max()),
        "original_residual_l2": float(np.linalg.norm(nutrients)),
    }

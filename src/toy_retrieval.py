"""Full-bound metrics and alternating projection for the illustrative toy LPs.

The toys compare exact Eq. (5) projection with an LSMR correction on A. They are
not the structured QR/Diet benchmark or its Eq. (19) Dykstra retrieval.
"""

import time
import numpy as np
import pandas as pd
from scipy.sparse.linalg import lsmr


def achlioptas_projection_matrix(k, m, rng):
    """Three-point entries with variance 1/k (nonzero probability 1/3)."""
    return rng.choice([-1., 0., 1.], size=(k, m), p=[1/6, 2/3, 1/6]) * np.sqrt(3. / k)


def compute_metrics(A, b, c, x, upper_bound=5., delta=1e-6):
    lower = np.maximum(-x, 0.)
    upper = np.maximum(x - upper_bound, 0.)
    residual = float(np.linalg.norm(A @ x - b))
    maximum = float(np.maximum(lower, upper).max())
    return dict(objective=float(c @ x), eq_residual=residual,
                negativity_sum=float(lower.sum()), min_entry=float(x.min()),
                upper_violation_sum=float(upper.sum()),
                bound_violation_sum=float((lower + upper).sum()),
                max_bound_violation=maximum,
                feasible=bool(np.isfinite(x).all() and residual <= delta and maximum <= 1e-10))


def exact_equality_projection_update(A, b, x):
    return x + A.T @ np.linalg.solve(A @ A.T, b - A @ x)


def approximate_equality_projection_update(A, b, x):
    """Approximate Eq. (5) directly on A, avoiding squared Gram conditioning."""
    return x + lsmr(A, b - A @ x, atol=1e-8, btol=1e-8, maxiter=500)[0]


def apm_retrieval(A, b, x_start, mode='exact', max_iter=30, delta=1e-6, upper_bound=5.):
    """Alternate affine projection and clipping to the complete LP interval."""
    updates = {'exact': exact_equality_projection_update,
               'approx_lsmr': approximate_equality_projection_update}
    if mode not in updates:
        raise ValueError("mode must be 'exact' or 'approx_lsmr'")
    x = x_start.copy()
    history = []
    started = time.perf_counter()
    for iteration in range(1, max_iter + 1):
        x = updates[mode](A, b, x)
        metrics = compute_metrics(A, b, np.zeros_like(x), x, upper_bound, delta)
        stage = 'after_eq_projection_stop'
        if not metrics['feasible']:
            x = np.clip(x, 0., upper_bound)
            metrics = compute_metrics(A, b, np.zeros_like(x), x, upper_bound, delta)
            stage = 'after_clip'
        history.append(dict(iteration=iteration, stage=stage, **metrics))
        if metrics['feasible']:
            break
    return x, time.perf_counter() - started, pd.DataFrame(history)

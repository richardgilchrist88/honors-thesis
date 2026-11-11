"""Corrected notebook comparison methods from Liberti et al., Sections 3.4–3.5.

These replace the historical notebook branches. Dykstra shares the table
generator's implementation. VN and AARM use the paper's operator order with the
same approximate affine step; finite/stagnated iterates are not certified feasible.
"""

from dataclasses import replace
import numpy as np
from scipy.sparse.linalg import lsmr
from .result_tables import config
from .result_tables.common import LinearProgram
from .result_tables.retrieval import (
    affine_line_projection, bisect_toward_projected, retrieve, errors,
)
from .result_tables.retrieval_trace import record_iteration


def bound_error_sum(x, lb, ub):
    return float((np.maximum(lb - x, 0) + np.maximum(x - ub, 0)).sum())


def retrieval_feasibility(x, A, b, lb, ub, tol=.01, *, eq_tol=None, bound_tol=None):
    """Require BOTH equality L2 and summed bound violation to pass their tolerances."""
    eq_tol = tol if eq_tol is None else eq_tol
    bound_tol = tol if bound_tol is None else bound_tol
    return bool(np.isfinite(x).all() and np.linalg.norm(A @ x - b) <= eq_tol
                and bound_error_sum(x, lb, ub) <= bound_tol)


def retrieval_start(x_projected, A, b):
    return x_projected + lsmr(A, b - A @ x_projected)[0]


def solve_projection_system(x0, A, b, x_check):
    return affine_line_projection(x0, A, b, x_check, lambda matrix, rhs: lsmr(matrix, rhs)[0])


def _settings(max_iter, delta):
    config.integer(max_iter, 'max_iter')
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('delta must be finite and positive')
    return replace(config.RETRIEVAL, max_iterations=max_iter, tolerance=delta)


def _alternating(x0, A, b, lb, ub, max_iter, delta, *, reflection=False, alpha=.9, beta=.9,
                 trace=None):
    _settings(max_iter, delta)
    x = retrieval_start(x0, A, b)
    x_check = lsmr(A, b)[0]
    method = 'AARM' if reflection else 'VN'
    for iteration in range(1, max_iter + 1):
        feasible = retrieval_feasibility(x, A, b, lb, ub, tol=delta)
        if trace is not None:
            bound, equality, _ = errors(x, A, b, lb, ub)
            record_iteration(trace, method, iteration - 1, 'before_update', bound,
                             equality, delta, max_iter, feasibility_checked=True,
                             branch='feasible' if feasible else 'continue')
        if feasible:
            return x, iteration - 1, 'feasible'
        affine = solve_projection_system(x, A, b, x_check)
        if reflection:
            reflected_affine = 2 * beta * affine - x
            reflected_bounds = 2 * beta * np.clip(reflected_affine, lb, ub) - reflected_affine
            new = (1 - alpha) * x + alpha * reflected_bounds
        else:
            new = np.clip(affine, lb, ub)  # P_S o P_U, Section 3.4
        if not np.isfinite(new).all():
            raise FloatingPointError('Retrieval produced a nonfinite iterate')
        feasible = retrieval_feasibility(new, A, b, lb, ub, tol=delta)
        step = float(np.linalg.norm(new - x)) if trace is not None or not feasible else None
        branch = ('feasible' if feasible else 'stagnated' if step <= delta else
                  'iteration_limit' if iteration == max_iter else 'continue')
        if trace is not None:
            bound, equality, _ = errors(new, A, b, lb, ub)
            record_iteration(trace, method, iteration, 'after_update', bound,
                             equality, delta, max_iter, step=step, branch=branch,
                             feasibility_checked=True, step_checked=not feasible)
        if feasible:
            return new, iteration, 'feasible'
        if step <= delta:
            return new, iteration, 'stagnated; check residuals'
        x = new
    return x, max_iter, 'iteration limit; check residuals'


def retrieve_apm(x0, A, b, lb, ub, max_iter=30, delta=.01, *, trace=None):
    """Von Neumann affine-then-bounds operator, preceded by the retrieval start."""
    return _alternating(x0, A, b, lb, ub, max_iter, delta, trace=trace)


def retrieve_dykstra(x0, A, b, lb, ub, max_iter=30, delta=.01, *, trace=None):
    """Bounds-first Dykstra, identical to the table generator before bisection."""
    settings = _settings(max_iter, delta)
    # A zero objective disables bisection here; callers refine with their true c.
    lp = LinearProgram(A, b, np.zeros_like(x0), lb, ub)
    answer = retrieve(x0, lp, settings, collect_trace=trace is not None)
    if trace is not None:
        trace.extend(answer.diagnostics['retrieval_trace'])
    return answer.x, answer.diagnostics['retrieval_iterations'], answer.diagnostics['retrieval_status']


def retrieve_aarm(x0, A, b, lb, ub, max_iter=30, delta=.01, alpha=.9, beta=.9, *, trace=None):
    """Apply (1-alpha) I + alpha (2 beta P_S-I) o (2 beta P_U-I)."""
    if not (0 <= alpha <= 1 and 0 <= beta <= 1):
        raise ValueError('alpha and beta must lie in [0, 1]')
    return _alternating(x0, A, b, lb, ub, max_iter, delta,
                        reflection=True, alpha=alpha, beta=beta, trace=trace)


def refine_retrieval(x, projected, A, b, c, lb, ub, delta=.01):
    """Section 3.5 minimization bisection, with explicit skip/status reporting."""
    return bisect_toward_projected(x, projected, LinearProgram(A, b, c, lb, ub),
                                  _settings(30, delta))

"""Dykstra + LSMR line projection (Liberti et al., Sections 3.4–3.5).

Adapted from Leo Liberti's rp4lp.py, CC0; see references/rp4lp/provenance.json.
Bounds-first order follows the authors. Corrected diagnostics and minimization
bisection follow the written method; see docs/result-table-methodology.md.
"""

from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import lsmr
from . import config
from .retrieval_trace import record_iteration, stopping_explanation


def errors(x, A, b, lower, upper):
    bounds = np.maximum(lower - x, 0) + np.maximum(x - upper, 0)
    return float(bounds.sum()), float(np.linalg.norm(A @ x - b)), float(bounds.max())


def affine_line_projection(x, A, b, x_check, least_squares):
    """Equation (19), including a finite fallback for a degenerate line."""
    x_prime = least_squares(A, b + A @ x) - x
    direction = x_check - x_prime
    norm_squared = float(direction @ direction)
    if norm_squared <= np.finfo(float).eps ** 2:
        return x_prime
    return x_prime + float(direction @ (x - x_prime)) / norm_squared * direction


def bisect_toward_projected(x, projected, lp, settings):
    """Minimize along the segment while retaining approximate affine/bound feasibility."""
    tol = settings.tolerance
    bound, equality, _ = errors(x, lp.A, lp.b, lp.lower, lp.upper)
    if equality > tol or bound > tol:
        return x, 0, "skipped_infeasible_start"
    if float(lp.c @ x - lp.c @ projected) <= tol:
        return x, 0, "skipped_no_objective_improvement"
    low, high = 0.0, 1.0
    best = x.copy()
    for iteration in range(1, settings.bisection_max_iterations + 1):
        fraction = (low + high) / 2
        candidate = (1 - fraction) * x + fraction * projected
        bound, equality, _ = errors(candidate, lp.A, lp.b, lp.lower, lp.upper)
        if equality <= tol and bound <= tol:
            low, best = fraction, candidate
        else:
            high = fraction
        if (high - low) * abs(float(lp.c @ (projected - x))) <= tol:
            return best, iteration, "objective_interval_tolerance"
    return best, settings.bisection_max_iterations, "iteration_limit"


@dataclass
class RetrievalResult:
    x: np.ndarray
    diagnostics: dict


def retrieve(projected, lp, settings=None, *, collect_trace=True):
    settings = config.RETRIEVAL if settings is None else settings
    trace = []
    history = [] if collect_trace else None

    def least_squares(A, rhs):
        answer = lsmr(A, rhs, atol=settings.lsmr_atol, btol=settings.lsmr_btol,
                      conlim=settings.lsmr_conlim, maxiter=settings.lsmr_maxiter)
        trace.append({"status": int(answer[1]), "iterations": int(answer[2]),
                      "residual_l2": float(answer[3]), "condition_estimate": float(answer[6])})
        if not np.isfinite(answer[0]).all():
            raise FloatingPointError("LSMR produced a nonfinite solution")
        return answer[0]

    A, b = lp.A, lp.b
    x = projected + least_squares(A, b - A @ projected)
    x_check = least_squares(A, b)
    p, q = np.zeros_like(x), np.zeros_like(x)
    reason = "iteration_limit"
    iterations = 0
    tol = settings.tolerance
    for iterations in range(1, settings.max_iterations + 1):
        bound, equality, _ = errors(x, A, b, lp.lower, lp.upper)
        record_iteration(history, "Dykstra", iterations - 1, "before_update",
                         bound, equality, tol, settings.max_iterations,
                         branch="feasible" if bound <= tol and equality <= tol else "continue",
                         feasibility_checked=True)
        if bound <= tol and equality <= tol:
            reason = "feasible"
            iterations -= 1
            break
        previous = x.copy()
        y = np.clip(x + p, lp.lower, lp.upper)
        p = x + p - y
        # Retain the authors' line-based approximate affine projection.
        x = affine_line_projection(y + q, A, b, x_check, least_squares)
        q = y + q - x
        if not np.isfinite(x).all():
            raise FloatingPointError("Dykstra produced a nonfinite iterate")
        bound, equality, _ = errors(x, A, b, lp.lower, lp.upper)
        # Extra measurement only: keep the original branch order and updates.
        step = float(np.linalg.norm(x - previous)) if collect_trace else None
        if bound + equality <= 2 * tol:
            reason = "small_combined_error"
            record_iteration(history, "Dykstra", iterations, "after_update", bound,
                             equality, tol, settings.max_iterations, step=step,
                             branch=reason, combined_checked=True)
            break
        if step is None:
            step = float(np.linalg.norm(x - previous))
        stagnated = step <= tol
        branch = ("stagnated" if stagnated else "iteration_limit"
                  if iterations == settings.max_iterations else "continue")
        record_iteration(history, "Dykstra", iterations, "after_update", bound,
                         equality, tol, settings.max_iterations, step=step,
                         branch=branch, combined_checked=True, step_checked=True)
        if stagnated:
            reason = "stagnated"
            break

    pre_bound, pre_equality, _ = errors(x, A, b, lp.lower, lp.upper)
    x, bisection_iterations, bisection_status = bisect_toward_projected(x, projected, lp, settings)
    bound, equality, max_bound = errors(x, A, b, lp.lower, lp.upper)
    return RetrievalResult(x, {
        "retrieval_status": reason,
        "retrieval_iterations": iterations,
        "retrieval_bound_error_sum": bound,
        "retrieval_max_bound_violation": max_bound,
        "retrieval_equality_l2": equality,
        "retrieval_feasible": bound <= tol and equality <= tol,
        "retrieval_combined_error_satisfied": bound + equality <= 2 * tol,
        "pre_bisection_bound_error_sum": pre_bound,
        "pre_bisection_equality_l2": pre_equality,
        "bisection_iterations": bisection_iterations,
        "bisection_status": bisection_status,
        "lsmr_calls": len(trace),
        "lsmr_iterations": sum(r["iterations"] for r in trace),
        "lsmr_trace": trace,
        "retrieval_trace": history if history is not None else [],
        "retrieval_explanation": stopping_explanation(
            history, reason, final_feasible=bound <= tol and equality <= tol,
            bisection_status=bisection_status, bisection_iterations=bisection_iterations),
    })

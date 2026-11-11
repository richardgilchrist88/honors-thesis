"""Iteration diagnostics shared by the table pipeline and comparison notebook.

Iteration counts completed updates. A before_update row tests the current point
without performing another update. Logging never selects the stopping branch.
"""

TRACE_COLUMNS = [
    "method", "iteration", "stage", "step_norm", "equality_l2",
    "bound_error_sum", "combined_error", "equality_tolerance", "bound_tolerance",
    "combined_tolerance", "step_tolerance", "max_iterations",
    "feasibility_checked", "combined_error_checked", "step_checked",
    "feasible", "stopping_branch",
]


def record_iteration(history, method, iteration, stage, bound, equality, tol,
                     max_iter, *, step=None, branch="continue",
                     feasibility_checked=False, combined_checked=False,
                     step_checked=False):
    if history is None:
        return
    history.append(dict(
        method=method, iteration=iteration, stage=stage, step_norm=step,
        equality_l2=equality, bound_error_sum=bound, combined_error=bound + equality,
        equality_tolerance=tol, bound_tolerance=tol, combined_tolerance=2 * tol,
        step_tolerance=tol, max_iterations=max_iter,
        feasibility_checked=feasibility_checked,
        combined_error_checked=combined_checked, step_checked=step_checked,
        feasible=bool(bound <= tol and equality <= tol), stopping_branch=branch,
    ))


def stopping_explanation(history, status, *, final_feasible=None,
                         bisection_status=None, bisection_iterations=None):
    """Explain the actual branch, including measurements before bisection."""
    if not history:
        return f"Retrieval: {status}; no iteration trace available."
    last = history[-1]
    branch = last["stopping_branch"]
    lines = [f"{last['method']} stopped after {last['iteration']} updates: {status} "
             f"(branch={branch}, stage={last['stage']})."]
    if branch == "feasible":
        lines.append(f"Both residual checks passed: equality {last['equality_l2']:.17g} "
                     f"<= {last['equality_tolerance']:.17g}; bounds "
                     f"{last['bound_error_sum']:.17g} <= {last['bound_tolerance']:.17g}.")
    elif branch == "small_combined_error":
        lines.append(f"Combined error {last['combined_error']:.17g} "
                     f"<= {last['combined_tolerance']:.17g}; "
                     "the step check was not evaluated.")
    elif branch == "stagnated":
        lines.append(f"Step norm {last['step_norm']:.17g} "
                     f"<= {last['step_tolerance']:.17g}.")
    elif branch == "iteration_limit":
        lines.append(f"Iteration cap {last['max_iterations']} reached; earlier exit checks did not pass.")
    step = "not applicable" if last['step_norm'] is None else f"{last['step_norm']:.17g}"
    lines.append(f"At retrieval exit: equality L2={last['equality_l2']:.17g}, "
                 f"summed bound violations={last['bound_error_sum']:.17g}, step norm={step}; "
                 f"feasible={last['feasible']} (each residual must be <= "
                 f"{last['equality_tolerance']:.17g}).")
    if bisection_status is not None:
        lines.append(f"Bisection: {bisection_status}; iterations={bisection_iterations}.")
    if final_feasible is not None:
        lines.append(f"Feasible after postprocessing in the retrieval space: {bool(final_feasible)}.")
    return "\n".join(lines)

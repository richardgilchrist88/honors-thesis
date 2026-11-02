"""QR data/LP helpers and corrected notebook retrieval interfaces (revision 1.1).

Retrieval methods share src.retrieval_methods; the historical implementation is
archived under references/pre_methodology_fixes_v1. The data generator retains
the reference notebook's fixed-count predictors and dense response.
"""

import numpy as np
from scipy import sparse

import gurobipy as gp
from gurobipy import GRB
from scipy.sparse.linalg import lsmr, lsqr
import time

def generate_quantile_instance(
    m: int,
    p: int,
    density: float = 0.8,
    seed: int | None = None,
):
    """
    Generate one synthetic Quantile Regression instance.

    Parameters
    ----------
    m : int
        Number of observations / records.

    p : int
        Number of explanatory variables (columns of D).

    density : float, default=0.8
        Target proportion of nonzero entries in D.

    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    D : ndarray, shape (m, p)
        Design matrix.

    b : ndarray, shape (m,)
        Dependent variable.
    """

    # create a local random number generator
    rng = np.random.default_rng(seed)

    # check input values
    if m <= 0:
        raise ValueError("m must be positive")

    if p <= 0:
        raise ValueError("p must be positive")

    if not 0.0 <= density <= 1.0:
        raise ValueError("density must lie between 0 and 1")

    # initialize D as an all-zero matrix
    D = np.zeros((m, p), dtype=float)

    # calculate the total number of entries
    n_total = m * p

    # calculate the number of nonzero entries
    n_nonzero = int(round(density * n_total))

    # randomly choose the positions of nonzero entries
    nonzero_indices = rng.choice(n_total, size=n_nonzero, replace=False)

    # fill the selected positions with values sampled from U[-1, 1]
    D.flat[nonzero_indices] = rng.uniform(-1.0, 1.0, size=n_nonzero)

    # generate the dependent variable from U[-1, 1]
    b = rng.uniform(-1.0, 1.0, size=m)

    return D, b

def build_quantile_lp(
    D: np.ndarray,
    b: np.ndarray,
    tau: float = 0.2,
):
    """
    Build the LP formulation of a quantile regression instance.

    Parameters
    ----------
    D : ndarray, shape (m, p)
        Design matrix

    b : ndarray, shape (m,)
        Dependent variable

    tau : float, default=0.2
        Quantile level

    Returns
    -------
    A : sparse matrix, shape (m, p + 2m)
        Equality constraint matrix

    c : ndarray, shape (p + 2m,)
        Objective coefficient vector

    """

    # get the dimensions of D
    m, p = D.shape

    # check the dimension of b
    if b.shape != (m,):
        raise ValueError("b must have shape (m,)")

    # check the quantile level
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie strictly between 0 and 1")

    # convert D to sparse format
    D_sparse = sparse.csr_matrix(D)

    # construct a sparse identity matrix
    I = sparse.eye(m, format="csr")

    # construct A = [D | I | -I]
    A = sparse.hstack((D_sparse, I, -I), format="csr")

    # construct the objective coefficient vector
    c = np.concatenate(
        (np.zeros(p), tau * np.ones(m), (1.0 - tau) * np.ones(m))
    )

    return A, c

def generate_projection_matrix(
    A,
    k: int,
    seed: int | None = None,
):
    """
    Generate a sparse random projection matrix T.

    Parameters
    ----------
    A : sparse matrix, shape (m, n)
        Constraint matrix

    k : int
        Projected dimension

    seed : int or None, default=None
        Random seed for reproducibility

    Returns
    -------
    T : sparse matrix, shape (k, m)
        Sparse random projection matrix
    """

    # create a local random number generator
    rng = np.random.default_rng(seed)

    # get the number of original constraints
    m = A.shape[0]

    # check the projected dimension
    if k <= 0:
        raise ValueError("k must be positive")

    # calculate the density of A
    density_A = A.nnz / (A.shape[0] * A.shape[1])

    # set the density of T
    sigma = density_A / 2.0

    # calculate the total number of entries in T
    n_total = k * m

    # calculate the number of nonzero entries in T
    n_nonzero = int(round(sigma * n_total))

    # randomly choose the positions of nonzero entries
    nonzero_indices = rng.choice(n_total, size=n_nonzero, replace=False)

    # convert the flat indices to row and column indices
    rows = nonzero_indices // m
    cols = nonzero_indices % m

    # calculate the standard deviation of nonzero entries
    std = 1.0 / np.sqrt(k * sigma)

    # sample the nonzero entries from the Gaussian distribution
    values = rng.normal(loc=0.0, scale=std, size=n_nonzero)

    # construct the sparse projection matrix
    T = sparse.csr_matrix((values, (rows, cols)), shape=(k, m))

    return T

def project_lp(
    A,
    b: np.ndarray,
    T,
):
    """
    Project the equality constraints of an LP.

    Parameters
    ----------
    A : sparse matrix, shape (m, n)
        Original constraint matrix

    b : ndarray, shape (m,)
        Original right-hand side vector

    T : sparse matrix, shape (k, m)
        Random projection matrix

    Returns
    -------
    A_projected : sparse matrix, shape (k, n)
        Projected constraint matrix

    b_projected : ndarray, shape (k,)
        Projected right-hand side vector
    """

    # check the dimensions of A and b
    if b.shape != (A.shape[0],):
        raise ValueError("b must have shape (m,)")

    # check the dimensions of T and A
    if T.shape[1] != A.shape[0]:
        raise ValueError("T and A have incompatible dimensions")

    # project the constraint matrix
    A_projected = T @ A

    # project the right-hand side vector
    b_projected = T @ b

    return A_projected, b_projected

def solve_lp(
    A,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    threads: int = 1,
    solver_seed: int = 42,
    output_flag: int = 0,
    log_file: str | None = None,
    timings: dict | None = None,
    method: int = -1,
    model_name: str = "lp_model",
):
    """
    Solve a linear program using Gurobi.

    Parameters
    ----------
    A : sparse matrix, shape (m, n)
        Equality constraint matrix

    b : ndarray, shape (m,)
        Right-hand side vector

    c : ndarray, shape (n,)
        Objective coefficient vector

    lb : ndarray, shape (n,)
        Lower bounds of the decision variables

    ub : ndarray, shape (n,)
        Upper bounds of the decision variables

    log_file : str or None
        Optional Gurobi progress log (enables logging).

    timings : dict or None
        Optional destination for setup, optimize, and cleanup wall-clock seconds.

    Returns
    -------
    x_opt : ndarray, shape (n,)
        Optimal solution

    objective_value : float
        Optimal objective value
    """

    # get the number of constraints and decision variables
    m, n = A.shape

    # check the dimension of b
    if b.shape != (m,):
        raise ValueError("b must have shape (m,)")

    # check the dimension of c
    if c.shape != (n,):
        raise ValueError("c must have shape (n,)")

    # check the dimension of the lower bounds
    if lb.shape != (n,):
        raise ValueError("lb must have shape (n,)")

    # check the dimension of the upper bounds
    if ub.shape != (n,):
        raise ValueError("ub must have shape (n,)")

    # create a Gurobi model
    setup_start = time.perf_counter()
    if method not in (-1, 0, 1, 2):
        raise ValueError("method must be -1, 0, 1, or 2")
    model = gp.Model(model_name)
    model_created = time.perf_counter()
    model.Params.Threads = threads
    model.Params.Method = method
    model.Params.Seed = solver_seed
    model.Params.OutputFlag = 1 if log_file else output_flag
    model.Params.LogToConsole = output_flag
    model.Params.LogFile = log_file or ""

    # create the decision variables
    x = model.addMVar(shape=n, lb=lb, ub=ub, name="x")

    # set the objective function
    model.setObjective(c @ x, GRB.MINIMIZE)

    # add the equality constraints
    model.addConstr(A @ x == b)

    # include pending model updates in setup when measuring its components
    if timings is not None:
        model.update()
    setup_end = time.perf_counter()

    # solve the LP
    model.optimize()
    solve_end = time.perf_counter()

    # check the optimization status
    if model.Status != GRB.OPTIMAL:
        status = model.Status
        model.dispose()
        raise RuntimeError(
            f"Gurobi did not find an optimal solution, status={status}"
        )

    # extract the optimal solution
    x_opt = x.X

    # extract the optimal objective value
    objective_value = model.ObjVal

    model.dispose()
    cleanup_end = time.perf_counter()
    if timings is not None:
        timings.update(
            model_creation_seconds=model_created - setup_start,
            setup_seconds=setup_end - setup_start,
            optimize_seconds=solve_end - setup_end,
            extraction_cleanup_seconds=cleanup_end - solve_end,
        )
    return x_opt, objective_value

def build_quantile_bounds(
    m: int,
    p: int,
):
    """
    Build the variable bounds for the quantile regression LP.

    Parameters
    ----------
    m : int
        Number of observations

    p : int
        Number of regression coefficients

    Returns
    -------
    lb : ndarray, shape (p + 2m,)
        Lower bounds

    ub : ndarray, shape (p + 2m,)
        Upper bounds
    """

    # construct the lower bounds
    lb = np.concatenate((np.full(p, -np.inf), np.zeros(m), np.zeros(m)))

    # construct the upper bounds
    ub = np.full(p + 2 * m, GRB.INFINITY)

    return lb, ub

def solve_projection_system(x0, A, b, x_check):
    """Delegate to the corrected, shared notebook method."""
    from src.retrieval_methods import solve_projection_system as implementation
    return implementation(x0, A, b, x_check)

def retrieval_feasibility(x, A, b, lb, ub, tol=0.01, *, eq_tol=None, bound_tol=None):
    """Delegate to the corrected, shared notebook method."""
    from src.retrieval_methods import retrieval_feasibility as implementation
    return implementation(x, A, b, lb, ub, tol, eq_tol=eq_tol, bound_tol=bound_tol)


def retrieve_apm(x0, A, b, lb, ub, max_iter=30, delta=0.01):
    """Delegate to the corrected, shared notebook method."""
    from src.retrieval_methods import retrieve_apm as implementation
    return implementation(x0, A, b, lb, ub, max_iter, delta)

def evaluate_retrieval(
    x: np.ndarray,
    A,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
):
    """
    Evaluate a retrieved solution.
    """

    # calculate the objective value
    objective_value = c @ x

    # calculate the equality constraint residual
    equality_residual = np.linalg.norm(A @ x - b)

    # calculate lower-bound violations
    lower_violation = np.maximum(lb - x, 0.0)

    # calculate upper-bound violations
    upper_violation = np.zeros_like(x)

    finite_upper = np.isfinite(ub)

    upper_violation[finite_upper] = np.maximum(
        x[finite_upper] - ub[finite_upper], 0.0
    )

    # calculate the maximum bound violation
    bound_violation = max(np.max(lower_violation), np.max(upper_violation))

    return (objective_value, equality_residual, bound_violation)

def retrieve_dykstra(x0, A, b, lb, ub, max_iter=30, delta=0.01):
    """Delegate to the corrected, shared notebook method."""
    from src.retrieval_methods import retrieve_dykstra as implementation
    return implementation(x0, A, b, lb, ub, max_iter, delta)

def retrieve_aarm(x0, A, b, lb, ub, max_iter=30, delta=0.01, alpha=0.9, beta=0.9):
    """Delegate to the corrected, shared notebook method."""
    from src.retrieval_methods import retrieve_aarm as implementation
    return implementation(x0, A, b, lb, ub, max_iter, delta, alpha, beta)

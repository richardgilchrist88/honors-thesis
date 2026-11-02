"""Diet TDP1/TDP2 formulations and authors' variant-specific retrieval spaces."""

import numpy as np
from scipy import sparse
from . import config
from .common import Instance, LinearProgram, Timer
from .gurobi_solver import solve_lp
from .projection import sparse_gaussian
from .pipeline import one_replication
from .metrics import diet_metrics

PROBLEM_TYPE = "Diet Problem"
VARIANTS = ("TDP1", "TDP2")


def generate_instance(row_size, instance_seed):
    m = config.integer(row_size, "row_size")
    seed = config.integer(instance_seed, "instance_seed", 0)
    rng = np.random.default_rng(seed)
    D = rng.uniform(0.0, 1.0, (m, 400))
    D[rng.random(D.shape) > 0.5] = 0.0
    return Instance(sparse.csr_matrix(D), rng.uniform(0, 1, m), rng.uniform(0, 1, 400), seed)


def build_original(instance):
    return LinearProgram(instance.D, instance.b, instance.cost,
                         np.zeros(instance.n), np.full(instance.n, np.inf), sense=">")


def solve_original(instance):
    with Timer() as timer:
        result = solve_lp(build_original(instance), config.ORIGINAL_SOLVER, "original_Diet")
    result.cpu_seconds, result.wall_seconds = timer.cpu, timer.wall
    return result


def build_projected(instance, k, projection_seed, variant):
    if variant not in VARIANTS:
        raise ValueError("Diet formulation_variant must be TDP1 or TDP2")
    m, n = instance.m, instance.n
    A = sparse.hstack([instance.D, -sparse.eye(m, format="csr")], format="csr")
    # Both variants use density([D,-I]) and n+m in k, even though TDP2 has n+2k variables.
    T, sigma = sparse_gaussian(A, k, projection_seed)
    mu = 2 * n
    if variant == "TDP1":
        matrix = (T @ A).tocsr()
        c = np.r_[instance.cost, np.full(m, mu)]
        retrieval_lp = LinearProgram(A, instance.b, c, np.zeros(n + m), np.full(n + m, np.inf))
    else:
        identity = sparse.eye(k, format="csr")
        matrix = sparse.hstack([T @ instance.D, identity, -identity], format="csr")
        c = np.r_[instance.cost, np.full(2 * k, mu)]
        # Upstream uses Dq=b for TDP2 retrieval (generally inconsistent), with
        # only q>=0 bounds. Evaluation still uses Dq>=b; do not silently change it.
        retrieval_lp = LinearProgram(instance.D, instance.b, instance.cost,
                                     np.zeros(n), np.full(n, np.inf))
    projected = LinearProgram(matrix, T @ instance.b, c, np.zeros(len(c)), np.full(len(c), np.inf))
    return projected, retrieval_lp, T, sigma


def objective(instance, x):
    return float(instance.cost @ x[:instance.n])


def solve_one_replication(instance, epsilon, projection_seed, original_result, formulation_variant=None):
    if formulation_variant not in VARIANTS:
        raise ValueError("Diet requires formulation_variant='TDP1' or 'TDP2'")
    result = one_replication(instance, epsilon, projection_seed, original_result, formulation_variant,
                             instance.n + instance.m, build_projected, objective,
                             diet_metrics, config.DIET_SOLVER)
    result["problem_type"] = PROBLEM_TYPE
    return result

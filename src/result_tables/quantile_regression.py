"""One QR instance / original solve / replication; no experiment loops.

Adapted from quantreggen.py, QuantReg in rp4lp.py, and the local reference
notebook. The original notebook-facing src/quantile_regression.py is preserved.
"""

import numpy as np
from scipy import sparse
from . import config
from .common import Instance, LinearProgram, Timer
from .gurobi_solver import solve_lp
from .projection import sparse_gaussian
from .pipeline import one_replication
from .metrics import quantile_metrics

PROBLEM_TYPE = "Quantile Regression"
VARIANTS = ("QR",)


def generate_instance(row_size, instance_seed):
    m = config.integer(row_size, "row_size")
    seed = config.integer(instance_seed, "instance_seed", 0)
    rng = np.random.default_rng(seed)
    data = rng.uniform(-1.0, 1.0, (m, 400))
    data[rng.random(data.shape) > 0.8] = 0.0
    return Instance(sparse.csr_matrix(data[:, :399]), data[:, 399].copy(), None, seed)


def build_original(instance):
    m, p = instance.m, instance.n
    identity = sparse.eye(m, format="csr")
    A = sparse.hstack([instance.D, identity, -identity], format="csr")
    c = np.r_[np.zeros(p), np.full(m, instance.tau), np.full(m, 1 - instance.tau)]
    lower = np.r_[np.full(p, -np.inf), np.zeros(2 * m)]
    return LinearProgram(A, instance.b, c, lower, np.full(p + 2 * m, np.inf))


def solve_original(instance):
    with Timer() as timer:
        result = solve_lp(build_original(instance), config.ORIGINAL_SOLVER, "original_QR")
    result.cpu_seconds, result.wall_seconds = timer.cpu, timer.wall
    return result


def build_projected(instance, k, projection_seed, variant="QR"):
    original = build_original(instance)
    T, sigma = sparse_gaussian(original.A, k, projection_seed)
    projected = LinearProgram((T @ original.A).tocsr(), T @ original.b,
                              original.c, original.lower, original.upper)
    return projected, original, T, sigma


def objective(instance, x):
    p, m = instance.n, instance.m
    return float(instance.tau * x[p:p + m].sum() + (1 - instance.tau) * x[p + m:].sum())


def solve_one_replication(instance, epsilon, projection_seed, original_result, formulation_variant=None):
    if formulation_variant not in (None, "QR"):
        raise ValueError("Quantile Regression has only the QR formulation")
    result = one_replication(instance, epsilon, projection_seed, original_result, "QR",
                             instance.n + 2 * instance.m, build_projected, objective,
                             quantile_metrics, config.QUANTILE_SOLVER)
    result["problem_type"] = PROBLEM_TYPE
    return result

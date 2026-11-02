"""Validated experiment grids and explicit, serializable numerical policies."""

from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
import math

PROBLEM_TYPES = ("Quantile Regression", "Diet Problem", "Basis Pursuit", "Max Flow")
IMPLEMENTATION_VERSION = "1.1"
SEED_POLICY = "keyed-sha256-uint64-v2"


def integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def decimal_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be a finite number")
    return result


def epsilon_grid(minimum, maximum, increment):
    """Both endpoints included; reject ranges not divisible by the increment."""
    lo, hi, step = [decimal_number(v, n) for v, n in zip(
        (minimum, maximum, increment),
        ("epsilon_min", "epsilon_max", "epsilon_increment"))]
    if not 0 < lo <= hi < 1 or step <= 0:
        raise ValueError("Require 0 < epsilon_min <= epsilon_max < 1 and increment > 0")
    count = (hi - lo) / step
    if count != count.to_integral_value():
        raise ValueError("The epsilon range must be divisible by epsilon_increment")
    values = [float(lo + i * step) for i in range(int(count) + 1)]
    if any(not 0 < x < 1 for x in values) or len(set(values)) != len(values):
        raise ValueError("Epsilon grid is not representable as distinct finite floats")
    return values


def row_grid(minimum, maximum, increment):
    lo = integer(minimum, "row_min")
    hi = integer(maximum, "row_max")
    step = integer(increment, "row_increment")
    if lo > hi or (hi - lo) % step:
        raise ValueError("Require row_min <= row_max and a range divisible by row_increment")
    return list(range(lo, hi + 1, step))


@dataclass(frozen=True)
class SolverSettings:
    """Edit these internal defaults to change solver policy; metadata records it."""

    method: int = -1
    crossover: int = -1
    threads: int = 1
    feasibility_tol: float = 1e-6
    optimality_tol: float = 1e-6
    bar_conv_tol: float = 1e-8
    numeric_focus: int = 0
    presolve: int = -1
    time_limit: float | None = None
    seed: int = 0
    output_flag: int = 0

    def parameters(self):
        return {
            "Method": self.method, "Crossover": self.crossover,
            "Threads": self.threads, "FeasibilityTol": self.feasibility_tol,
            "OptimalityTol": self.optimality_tol, "BarConvTol": self.bar_conv_tol,
            "NumericFocus": self.numeric_focus, "Presolve": self.presolve,
            "TimeLimit": math.inf if self.time_limit is None else self.time_limit,
            "Seed": self.seed, "OutputFlag": self.output_flag,
        }


@dataclass(frozen=True)
class RetrievalSettings:
    max_iterations: int = 30
    tolerance: float = 0.01
    lsmr_atol: float = 1e-6
    lsmr_btol: float = 1e-6
    lsmr_conlim: float = 1e8
    lsmr_maxiter: int | None = None
    bisection_max_iterations: int = 60


ORIGINAL_SOLVER = SolverSettings()
QUANTILE_SOLVER = SolverSettings(method=1)
DIET_SOLVER = SolverSettings(method=2, crossover=0)
RETRIEVAL = RetrievalSettings()
OBJECTIVE_ZERO_TOL = 1e-9


def policy_metadata():
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "seed_policy": SEED_POLICY,
        "original_solver": asdict(ORIGINAL_SOLVER),
        "quantile_projected_solver": asdict(QUANTILE_SOLVER),
        "diet_projected_solver": asdict(DIET_SOLVER),
        "retrieval": asdict(RETRIEVAL),
        "objective_zero_tolerance": OBJECTIVE_ZERO_TOL,
        "projection": {"C0": 1, "density_factor": 0.5,
                       "support": "fixed count, uniform without replacement",
                       "k": "int(round((1 / epsilon)**2) * log(n_variables))"},
        "quantile": {"predictors": 399, "total_columns": 400, "tau": 0.2,
                     "data_density": 0.8, "nonzero_distribution": "Uniform(-1,1)",
                     "dependent_column": 400, "mask": "Bernoulli on all 400 columns"},
        "diet": {"foods": 400, "data_density": 0.5,
                 "nonzero_distribution": "Uniform(0,1)", "mu": 800},
    }

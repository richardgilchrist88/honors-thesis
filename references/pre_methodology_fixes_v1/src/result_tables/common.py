"""Small data contracts shared by problem modules and the runner."""

from dataclasses import dataclass, field
from time import process_time_ns, perf_counter_ns
import numpy as np
from scipy import sparse


class Timer:
    """CPU includes native Gurobi/BLAS threads in this process; excludes sleep."""

    def __enter__(self):
        self._cpu = process_time_ns()
        self._wall = perf_counter_ns()
        return self

    def __exit__(self, *exc):
        self.cpu = (process_time_ns() - self._cpu) / 1e9
        self.wall = (perf_counter_ns() - self._wall) / 1e9


@dataclass
class Instance:
    D: sparse.csr_matrix
    b: np.ndarray
    cost: np.ndarray | None
    instance_seed: int
    tau: float = 0.2

    @property
    def m(self):
        return self.D.shape[0]

    @property
    def n(self):
        return self.D.shape[1]


@dataclass
class LinearProgram:
    A: sparse.csr_matrix
    b: np.ndarray
    c: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sense: str = "="


@dataclass
class SolveResult:
    status: str
    status_code: int | None = None
    x: np.ndarray | None = None
    objective: float | None = None
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    message: str = ""
    parameters: dict = field(default_factory=dict)
    version: str | None = None
    components: dict = field(default_factory=dict)

    @property
    def optimal(self):
        return (self.status == "OPTIMAL" and self.x is not None
                and self.objective is not None and np.isfinite(self.objective)
                and np.isfinite(self.x).all())

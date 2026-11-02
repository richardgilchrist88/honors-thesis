"""Source, environment, timing, and methodological provenance for every run."""

from importlib import metadata
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def environment_metadata():
    versions = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "gurobipy", "openpyxl"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT,
                                   capture_output=True, text=True, check=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    reference = REPOSITORY_ROOT / "references/rp4lp/provenance.json"
    sources = {str(p.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): file_hash(p)
               for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "libraries": versions, "repository_commit": commit, "repository_dirty": dirty,
        "implementation_sha256": sources,
        "upstream": json.loads(reference.read_text(encoding="utf-8")),
        "paper_sha256": file_hash(REPOSITORY_ROOT / "Liberti et al.pdf"),
        "reference_notebook_sha256": file_hash(REPOSITORY_ROOT / "problems/quantile_regression/Liberti_QuantileRegression_m5000_RP_Retrieval.ipynb"),
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "process_clock": vars(time.get_clock_info("process_time")),
    }


METHODOLOGY = {
    "claim": "Gurobi-based reproduction of Liberti et al.'s methodology, not solver-identical or numerical reproduction of published tables.",
    "timing": {
        "primary": "time.process_time_ns; native Gurobi and BLAS run directly in this process",
        "original": "LP matrix/formulation construction, environment/model setup, optimization, extraction and cleanup",
        "projected": "sum of projection preparation, projected model/solve, and retrieval CPU components",
        "excluded": "random instance generation (recorded separately), reporting/digests, aggregation, exports; no disk instance loading",
        "paper_difference": "Paper includes reading instances and calls timings CPU; upstream uses time.time elapsed clocks. In-memory data replace file loading here.",
        "wall": "time.perf_counter_ns retained separately, never used for the publication CPU ratio",
        "zero_duration": "An original CPU duration of zero gives an undefined CPU ratio (possible for tiny fixtures).",
    },
    "deviations": [
        "Published 5 replications and caller-selected epsilon sweeps replace upstream [0.5]/1 overrides.",
        "Gurobi original Method=-1; projected QR dual simplex Method=1; Diet Method=2/Crossover=0. Threads=1 for repeatability; all policies recorded.",
        "Modern local seeded generators replace upstream global NumPy/Python RNGs and six-decimal AMPL serialization.",
        "QR generator masks all 400 columns as upstream; reference notebook instead uses fixed-count predictor sparsity and a dense response.",
        "Projection uses fixed-count sparse support as upstream and notebook, rather than the paper's independent Bernoulli description.",
        "QR sigma uses actual density([D,I,-I]), correcting upstream p versus p-1 density inconsistency.",
        "QR beta stays free in retrieval; upstream QuantReg omits lower bounds when calling altProjRetrieval.",
        "QR uses [TD,T,-T] directly; inactive upstream AMPL projected model negates the already-negative final block a second time.",
        "Dykstra bounds-first order and LSMR equation (19) retained; stopping errors recomputed on current iterate and zero-length line guarded.",
        "Bisection follows paper's minimization intent with corrected direction, feasible endpoint, bound checks and finite iteration guard; upstream swaps objectives twice without swapping vectors.",
        "Diet TDP1 retrieves [q,r] against [D,-I] with augmented objective; TDP2 retrieves q against Dq=b (usually inconsistent), as upstream. Both are evaluated on food cost and Dq>=b.",
        "Cached Diet original result preserved; upstream Diet overwrites a provided cached fstar/xstar when runOrg is false.",
        "One instance per size reused across epsilons; paired Diet variants share the projection seed and digest.",
        "k>m is executed without clipping; original notebook rejects it. Only Dykstra baseline is used here.",
    ],
    "success_definition": "Both LP solves optimal and retrieval finite. Approximate or infeasible retrieved points remain in averages; feasibility counts and errors are separate.",
    "metrics": {
        "QR_avgin": "sum of negative parts of u_plus and u_minus divided by m",
        "QR_avgeq": "mean(abs(D beta + u_plus - u_minus - b))",
        "Diet_avgin": "mean(max(-q,0))",
        "Diet_avgeq": "mean(max(b-Dq,0)); historical avgeq column name, nutrient inequalities",
        "maxin_QR": "sum of block maxima, as upstream; actual max_bound_violation also recorded",
        "objective_ratios": "Food cost for Diet, excluding penalty; undefined if abs(original objective)<=1e-9",
    },
    "reproducibility": "Seeds reproduce instances and projections with recorded software. Timings vary; floating-point/solver results may vary across platforms or libraries.",
}

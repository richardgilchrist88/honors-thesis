"""The sole owner of instance-size, epsilon, replication and variant loops."""

from pathlib import Path
import hashlib
import json
import re
import secrets
import warnings
import numpy as np
import pandas as pd
from . import config, quantile_regression, diet_problem
from .common import Timer
from .projection import projection_dimension
from .provenance import REPOSITORY_ROOT, METHODOLOGY, environment_metadata
from .reporting import aggregate_trials, make_result, save_outputs

PROBLEMS = {"Quantile Regression": quantile_regression, "Diet Problem": diet_problem,
            "Basis Pursuit": None, "Max Flow": None}


def result_table_generator(problem_type, epsilon_min, epsilon_max, epsilon_increment,
                           row_min, row_max, row_increment, num_runs=5, *,
                           output_dir=None, experiment_name=None, master_seed=None):
    """Run a reproducible Gurobi experiment and export publication-style tables.

    Both range endpoints must lie on their increment grid. Returns ExperimentResult
    with raw_trials, averaged_results, tables, counts, files, metadata and preview().
    Results default to the sibling thesis_results directory, independent of cwd.
    The same named configuration replaces its prior artifacts; configurations with
    different grids, repetitions, explicit seeds or numerical policies coexist.
    """
    if not isinstance(problem_type, str) or problem_type not in config.PROBLEM_TYPES:
        raise ValueError(f"problem_type must be exactly one of {config.PROBLEM_TYPES}")
    epsilons = config.epsilon_grid(epsilon_min, epsilon_max, epsilon_increment)
    sizes = config.row_grid(row_min, row_max, row_increment)
    num_runs = config.integer(num_runs, "num_runs")
    if master_seed is not None:
        master_seed = config.integer(master_seed, "master_seed", 0)
    if experiment_name is not None and (not isinstance(experiment_name, str) or not experiment_name.strip()):
        raise ValueError("experiment_name must be a nonempty string or None")
    if output_dir is not None and (not isinstance(output_dir, (str, Path)) or not str(output_dir).strip()):
        raise ValueError("output_dir must be a nonempty path or None")
    module = PROBLEMS[problem_type]
    if module is None:
        raise NotImplementedError(f"{problem_type} is reserved for a future single-replication solver")
    # Validate representability before solving anything; never cap k at m.
    for size in sizes:
        variables = 399 + 2 * size if problem_type == "Quantile Regression" else 400 + size
        for epsilon in epsilons:
            projection_dimension(epsilon, variables)

    # Fail early if export dependencies are missing, before expensive solves.
    import openpyxl  # noqa: F401
    import matplotlib  # noqa: F401

    parameters = {"problem_type": problem_type, "epsilons": epsilons, "row_sizes": sizes,
                  "num_runs": num_runs, "requested_master_seed": master_seed,
                  "experiment_name": experiment_name, "policies": config.policy_metadata()}
    configuration_hash = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:16]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", experiment_name or problem_type).strip("-_")[:70] or "experiment"
    directory = (Path(output_dir).expanduser() if output_dir is not None
                 else REPOSITORY_ROOT.parent / "thesis_results").resolve() / f"{slug.lower()}-{configuration_hash}"
    # Probe the configured output directory before the expensive pipeline starts.
    directory.mkdir(parents=True, exist_ok=True)
    actual_master_seed = master_seed if master_seed is not None else secrets.randbits(128)
    seed_source = np.random.SeedSequence(actual_master_seed)
    seen_seeds = set()

    def next_seed():
        # Independent SeedSequence children, with explicit collision protection.
        while True:
            value = int(seed_source.spawn(1)[0].generate_state(1, dtype=np.uint64)[0])
            if value not in seen_seeds:
                seen_seeds.add(value)
                return value

    trials, instances = [], []
    for size in sizes:
        instance_seed = next_seed()
        with Timer() as generation:
            instance = module.generate_instance(size, instance_seed)
        original = module.solve_original(instance)
        instances.append({"row_size": size, "instance_seed": instance_seed,
                          "generation_cpu_seconds": generation.cpu,
                          "generation_wall_seconds": generation.wall,
                          "original_status": original.status, "original_message": original.message,
                          "data_shape": list(instance.D.shape), "data_nnz": instance.D.nnz})
        for epsilon in epsilons:
            for replication in range(1, num_runs + 1):
                projection_seed = next_seed()
                for variant in module.VARIANTS:
                    trial = module.solve_one_replication(instance, epsilon, projection_seed, original,
                                                         formulation_variant=variant)
                    trial.update(problem_type=problem_type, replication=replication)
                    trials.append(trial)
    raw = pd.DataFrame(trials)
    # Avoid signed coercion / floating-point seed loss in DataFrames and CSVs.
    for column in ("instance_seed", "projection_seed"):
        raw[column] = pd.Series([r[column] for r in trials], dtype="uint64")
    averaged = aggregate_trials(raw)
    metadata = {"parameters": parameters, "master_seed": actual_master_seed,
                "configuration_hash": configuration_hash, "output_path": str(directory),
                "seed_algorithm": "SeedSequence.spawn children -> uint64 seed -> default_rng (PCG64)",
                "instances": instances,
                "trials": raw[["problem_type", "row_size", "epsilon", "formulation_variant", "replication",
                                "instance_seed", "projection_seed", "success", "projected_status",
                                "projection_digest"]].to_dict(orient="records"),
                "environment": environment_metadata(), "methodology": METHODOLOGY,
                "successful_runs": int(raw.success.sum()), "failed_runs": int((~raw.success).sum())}
    result = make_result(raw, averaged, metadata, directory)
    save_outputs(result)
    if metadata["failed_runs"]:
        warnings.warn(f"{metadata['failed_runs']} trial(s) failed; inspect raw_trials and counts in {directory}", RuntimeWarning)
    return result

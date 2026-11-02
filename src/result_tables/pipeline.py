"""Execute one projected solve/retrieval; experiment repetition lives in runner."""

import json
import numpy as np
from .common import Timer
from .gurobi_solver import solve_lp
from .projection import projection_dimension, projection_fingerprint
from .retrieval import retrieve
from .metrics import ratio
from . import config


def json_text(value):
    """Human-readable solver diagnostics; infinity parameters encoded as strings."""
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x):
            return str(x)
        return x
    return json.dumps(clean(value), sort_keys=True, allow_nan=False)


def one_replication(instance, epsilon, projection_seed, original_result, variant,
                    n_variables, build_projected, objective, evaluate, settings):
    """Shared mechanics; build_projected must return LP, retrieval LP, T, sigma."""
    config.integer(projection_seed, "projection_seed", 0)
    k = projection_dimension(epsilon, n_variables)
    row = {
        "row_size": instance.m, "epsilon": float(epsilon),
        "instance_seed": instance.instance_seed, "projection_seed": int(projection_seed),
        "formulation_variant": variant, "k": k, "k / m": k / instance.m,
        "number_of_lp_variables": n_variables, "data_columns": instance.n,
        "f_original": original_result.objective,
        "original_status": original_result.status,
        "original_status_code": original_result.status_code,
        "original_message": original_result.message,
        "t_original": original_result.cpu_seconds,
        "t_original_wall": original_result.wall_seconds,
        "original_solver_parameters": json_text(original_result.parameters),
        "original_timing_components": json_text(original_result.components),
        "solver_version": original_result.version,
        "success": False, "original_feasible": False,
        "retrieval_feasible": False, "failure_stage": "",
        "message": "", "projected_status": "NOT_RUN", "projected_status_code": None,
        "retrieval_status": "NOT_RUN", "retrieval_iterations": None,
        "projection_digest": None, "f_projected": None, "f_retrieved": None,
        "projected_solver_objective": None, "retrieved_augmented_objective": None,
        "f_projected / f_original": float("nan"),
        "f_retrieved / f_original": float("nan"),
        "avgin": float("nan"), "avgeq": float("nan"),
        "maxin": float("nan"), "maxeq": float("nan"),
        "max_bound_violation": float("nan"), "original_residual_l2": float("nan"),
        "t_projected": float("nan"), "t_projected_wall": float("nan"),
        "t_projected / t_original": float("nan"),
        "projection_cpu": 0.0, "projected_solve_cpu": 0.0, "retrieval_cpu": 0.0,
        "projection_wall": 0.0, "projected_solve_wall": 0.0, "retrieval_wall": 0.0,
    }
    if not original_result.optimal:
        row.update(failure_stage="original", message=original_result.message,
                   projected_status="SKIPPED_ORIGINAL_FAILED")
        return row

    stage = "projection"
    try:
        with Timer() as projection_timer:
            projected_lp, retrieval_lp, T, sigma = build_projected(instance, k, projection_seed, variant)
        row.update(projection_cpu=projection_timer.cpu, projection_wall=projection_timer.wall,
                   projection_sigma=sigma, projection_nnz=T.nnz,
                   projection_density=T.nnz / (k * instance.m),
                   projected_variables=projected_lp.A.shape[1],
                   projected_matrix_nnz=projected_lp.A.nnz,
                   retrieval_variables=retrieval_lp.A.shape[1])
        # Digest and reporting are deliberately outside primary timing scopes.
        row["projection_digest"] = projection_fingerprint(T)
        stage = "projected_solve"
        solved = solve_lp(projected_lp, settings, name=variant)
        row.update(projected_status=solved.status, projected_status_code=solved.status_code,
                   projected_solver_parameters=json_text(solved.parameters),
                   projected_timing_components=json_text(solved.components),
                   projected_solve_cpu=solved.cpu_seconds, projected_solve_wall=solved.wall_seconds,
                   solver_version=solved.version, message=solved.message)
        if not solved.optimal:
            row["failure_stage"] = stage
            return row
        row["projected_solver_objective"] = solved.objective
        row["f_projected"] = objective(instance, solved.x)
        row["projected_constraint_max_error"] = float(np.max(np.abs(projected_lp.A @ solved.x - projected_lp.b)))
        stage = "retrieval"
        with Timer() as retrieval_timer:
            # TDP2 retrieves in food-variable space; QR and TDP1 retain slacks.
            answer = retrieve(solved.x[:retrieval_lp.A.shape[1]], retrieval_lp)
        row.update(retrieval_cpu=retrieval_timer.cpu, retrieval_wall=retrieval_timer.wall,
                   **answer.diagnostics)
        row["lsmr_trace"] = json_text(row["lsmr_trace"])
        row["f_retrieved"] = objective(instance, answer.x)
        row["retrieved_augmented_objective"] = float(retrieval_lp.c @ answer.x)
        row.update(evaluate(instance, answer.x, retrieval_lp))
        if not all(np.isfinite(row[key]) for key in ("f_projected", "f_retrieved", "avgin", "avgeq")):
            raise FloatingPointError("Nonfinite objective or feasibility metrics")
        row["original_feasible"] = (row["max_bound_violation"] <= config.RETRIEVAL.tolerance
                                    and row["original_residual_l2"] <= config.RETRIEVAL.tolerance)
        row["success"] = True  # Finite completed trial, not a feasibility certificate.
        row["f_projected / f_original"] = ratio(row["f_projected"], row["f_original"])
        row["f_retrieved / f_original"] = ratio(row["f_retrieved"], row["f_original"])
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        if stage == "projection":
            row.update(projection_cpu=projection_timer.cpu, projection_wall=projection_timer.wall)
        elif stage == "retrieval":
            row.update(retrieval_cpu=retrieval_timer.cpu, retrieval_wall=retrieval_timer.wall)
        row.update(failure_stage=stage, message=str(exc), retrieval_status="NUMERICAL_ERROR")
    finally:
        row["t_projected"] = sum(row[key] for key in ("projection_cpu", "projected_solve_cpu", "retrieval_cpu"))
        row["t_projected_wall"] = sum(row[key] for key in ("projection_wall", "projected_solve_wall", "retrieval_wall"))
        if row["success"]:
            row["t_projected / t_original"] = ratio(row["t_projected"], row["t_original"], 0.0)
    return row

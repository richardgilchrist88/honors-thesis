"""Run the original quantile regression workflow across observation counts.

Example: python problems/quantile_regression/run_comparison.py
Use --fixed-k 1000 for a fixed projection budget, or --sizes 100 --p 10
for a quick check. Existing completed runs are skipped unless --overwrite.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import gurobipy as gp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from quantile_regression import (
    build_quantile_bounds, build_quantile_lp, evaluate_retrieval,
    generate_projection_matrix, generate_quantile_instance, project_lp,
    retrieval_feasibility, retrieve_aarm, retrieve_apm, retrieve_dykstra, solve_lp,
)


def run_instance(m, data_seed, projection_seed, args, repeat=1):
    """Return four comparison rows for one dataset and projection."""
    k = args.fixed_k if args.fixed_k is not None else round(args.ratio * m)
    if not 0 < k < m:
        raise ValueError("k must lie strictly between 0 and m")

    # generate one quantile regression instance
    D, b = generate_quantile_instance(m, args.p, args.density, data_seed)
    A, c = build_quantile_lp(D, b, args.tau)
    lb, ub = build_quantile_bounds(m, args.p)
    settings = dict(threads=args.threads, solver_seed=args.solver_seed)
    original_timings, projected_timings = {}, {}

    def diagnostic_settings(label, timings):
        if not args.diagnostics:
            return {}
        log_directory = args.output.parent / (args.output.stem + "_logs")
        log_directory.mkdir(parents=True, exist_ok=True)
        log_path = log_directory / f"m{m}_data{data_seed}_projection{projection_seed}_repeat{repeat}_{label}.log"
        print(f"Solving {label} LP; progress log: {log_path}", flush=True)
        return dict(log_file=str(log_path), timings=timings)

    # solve the original LP (including Gurobi model setup)
    original_settings = diagnostic_settings("original", original_timings)
    start_time = time.perf_counter()
    x_original, obj_original = solve_lp(A, b, c, lb, ub, **settings, **original_settings)
    time_original_solve = time.perf_counter() - start_time

    # generate and apply the sparse random projection
    start_time = time.perf_counter()
    T = generate_projection_matrix(A, k, projection_seed)
    time_T_generation = time.perf_counter() - start_time
    start_time = time.perf_counter()
    A_projected, b_projected = project_lp(A, b, T)
    time_projection = time.perf_counter() - start_time

    # solve the projected LP (including Gurobi model setup)
    projected_settings = diagnostic_settings("projected", projected_timings)
    start_time = time.perf_counter()
    x_tilde, obj_tilde = solve_lp(A_projected, b_projected, c, lb, ub, **settings, **projected_settings)
    time_TLP_solve = time.perf_counter() - start_time
    time_projected_base = time_T_generation + time_projection + time_TLP_solve

    metadata = dict(
        m=m, p=args.p, k=k, projection_ratio=k / m, density=args.density, repeat=repeat,
        tau=args.tau, data_seed=data_seed, projection_seed=projection_seed,
        threads=args.threads, solver_seed=args.solver_seed,
        gurobi_version=".".join(map(str, gp.gurobi.version())),
        max_iter=args.max_iter, eq_tol=args.eq_tol, bound_tol=1e-8,
        alpha=0.9, beta=0.9, original_objective=obj_original,
        projected_objective=obj_tilde, time_original_solve=time_original_solve,
        time_T_generation=time_T_generation, time_projection=time_projection,
        time_TLP_solve=time_TLP_solve,
    )
    metadata.update({f"original_{key}": value for key, value in original_timings.items()})
    metadata.update({f"projected_{key}": value for key, value in projected_timings.items()})
    rows = []

    # compare objectives only when ORIGINAL constraints are satisfied
    def record(name, x, steps, status, retrieval_time, pipeline_time):
        objective, eq_residual, bound_error = evaluate_retrieval(x, A, b, c, lb, ub)
        feasible = retrieval_feasibility(x, A, b, lb, ub, eq_tol=args.eq_tol)
        gap = (abs(objective - obj_original) / abs(obj_original)
               if feasible and abs(obj_original) > 1e-12 else np.nan)
        rows.append(dict(
            **metadata, method=name, iterations=steps, stop_reason=status,
            feasible=feasible, objective=objective, objective_per_observation=objective / m,
            relative_objective_error=gap, equality_residual=eq_residual,
            equality_residual_rms=eq_residual / np.sqrt(m),
            equality_residual_max=float(np.max(np.abs(A @ x - b))),
            bound_violation=bound_error, time_retrieval=retrieval_time,
            time_pipeline=pipeline_time, speedup=time_original_solve / pipeline_time,
        ))
        print(f"{name}: feasible={feasible}, time={pipeline_time:.3f}s, gap={gap:.6g}")

    record("Original LP", x_original, 0, "optimal", 0.0, time_original_solve)
    for name, retrieve in [
        ("von Neumann APM", retrieve_apm),
        ("Dykstra APM", retrieve_dykstra),
        ("AARM", retrieve_aarm),
    ]:
        start_time = time.perf_counter()
        x, steps, status = retrieve(
            x_tilde, A, b, lb, ub, max_iter=args.max_iter, delta=args.eq_tol
        )
        retrieval_time = time.perf_counter() - start_time
        record(name, x, steps, status, retrieval_time, time_projected_base + retrieval_time)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[5000, 6000, 7000, 8000, 9000])
    parser.add_argument("--p", type=int, default=399)
    parser.add_argument("--density", type=float, default=0.8)
    parser.add_argument("--tau", type=float, default=0.2)
    projection = parser.add_mutually_exclusive_group()
    projection.add_argument("--ratio", type=float, default=0.2)
    projection.add_argument("--fixed-k", type=int)
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--projection-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--solver-seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--eq-tol", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=ROOT / "results/quantile_regression/comparison.csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="Save solver logs and separate setup/solve timings")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat identical instances to check timing variability")
    args = parser.parse_args()
    if args.max_iter < 1 or args.eq_tol <= 0 or args.threads < 1 or args.repeats < 1:
        parser.error("max-iter, eq-tol, threads, and repeats must be positive")

    # save the experiment settings and resume only matching configurations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.output.with_suffix(".config.json")
    config = {key: value for key, value in vars(args).items()
              if key not in ("output", "overwrite")}
    config["gurobi_version"] = list(gp.gurobi.version())
    # retain compatibility with the original single-run configuration
    if not args.diagnostics:
        config.pop("diagnostics")
    if args.repeats == 1:
        config.pop("repeats")
    rows = []
    if args.output.exists() and not args.overwrite:
        if not config_path.exists() or json.loads(config_path.read_text()) != config:
            parser.error("Existing results use different settings; choose --output or --overwrite")
        rows = pd.read_csv(args.output).to_dict("records")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    completed = {(r["m"], r["data_seed"], r["projection_seed"], r.get("repeat", 1)) for r in rows}
    for m in args.sizes:
        for data_seed in args.data_seeds:
            for projection_seed in args.projection_seeds:
                for repeat in range(1, args.repeats + 1):
                    if (m, data_seed, projection_seed, repeat) in completed:
                        print(f"Skipping completed m={m}, seeds={data_seed}/{projection_seed}, repeat={repeat}")
                        continue
                    print(f"\n===== m={m}, data seed={data_seed}, projection seed={projection_seed}, repeat={repeat} =====")
                    rows.extend(run_instance(m, data_seed, projection_seed, args, repeat))
                    # checkpoint after each complete instance
                    temporary = args.output.with_suffix(".tmp.csv")
                    pd.DataFrame(rows).to_csv(temporary, index=False)
                    temporary.replace(args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

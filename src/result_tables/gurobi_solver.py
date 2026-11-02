"""Only optimization backend used by the new pipeline (lazy Gurobi import)."""

import numpy as np
from .common import SolveResult, Timer

STATUS_NAMES = {
    1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
    5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT",
    9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC",
    13: "SUBOPTIMAL", 14: "INPROGRESS", 15: "USER_OBJ_LIMIT",
    16: "WORK_LIMIT", 17: "MEM_LIMIT",
}


def solve_lp(lp, settings, name="rp_lp"):
    """Construct and solve once. Never query X/ObjVal on a failed solve."""
    result = SolveResult("NOT_STARTED", parameters=settings.parameters())
    with Timer() as total:
        try:
            import gurobipy as gp
        except ImportError as exc:
            result.status = "GUROBI_UNAVAILABLE"
            result.message = str(exc)
        else:
            result.version = ".".join(map(str, gp.gurobi.version()))
            try:
                # Private environment prevents global parameters and log-file side effects.
                with gp.Env(empty=True) as env:
                    env.setParam("OutputFlag", settings.output_flag)
                    env.start()
                    with gp.Model(name, env=env) as model:
                        with Timer() as construct:
                            for key, value in settings.parameters().items():
                                model.setParam(key, value)
                            x = model.addMVar(lp.A.shape[1], lb=lp.lower, ub=lp.upper,
                                             obj=lp.c, name="x")
                            model.addMConstr(lp.A, x, lp.sense, lp.b, name="constraints")
                            model.ModelSense = gp.GRB.MINIMIZE
                            model.update()
                        with Timer() as optimize:
                            model.optimize()
                        result.status_code = model.Status
                        result.status = STATUS_NAMES.get(model.Status, str(model.Status))
                        result.parameters = {key: model.getParamInfo(key)[2]
                                             for key in settings.parameters()}
                        result.components = {
                            "construct_cpu": construct.cpu, "optimize_cpu": optimize.cpu,
                            "construct_wall": construct.wall, "optimize_wall": optimize.wall,
                            "gurobi_runtime_wall": model.Runtime,
                            "gurobi_iterations": model.IterCount,
                            "gurobi_barrier_iterations": model.BarIterCount,
                        }
                        if model.Status == gp.GRB.OPTIMAL and model.SolCount > 0:
                            candidate = np.asarray(x.X, dtype=float)
                            objective = float(model.ObjVal)
                            if np.isfinite(candidate).all() and np.isfinite(objective):
                                result.x, result.objective = candidate, objective
                            else:
                                result.status = "NONFINITE_SOLUTION"
                        elif model.Status == gp.GRB.OPTIMAL:
                            result.status = "NO_SOLUTION"
            except gp.GurobiError as exc:
                result.status = "GUROBI_ERROR"
                result.message = f"Gurobi error {exc.errno}: {exc}"
    result.cpu_seconds, result.wall_seconds = total.cpu, total.wall
    return result

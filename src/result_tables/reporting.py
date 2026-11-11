"""Paper-style summaries and staged CSV/Excel/LaTeX/HTML/PNG exports."""

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from .metrics import TABLE_COLUMNS
from .retrieval_trace import TRACE_COLUMNS

TRIAL_ID_COLUMNS = ["problem_type", "row_size", "epsilon", "replication",
                    "formulation_variant", "instance_seed", "projection_seed"]


def iteration_table(raw):
    """Flatten histories without coercing uint64 seeds through floating point."""
    records = []
    for trial in raw.to_dict(orient="records"):
        history = trial.get("retrieval_trace", [])
        if isinstance(history, str):
            history = json.loads(history)
        if not isinstance(history, list):
            continue  # Failed/untraced trials have no invented iteration rows.
        identity = {key: trial.get(key) for key in TRIAL_ID_COLUMNS}
        records.extend(identity | row for row in history)
    return pd.DataFrame.from_records(records, columns=TRIAL_ID_COLUMNS + TRACE_COLUMNS)


def format_run_log(result):
    lines = [f"Output: {result.output_path}",
             f"Master seed: {result.metadata.get('master_seed')}",
             "Parameters: " + json.dumps(result.metadata.get("parameters", {}), sort_keys=True),
             "Tracing overhead is included in retrieval timing; file exports are excluded."]
    for trial in result.raw_trials.to_dict(orient="records"):
        lines.append("\n" + ", ".join(f"{key}={trial.get(key)}" for key in TRIAL_ID_COLUMNS))
        lines.append(trial.get("retrieval_explanation", "No retrieval explanation available."))
        lines.append(f"Original-space feasible: {trial.get('original_feasible')}; "
                     f"completed trial: {trial.get('success')}; "
                     f"failure stage: {trial.get('failure_stage', '')}.")
        if not trial.get("success"):
            lines.append(f"Solver statuses: original={trial.get('original_status')}, "
                         f"projected={trial.get('projected_status')}; {trial.get('message', '')}")
    return "\n".join(lines) + "\n"

GROUP_COLUMNS = ["problem_type", "row_size", "epsilon", "formulation_variant"]
AVERAGE_METRICS = TABLE_COLUMNS[1:] + [
    "f_original", "f_projected", "f_retrieved", "projected_solver_objective",
    "retrieved_augmented_objective", "maxin", "maxeq", "max_bound_violation",
    "original_residual_l2", "k", "t_original", "t_projected", "t_original_wall",
    "t_projected_wall", "projection_cpu", "projected_solve_cpu", "retrieval_cpu",
    "retrieval_iterations", "retrieval_equality_l2", "retrieval_bound_error_sum",
]
FORMATS = {name: "{:.4f}" for name in TABLE_COLUMNS}
FORMATS.update({"epsilon": "{:.2f}", "t_projected / t_original": "{:.2f}"})


def aggregate_trials(raw):
    """Average replication ratios, keep every group, expose failures and undefined ratios."""
    records = []
    for keys, group in raw.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        successful = group.loc[group.success]
        row = dict(zip(GROUP_COLUMNS, keys))
        row.update(num_runs=len(group), successful_runs=len(successful),
                   failed_runs=len(group) - len(successful),
                   original_feasible_runs=int(successful.original_feasible.sum()),
                   retrieval_feasible_runs=int(successful.retrieval_feasible.sum()))
        for metric in AVERAGE_METRICS:
            values = pd.to_numeric(successful[metric], errors="coerce") if metric in successful else pd.Series(dtype=float)
            row[metric] = float(values.mean()) if values.notna().any() else np.nan
        # Dimensions are known even if no solver succeeds, never fabricated errors.
        row["k"] = int(group.k.iloc[0])
        row["k / m"] = float(group["k / m"].iloc[0])
        for metric, label in ((TABLE_COLUMNS[1], "projected_objective_ratio_runs"),
                              (TABLE_COLUMNS[2], "retrieved_objective_ratio_runs"),
                              (TABLE_COLUMNS[6], "cpu_ratio_runs")):
            row[label] = int(pd.to_numeric(successful[metric], errors="coerce").notna().sum())
        records.append(row)
    return pd.DataFrame.from_records(records).sort_values(GROUP_COLUMNS).reset_index(drop=True)


@dataclass
class ExperimentResult:
    raw_trials: pd.DataFrame
    averaged_results: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    counts: dict[str, pd.DataFrame]
    metadata: dict
    output_path: Path
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def retrieval_trace(self):
        """One row per stopping-check stage, identified by trial and method."""
        return iteration_table(self.raw_trials)

    def preview(self, formulation_variant=None):
        """A Jupyter-renderable table with visible per-row success/failure counts."""
        if formulation_variant is None:
            if len(self.tables) != 1:
                raise ValueError("Choose TDP1 or TDP2 for a Diet preview")
            formulation_variant = next(iter(self.tables))
        table = self.tables[formulation_variant]
        counts = self.counts[formulation_variant]
        annotations = "; ".join(
            f"m={int(r.row_size)}, epsilon={r.epsilon:g}: {int(r.successful_runs)} successful, "
            f"{int(r.failed_runs)} failed, {int(r.original_feasible_runs)} feasible"
            for r in counts.itertuples(index=False))
        return table.style.format(FORMATS, na_rep="undefined").set_caption(
            f"{formulation_variant} — finite completed runs are averaged; feasibility is separate. {annotations}")


def make_result(raw, averaged, metadata, output_path):
    tables, counts = {}, {}
    for variant, group in averaged.groupby("formulation_variant", sort=True):
        ordered = group.sort_values(["row_size", "epsilon"])
        tables[variant] = ordered.set_index("row_size")[TABLE_COLUMNS].copy()
        counts[variant] = ordered[["row_size", "epsilon", "num_runs", "successful_runs", "failed_runs",
                                   "original_feasible_runs", "retrieval_feasible_runs",
                                   "projected_objective_ratio_runs", "retrieved_objective_ratio_runs",
                                   "cpu_ratio_runs"]].reset_index(drop=True)
    return ExperimentResult(raw, averaged, tables, counts, metadata, output_path)


def latex_table(result, variant):
    group = result.averaged_results.query("formulation_variant == @variant")
    lines = [r"\begin{tabular}{r|rrrrrr}",
             r"$\epsilon$ & $\bar f/f^\ast$ & $\tilde f/f^\ast$ & \texttt{avgin} & \texttt{avgeq} & $k/m$ & $\tilde t/t^\ast$ \\ \hline"]
    for size, rows in group.groupby("row_size", sort=True):
        lines.append(r"\multicolumn{7}{c}{" + f"{variant}, m={int(size)}" + r"} \\ \hline")
        for _, row in rows.sort_values("epsilon").iterrows():
            values = ["--" if pd.isna(row[c]) else FORMATS[c].format(row[c]) for c in TABLE_COLUMNS]
            lines.append(" & ".join(values) + r" \\")
        counts = "; ".join(
            f"{r.epsilon:.2f}: {int(r.successful_runs)}/{int(r.failed_runs)}"
            for r in rows.itertuples())
        lines.append(r"\multicolumn{7}{l}{\scriptsize $\epsilon$: successful/failed = " + counts + r"} \\")
    lines.extend([r"\hline\end{tabular}",
                  "% -- denotes an undefined measure; see counts CSV for valid-ratio and feasibility counts.",
                  "% Success means optimal LP solves and finite retrieval; it does not imply retrieved feasibility."])
    return "\n".join(lines) + "\n"


def save_plots(result, variant, directory):
    # Object-oriented Agg canvas avoids changing a Jupyter session's backend.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    groups = list(result.averaged_results.query("formulation_variant == @variant").groupby("row_size", sort=True))
    categories = {
        "objectives": (TABLE_COLUMNS[1:3], "Objective ratio"),
        "feasibility": (["avgin", "avgeq"], "Average violation"),
        "size_cpu": (["k / m", "t_projected / t_original"], "Ratio"),
    }
    for category, (metrics, ylabel) in categories.items():
        figure = Figure(figsize=(8, 3.1 * len(groups)), layout="constrained")
        FigureCanvasAgg(figure)
        axes = np.atleast_1d(figure.subplots(len(groups), 1))
        for axis, (size, rows) in zip(axes, groups):
            rows = rows.sort_values("epsilon")
            for metric in metrics:
                axis.plot(rows.epsilon, rows[metric], marker="o", label=metric)
            failed = int(rows.failed_runs.sum())
            axis.set(title=f"{variant}: m={int(size)} ({failed} failed trials)", xlabel="epsilon", ylabel=ylabel)
            axis.grid(alpha=0.25)
            axis.legend(fontsize="small")
        figure.savefig(directory / f"{variant}_{category}.png", dpi=160)
        figure.clear()


def save_outputs(result):
    """Stage all artifacts before replacement; restore old files if commit fails.

    Only the explicitly generated filenames are replaced. Metadata (with hashes)
    is committed last. A per-experiment lock rejects simultaneous writers.
    """
    directory = result.output_path.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".result_table_generator.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Output is locked by another writer: {lock}") from exc
    os.close(lock_fd)
    try:
        with tempfile.TemporaryDirectory(prefix=".staging-", dir=directory) as temporary:
            stage = Path(temporary).resolve()
            if not stage.is_relative_to(directory):
                raise RuntimeError("Staging directory escaped experiment output")
            result.raw_trials.to_csv(stage / "raw_trials.csv", index=False, float_format="%.17g")
            result.retrieval_trace.to_csv(stage / "retrieval_trace.csv", index=False, float_format="%.17g")
            (stage / "run.log").write_text(format_run_log(result), encoding="utf-8")
            configuration = {key: result.metadata.get(key) for key in
                             ("parameters", "master_seed", "configuration_hash", "environment")}
            (stage / "config.json").write_text(json.dumps(configuration, indent=2, allow_nan=False) + "\n",
                                              encoding="utf-8")
            result.averaged_results.to_csv(stage / "averaged_results.csv", index=False, float_format="%.17g")
            with pd.ExcelWriter(stage / "results.xlsx", engine="openpyxl") as workbook:
                excel_raw = result.raw_trials.copy()
                # Excel's 15-digit numeric precision would otherwise destroy 64-bit seeds.
                for col in ("instance_seed", "projection_seed"):
                    excel_raw[col] = excel_raw[col].map(str)
                excel_raw.to_excel(workbook, sheet_name="raw_trials", index=False)
                result.averaged_results.to_excel(workbook, sheet_name="averaged_results", index=False)
                for variant in result.tables:
                    result.tables[variant].to_excel(workbook, sheet_name=variant)
                    result.counts[variant].to_excel(workbook, sheet_name=f"{variant}_counts", index=False)
            for variant, table in result.tables.items():
                table.to_csv(stage / f"table_{variant}.csv", float_format="%.17g")
                result.counts[variant].to_csv(stage / f"counts_{variant}.csv", index=False)
                (stage / f"table_{variant}.tex").write_text(latex_table(result, variant), encoding="utf-8")
                (stage / f"preview_{variant}.html").write_text(result.preview(variant).to_html(), encoding="utf-8")
                save_plots(result, variant, stage)
            names = sorted(p.name for p in stage.iterdir() if p.is_file())
            result.metadata["artifacts"] = {
                name: hashlib.sha256((stage / name).read_bytes()).hexdigest() for name in names}
            (stage / "metadata.json").write_text(json.dumps(result.metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            names.append("metadata.json")
            backup = stage / "backup"
            backup.mkdir()
            replaced = []
            try:
                for name in names:
                    destination = directory / name
                    if destination.is_symlink():
                        raise ValueError(f"Refusing to overwrite a symlink: {destination}")
                    if destination.exists():
                        shutil.copy2(destination, backup / name)
                    os.replace(stage / name, destination)
                    replaced.append(name)
            except BaseException:
                for name in reversed(replaced):
                    if (backup / name).exists():
                        os.replace(backup / name, directory / name)
                    else:
                        (directory / name).unlink()
                raise
            result.files = {name: directory / name for name in names}
    finally:
        lock.unlink()

# Result Table Generator

The public function runs Quantile Regression and both Diet projected formulations,
using Gurobi for every LP. Basis Pursuit and Max Flow are recognized names and raise
`NotImplementedError`. The [revision 1.1 corrections](methodology-fixes.md) update the
legacy notebooks and helper; their original versions are archived separately.

These are **Gurobi-based reproductions of Liberti et al.'s experimental methodology**.
They are not exact reproductions of the published numerical results. Read
[the methodology audit](result-table-methodology.md) before making comparisons.

## Install and run

From the repository root, using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-result-tables.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Gurobi must have a working license appropriate to the problem size. No AMPL, CPLEX,
or HiGHS dependency is used in this pipeline. Numerical tests run without licensed
optimization; the direct Gurobi integration tests skip explicitly if unavailable.

```python
from src.result_table_generator import result_table_generator

result = result_table_generator(
    problem_type="Quantile Regression",
    epsilon_min=0.2, epsilon_max=0.5, epsilon_increment=0.3,
    row_min=450, row_max=450, row_increment=1,
    num_runs=2,
    output_dir="results/result_table_generator_smoke",  # optional override
    experiment_name="qr-smoke",
    master_seed=2026,
)
print(result.averaged_results)
print(result.output_path)
# In Jupyter:
display(result.preview("QR"))
display(result.counts["QR"])
```

The runnable examples are in
[`problems/result_table_generator.ipynb`](../problems/result_table_generator.ipynb).
The notebook now has `RUN_PUBLISHED = True` saved. Select the project's `.venv`
kernel, restart it, then **Run All** to execute the small examples and all published
QR/Diet configurations. Set the switch false for only the small examples. The
command `python scripts/validate_result_tables.py --smoke-only` safely forces the
switch false in memory without editing the saved notebook.
The [validation record](result-table-validation.md) lists the executed checks,
smoke outcomes, and generated artifact locations.

Required parameters are `problem_type`, `epsilon_min`, `epsilon_max`,
`epsilon_increment`, `row_min`, `row_max`, `row_increment`; `num_runs` defaults to 5.
The four exact names are `"Quantile Regression"`, `"Diet Problem"`, `"Basis Pursuit"`,
and `"Max Flow"`. Epsilon must lie strictly between zero and one. Sizes and repetition
counts are positive integers; booleans are rejected. Seeds are nonnegative integers.

Ranges include both endpoints. The difference between endpoints must be divisible
by the increment; a nonaligned maximum is rejected rather than silently omitted or
added as a shorter final step. Decimal arithmetic prevents the usual `0.1` endpoint
error. There is no `k <= m` restriction. Very small epsilons may require large matrices.

## Published configurations

Use the same function for all sweeps (each with `num_runs=5`):

| Paper table | problem_type | rows: min, max, increment | epsilon: min, max, increment |
|---|---|---|---|
| 5 | Quantile Regression | 5000, 9000, 1000 | 0.2, 0.9, 0.1 |
| 6 | Quantile Regression | 5000, 9000, 1000 | 0.1, 0.15, 0.05 |
| 3 and 4 | Diet Problem | 3000, 7000, 1000 | 0.2, 0.9, 0.1 |

A Diet call produces TDP1 and TDP2 tables together. Each pair uses the same original
instance and projection seed. Each variant is timed independently, including its
projection construction. The digest recorded in `projection_digest` verifies that
the resulting matrices are identical. No other retrieval methods are run.

With the same master seed and problem, a row size now has the same original
instance across different epsilon grids, repetition counts and row subsets.
Overlapping epsilon/replication pairs also retain their projection seeds. This
allows Tables 5 and 6 to compare epsilons on matching original QR data.

## Results and files

`ExperimentResult` exposes `raw_trials`, `averaged_results`, `tables`, `counts`,
`metadata`, `output_path`, `files`, and `preview(variant)`. Table variants are `QR`,
`TDP1`, and `TDP2`. The table index is the original row size. Its seven columns are:

1. `epsilon`
2. `f_projected / f_original`
3. `f_retrieved / f_original`
4. `avgin`
5. `avgeq`
6. `k / m`
7. `t_projected / t_original`

The default output root is `thesis_results` beside this repository, independent of
the current working directory. An explicit `output_dir` overrides it. Each experiment
has a directory consisting of a readable name and a configuration hash. Changing the
grid, replication count, explicit master seed, experiment name, or recorded numerical
policy creates a different directory. The same configuration replaces prior files.
With `master_seed=None`, a fresh seed is recorded on each invocation while retaining
the same configuration directory. Use an explicit seed to repeat the same random data.

Files inside each directory:

| File | Contents |
|---|---|
| `raw_trials.csv` | Every attempted replication, seeds, statuses, objectives, diagnostics, timing components |
| `averaged_results.csv` | High-precision means and success/failure/feasibility/defined-ratio counts |
| `results.xlsx` | Raw and averaged sheets, one table and count sheet per variant |
| `table_<variant>.csv` | Seven-column table with row-size index, full precision |
| `table_<variant>.tex` | Grouped publication table; epsilon/time at 2 decimals, other metrics at 4 |
| `counts_<variant>.csv` | Per-group successful, failed, feasible, and defined-ratio counts |
| `preview_<variant>.html` | Formatted Jupyter-style preview with explicit counts |
| `<variant>_objectives.png` | Both objective ratios versus epsilon, grouped by size |
| `<variant>_feasibility.png` | Both average errors versus epsilon, grouped by size |
| `<variant>_size_cpu.png` | Projection-size and CPU-time ratios versus epsilon |
| `metadata.json` | Parameters, every seed, policies, versions, source hashes, hardware, artifact hashes |

All outputs are staged before replacement. Existing owned files are backed up during
replacement and restored if replacement raises an error. Metadata is written last
and contains hashes of the other files, so readers can verify a complete generation.
This is not a crash-atomic filesystem transaction: a process/power failure during
replacement can leave a partial generation. Only owned filenames are replaced;
unrelated files are retained. A lock rejects simultaneous export writers. If a process
is forcibly terminated, verify it is no longer running before removing its leftover
`.result_table_generator.lock` and rerunning the same configuration.

Excel stores 64-bit seeds as **text**, preserving all digits. CSVs and JSON also retain
exact seeds; when reading CSVs, use `dtype={"instance_seed": "uint64",
"projection_seed": "uint64"}` (or strings). Other CSV floating-point fields use 17
significant digits. Publication rounding occurs only in the HTML/LaTeX display.

## Success, feasibility, and timing

A successful trial has optimal original/projected solver statuses and a finite
retrieved point. As in the paper, approximate retrieved points remain in the averages
even if they are not feasible. `original_feasible`, `retrieval_feasible`, the remaining
errors, and corresponding counts are reported separately. This distinction matters
especially for Diet TDP2. A stopping reason such as `stagnated` is not a feasibility
certificate. The original-space feasibility flag requires residual L2 and maximum
bound violation both at most 0.01; Diet's residual is its nutrient shortfall vector.
The retrieval-space flag uses equality L2 and summed bound error, both at most 0.01.

Failed LPs never have fabricated primal vectors or objectives. All failed replications
remain in the raw dataset and failure counts; their metrics are excluded from means.
An all-failed group has undefined objective/error/time averages but retains its known
dimension. Objective ratios are undefined for `abs(f_original) <= 1e-9`. This happens
naturally in small QR fixtures with fewer records than predictors. Valid-ratio counts
are recorded separately from successful-trial counts.

Primary timing is process CPU time, including native Gurobi/BLAS work executed in the
same process. Original timing includes formulation construction and the complete
Gurobi call; projected timing is the sum of projection preparation, projected LP
construction/solve, and retrieval. Instance generation is measured separately and
excluded from both pipelines. No on-disk instance loading is needed. Formatting,
hashing, metadata, plots, and aggregation are excluded. Wall time is diagnostic only.
For very small solves the CPU timer can report zero; that CPU ratio is undefined.

## Reproduce one replication

```python
from src.result_tables import quantile_regression as qr

trial = result.raw_trials.iloc[0]
instance = qr.generate_instance(int(trial.row_size), int(trial.instance_seed))
original = qr.solve_original(instance)
replayed = qr.solve_one_replication(
    instance, float(trial.epsilon), int(trial.projection_seed), original,
    formulation_variant="QR",
)
assert replayed["projection_digest"] == trial.projection_digest
```

For Diet, import `diet_problem` and pass `"TDP1"` or `"TDP2"`. Single-replication
functions contain no sweeps or baseline solves. Seeds reproduce the matrices and
projection under the recorded software versions; elapsed/CPU timings are measurements
and will vary. Numerical values can also vary with solver/library/platform versions.

Internal solver choices are dataclasses in `src/result_tables/config.py`:
`ORIGINAL_SOLVER`, `QUANTILE_SOLVER`, `DIET_SOLVER`, `RETRIEVAL`. For example, replace a
solver configuration using `dataclasses.replace(config.ORIGINAL_SOLVER, threads=2,
time_limit=60)` before calling the generator. Such changes are reflected in metadata
and the configuration hash. No optimizer backend switch or extra experimental
algorithm is included.

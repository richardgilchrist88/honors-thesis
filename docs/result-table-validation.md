# Result Table Generator validation

**Iteration logging update (2026-11-10):** 49 tests passed, including all licensed
Gurobi checks, branch-precedence/zero-update/iteration-cap cases, trace seed
precision, failed-trial exports, and numerical-failure explanations. Twelve
seeded method comparisons (three seeds, VN/Dykstra/AARM and pipeline Dykstra)
matched the pre-change solution vectors exactly, with identical iteration
counts/statuses and existing pipeline diagnostics. Traced/untraced comparisons
also passed. This checks logging invariance, not agreement with the paper.

`python scripts/validate_retrieval_logging.py` executed the reference QR notebook
with 450 rows **in memory**, plus one QR and one paired Diet projection through
the standard exporter. Saved trace identities/seeds, stopping branches,
configuration and console/log consistency passed. Samples are in
`results/retrieval_logging_sample/`, including `reference-qr-m450/run.log`,
`reference-qr-m450/console.log`, `reference-qr-m450/retrieval_trace.csv`, and
`validation.json`. The saved reference remains 5000 rows; no published sweeps
were executed. Windows private staging directories required the tests/export
validation to run outside the filesystem sandbox. See the usage guide for the
logging schema, timing overhead, and rerun instructions.

**Revision 1.1:** 41 tests passed and the updated table notebook smoke examples
completed using `python scripts/validate_result_tables.py --smoke-only`. See
[methodology-fixes.md](methodology-fixes.md) for current outcomes and artifact paths.
The remainder of this document is the historical revision 1.0 validation record.

Subsequent repository-wide findings are recorded in [methodology-audit.md](methodology-audit.md).
The smoke entry point now checks `RUN_PUBLISHED = False` before any notebook cell
executes. Three focused guard tests passed after this change; the original 28-test
and solver-smoke results below describe the earlier validation run and were not
rerun while the user's full experiment was active.

Validation performed on 2026-11-02 in the repository's Python 3.14 environment,
with the installed Gurobi 13.0.3 and a working license. Full published sweeps were
not executed. `openpyxl==3.1.5` and its `et-xmlfile==2.0.0` dependency were installed
into the workspace virtual environment for Excel export.

## Commands and outcomes

```powershell
.\.venv\Scripts\python.exe -m compileall -q src/result_tables src/result_table_generator.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 scripts/validate_result_tables.py
```

- Compilation passed.
- **28 automated tests passed**, including 7 licensed Gurobi integration tests.
  The suite covers validation and decimal grids; published data dimensions and
  distributions; projection rounding, support, scaling and seeds; free QR bounds
  and slack signs; both Diet formulations; Dykstra diagnostics and bisection;
  problem-specific metrics; zero denominators; original caching; paired seeds;
  means and failure counts; unavailable/infeasible/time-limited solves; all-failed
  exports; exact seed storage in Excel; replacement without appended rows; unrelated
  file preservation; and rollback after staged plotting and commit failures.
- All Python cells in the example notebook executed in order with the published
  sweep guard **false**, using the smoke validation script. The notebook schema was
  also validated with `nbformat`. The script executes cells as ordinary Python;
  it does not certify an interactive Jupyter kernel session.
- The 12 downloaded upstream files were checked against their Git blob SHA-1
  values in `references/rp4lp/provenance.json`; all matched byte-for-byte.
- The saved QR objective plot was visually inspected; the grouped LaTeX output,
  CSV values, workbook sheets, seeds and artifact checksums were inspected/tested.

The initial sandboxed test attempt passed the numerical/Gurobi checks but could not
access Python's private temporary directories under the Windows sandbox token.
The suite and artifact smoke execution were rerun outside that sandbox after
automatic approval, and completed successfully. This was a filesystem permission
issue, not a failed numerical assertion. Ordinary user execution is the intended
runtime. No full-size optimization run or assertion of published numerical agreement
was made.

## Small experiment results

Both smoke experiments use master seed 2026, epsilon values 0.2 and 0.5, and two
replications per epsilon. The actual generated instance/projection seeds are in each
experiment's raw CSV and metadata. All LP solves were optimal.

| Experiment | Row size | Trials | Finite completed trials | Original-space feasible retrievals |
|---|---:|---:|---:|---:|
| QR, 399 predictors | 450 | 4 | 4 | 4 |
| Diet, 400 foods, both variants | 80 | 8 | 8 | 2 |

The QR original objective was 9.855423899368253; its retrieved objective ratios in
these small diagnostic instances were approximately 3.68–5.09 after averaging.
The Diet original food-cost objective was 0.019035609896748146. Large relative
objective changes and infeasible Diet retrievals are retained as measurements.
These small sizes are not proxies for the paper's published row ranges.

The smoke script also replayed one QR trial from its recorded seeds and verified
the projection digest. All corresponding Diet pairs had identical digests.
Projected CPU totals equaled the sum of projection, solve and retrieval CPU
components; CPU ratios used those totals divided by original CPU time.

For the process-clock check, native NumPy work consumed 0.703125 seconds of process
CPU, matching `os.times()` user+system CPU exactly in that run, while wall time was
0.0588877 seconds. The multi-threaded native work is thus represented in process CPU.
An 0.08-second sleep consumed 0 CPU seconds. These durations are diagnostic
observations, not reproducible performance targets.

Artifacts are stored under the explicit in-repository smoke override:

- `results/result_table_generator_smoke/qr-smoke-ae76f0f65ac8ec74/`
- `results/result_table_generator_smoke/diet-smoke-dbc808414e4c8764/`
- `results/result_table_generator_smoke/validation.json`

The outputs include raw/averaged CSVs, Excel workbooks, seven-column LaTeX and HTML
tables, count files, all three plot categories, and JSON metadata. These generated
artifacts are ignored by Git and can be regenerated with the script above. The
ordinary function's default output root remains the sibling `../thesis_results/`.

## Implementation files

- `src/result_table_generator.py`: public function export.
- `src/result_tables/__init__.py`, `runner.py`: package API and experiment loops.
- `src/result_tables/config.py`, `common.py`: policies, validation and data contracts.
- `src/result_tables/quantile_regression.py`, `diet_problem.py`: single-instance and
  single-replication problem implementations.
- `src/result_tables/gurobi_solver.py`: direct solver and status handling.
- `src/result_tables/projection.py`, `retrieval.py`: seeded projection and retrieval.
- `src/result_tables/pipeline.py`, `metrics.py`: one replication and measurements.
- `src/result_tables/reporting.py`, `provenance.py`: exports and provenance.
- `tests/test_result_table_core.py`, `test_result_table_gurobi.py`,
  `test_result_table_runner.py`: automated validation.
- `problems/result_table_generator.ipynb`: lightweight examples.
- `scripts/validate_result_tables.py`: notebook-cell smoke runner and clock check.
- `requirements-result-tables.txt`: dependencies for the new pipeline.
- `docs/result-table-generator.md`, `result-table-methodology.md`, this file:
  usage, source audit and validation evidence.
- `references/rp4lp/`: unmodified inspected authors' files, license and provenance.

`README.md` and `.gitignore` were updated for discoverability and generated output.
The existing shared QR helper, reference notebook and toy notebooks were preserved.
Two pre-existing deleted comparison notebooks and an untracked `debug.log` were left
as found.

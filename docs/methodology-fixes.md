# Implemented methodology corrections — revision 1.1

The user authorized these changes after the [repository audit](methodology-audit.md).
The active notebooks now use the corrected routines. The 19 pre-change source and
notebook files are preserved byte-for-byte under
[`references/pre_methodology_fixes_v1/`](../references/pre_methodology_fixes_v1/manifest.json),
with a SHA-256 manifest. The authors' files in `references/rp4lp/` are unchanged.

## Run the complete QR and Diet experiment

1. Open [`problems/result_table_generator.ipynb`](../problems/result_table_generator.ipynb).
2. Select the project's `.venv` Python kernel.
3. **Restart the kernel**, then click **Run All** in the notebook toolbar.

`RUN_PUBLISHED = True` is already saved in the first code cell. This runs the small
checks, QR Tables 5 and 6, both Diet variants for Tables 3–4, and a seed replay.
The published configurations contain 650 projected solves, plus the small checks.
The full published sweeps have not been executed as part of this validation.

Full results go to `C:\Users\richa\OneDrive\Documents\thesis_results`. Each sweep
prints its output folder when complete. The implementation version and seed policy
are part of the configuration hash, so revision 1.1 creates different folders from
revision 1.0. Repeating the same **1.1** configuration replaces its own files as before.
Restarting is necessary because an existing kernel can retain the old imported code.

## Changes

| Area | Correction |
|---|---|
| Table generator seeds | Instance seed keyed by master seed, problem and row size; projection seed additionally keyed by epsilon and replication. Neither depends on loop order, other sizes, grid length or repetition count. Diet pairs still share T. |
| Output identity | Implementation version `1.1`, seed policy `keyed-sha256-uint64-v2`; recorded in metadata and the configuration hash. |
| QR comparison methods | `src/retrieval_methods.py` supplies the notebook and legacy helper. VN uses affine-then-bounds; Dykstra uses the generator's bounds-first implementation; AARM composes both reflections in the paper's order. |
| QR feasibility | Checks equality L2 and summed bound violations separately, each <= 0.01. A stagnation or iteration-limit status is not labeled feasible. Free beta bounds are retained. |
| QR postprocessing | The comparison notebook now applies the generator's minimization bisection after each method and includes it in retrieval wall time. An infeasible starting endpoint causes an explicit skip. |
| QR solver helper | Explicit `method` and `model_name` options; the notebook uses automatic original and dual-simplex projected solves, one thread, and disposes models. |
| Projection dimensions | Reference notebook and legacy helper now allow k >= m, consistent with the table generator; no artificial clipping. |
| Toy retrieval | Projects onto the entire interval `0 <= x <= upper_bound`; checks equality and both bounds before stopping. Raw results, traces, summary counts and plots expose upper-bound errors. |
| Toy projection/linear algebra | Three-point scaling corrected to sqrt(3/k); approximate equality correction uses LSMR on A instead of A A.T. The toy exact mode still demonstrates Equation (5). |
| Toy reproducibility/timing | GLOBAL_SEED actually controls instance seeds; seeds/statuses and environment metadata saved; root discovery independent of notebook working directory; projected wall timing includes T, TA/Tb and model construction. |
| Stale claims | Edited notebook outputs cleared; the reference's hardcoded numerical conclusion replaced with instructions to interpret the current errors and feasibility flag. |
| Solver import timing | The runner loads Gurobi before either timed pipeline, avoiding a first-import charge on the first original LP. |

The key is encoded as canonical JSON (epsilon uses its exact float hexadecimal
representation), hashed with SHA-256, and the first eight bytes are interpreted as
an unsigned big-endian seed for NumPy's local generator. Python's randomized
`hash()` is not used. All changed seeds are recorded; old recorded instance and
projection seeds remain usable through the single-replication interface.

Toy outputs were regenerated in `results/toy_example_v2/` and
`results/gurobi_toy_example_v2/`; historical toy CSVs and plots were hash-checked
and left unchanged. The reference's original dimensions, seed-42 data generator,
fixed-count predictor sparsity and dense response are retained. Its old numerical
results remain in the archived notebook, not displayed as new results.

## Verification

- **41 automated tests passed**, including licensed Gurobi integration, export
  rollback, cross-grid/cross-size seed identity, the audit's retrieval witnesses,
  custom upper bounds, dependent-equation LSMR, and smoke-execution guards.
- The table notebook ran with full sweeps disabled **in memory**: all four QR and
  eight Diet smoke trials completed; all Diet pairs had identical projection
  digests; seed replay, CPU component accounting and CSV/Excel/HTML/LaTeX/PNG
  exports passed. Feasible counts were 4/4 QR and 2/8 Diet.
- All code cells of both toy notebooks executed at their configured sizes, 50
  trials each. APM upper/lower-bound violation was zero for every trial. None
  met the strict equality tolerance within the retained 30-iteration cap;
  their new feasibility flags report this instead of implying feasibility.
- All code cells of the QR comparison notebook executed with **450 rows in
  memory** for validation; the saved configuration remains 5000. In that small
  run VN passed the final feasibility check, while Dykstra and AARM did not;
  bisection correctly skipped those infeasible endpoints.
- Notebook schema and Python cell syntax validation passed. Before/after
  observations are in [audit evidence](methodology-audit-evidence.json) and
  [current evidence](methodology-fixes-evidence.json); notebook execution results
  are in [methodology-fixes-validation.json](methodology-fixes-validation.json).

Reproduce the checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 scripts/validate_result_tables.py --smoke-only
.\.venv\Scripts\python.exe -X utf8 scripts/validate_legacy_notebooks.py
.\.venv\Scripts\python.exe -X utf8 scripts/audit_methodology.py
```

`--smoke-only` overrides the enabled sweep flag in memory before any notebook cell
executes; it does not edit the saved notebook. Without this option, the smoke script
refuses a saved `RUN_PUBLISHED = True`. The validation summary is written to
`results/result_table_generator_smoke/validation-v1.1.json`; the old summary is retained.
Windows sandbox restrictions on private export directories required the full test
suite and table export smoke to run with automatic escalation approval.

## Remaining methodological distinctions

These fixes do not make this an exact numerical reproduction. Gurobi replaces
CPLEX/HiGHS; fixed-count projection support follows the authors' code rather than
the paper's Bernoulli description; data are generated in memory at full precision;
and process-CPU timing differs from the authors' elapsed-clock code and file-reading
scope. The original reference data generator and the benchmark data generator are
still explicitly different. Toy problems remain illustrative experiments with
fresh instances per replication, not structured QR/Diet benchmarks.

The finite LSMR/Dykstra heuristic and TDP2's generally inconsistent retrieval space
can still return infeasible points. Their errors and counts are retained as
measurements. No tolerances or iteration caps were loosened to make the checks pass,
and no replacement optimizer or unreported repair was introduced.

# Thesis methodology audit — 2026-11-02

**Historical audit of revision 1.0.** The user subsequently authorized the fixes;
see [methodology-fixes.md](methodology-fixes.md) for revision 1.1, validation and rerun
instructions. The findings/cell numbers below describe the preserved pre-fix files
in `references/pre_methodology_fixes_v1/`, not the current corrected notebooks.

**The repository is not an exact reproduction of Liberti et al.** The new result
table generator implements the main QR and Diet formulations and much of the
published experimental protocol, with explicit adaptations. The older notebooks
contain additional methodological differences and retrieval bugs. Their results
should not be presented as interchangeable implementations of the paper.

This review preserved the experiment implementation and all four notebooks while
the user's experiment was running. It added a diagnostic script and this report,
and fixed a separate smoke-validation guard. It did not rerun the full experiments
or modify the user-designated reference notebook.

## Sources and scope

The scientific authority is the local [paper](<../Liberti et al.pdf>), especially
Sections 3.2–3.8, Algorithms 1–2, Equation (19), and Tables 3–6. The public
implementation was inspected separately at commit
`a847aa23f8ce597e181c1a1cea75c1cbfcb325c9`; all 12 preserved upstream files still
match the Git blob hashes in [the provenance manifest](../references/rp4lp/provenance.json).
That snapshot is research code with parameter overrides, not proof of the exact
code/settings used to produce the publication.

Reviewed repository material:

| Files | Review coverage |
|---|---|
| `problems/result_table_generator.ipynb` | Example configurations, published sweeps, imports, output locations, execution guard |
| `src/result_table_generator.py`, all `src/result_tables/*.py` | Formulations, data, random projections, retrieval, solvers, timing, seeds, aggregation, exports, provenance |
| `problems/quantile_regression/Liberti_QuantileRegression_m5000_RP_Retrieval.ipynb` | Reference setup, all three retrieval variants, metrics, saved outputs and conclusions |
| `src/quantile_regression.py` | Legacy helper implementation and differences from the reference notebook |
| Both `problems/toy_example/*.ipynb` | LP generation/bounds, projection, exact/approximate retrieval, timing, summaries and plots |
| Both legacy `results/*toy_example/` directories | Six CSVs: 50 trials, 1,500 trace rows and 10 summary rows per directory; plotting code for the 16 PNGs |
| `results/result_table_generator_smoke/` | Existing validation record; 27 exported artifact checksums, all matching; raw success/feasibility counts |
| Existing three `tests/test_result_table_*.py`, smoke script | Coverage and missing methodological checks |
| README, AGENTS, existing research/usage/methodology/validation docs, both requirements files | Research claims, baseline policy and reproducibility instructions |
| `references/rp4lp/` | Preserved generators, QR/Diet model files, relevant driver/retrieval/reporting code, license and manifest |

No separate thesis manuscript was found among the current project files. Virtual
environment dependencies, caches, Git internals and diagnostic logs are not thesis
methods. Two comparison notebooks already marked deleted in Git are outside this
audit's current-file scope; they were not restored. The sibling `thesis_results/`
had no completed `metadata.json` at inspection time. This does not determine the
state of an unsaved or running IDE notebook. No final full-size results are certified
by this audit.

Notebook cell numbers below are **one-based positions including markdown cells**,
not Jupyter execution counts. Search by function name if the notebook layout changes.

## Findings that affect interpretation

### 1. Separate QR sweeps do not consistently reuse the same instances

**Affected:** the current table generator, [runner.py](../src/result_tables/runner.py),
`next_seed()` and the instance/epsilon loops.

Instance seeds and projection seeds consume one shared sequence. The number of
epsilon values and replications therefore changes the seed used for the next row
size. The actual runner, exercised with generation/solves/exports replaced by
small diagnostic stand-ins, gives:

| Row size | Table 5 vs. Table 6, both master seed 2026 |
|---:|---|
| 5000 | Same instance seed: `3339473683144027492` |
| 6000 | Different: `2970585596583893782` vs. `5671711579944790672` |
| 7000, 8000, 9000 | Different at every size |

Within **each** call, one instance per size is reused correctly across all epsilon
values and replications. Diet TDP1/TDP2 also share the same instance and projection
within each paired trial. Those properties are unaffected.

**Consequence:** the two current QR sweeps are reproducible experiments, but their
differences at sizes 6000–9000 cannot be attributed solely to epsilon. The paper's
description of further QR tests and the authors' reuse of instance files motivate
a same-instance comparison; this is an interpretation of the protocol, not a claim
that the paper specifies a particular seed algorithm.

**Next implementation change:** derive instance seeds independently from
`(master_seed, problem, row_size)`, or persist and share the actual instances.
Projection seeds should also be independent of loop order where paired comparisons
are intended. Version that policy and use new output names so current results stay
identifiable. The running implementation was not changed.

### 2. The reference notebook's von Neumann update solves the wrong right-hand side

**Affected:** reference notebook cell 13, `retrieve_apm`; corresponding public-code
branch in `rp4lp.py` around lines 1924–1932.

After clipping, the notebook sets `x = lsmr(A, b + A @ x)[0]`. The result approximately
satisfies `A x_new = b + A x_old`, not `A x_new = b`. Equation (19) requires subsequent
construction of `x_prime = x_hat - x_old` and the line projection; a direct affine
correction is another algorithm, which must be identified separately.

A small feasible LP witness with
`A = [[1,0,0],[0,1,1]]`, `b = [1,1]`, bounds `x >= 0` and projected input `[1,0,3]`
returns approximately `[2,1.5,1.5]`, then reports **"no bound error"** with equality
L2 error **2.23607**. The input satisfies the projected equation for `T = [1,0]`.
Stopping on bounds alone does not establish original feasibility.

This defect is inherited from the inspected public-code branch. Copying that branch
does not establish conformity with the mathematical APM in the paper. The new table
generator uses Dykstra and does not call this routine.

### 3. The reference notebook's Dykstra branch can discard a corrected point

**Affected:** reference notebook cell 19, `retrieve_dykstra`; public-code Dykstra
branch around lines 1798–1837.

The notebook forms `y = clip(x + p)`. If `y` already satisfies the equality tolerance,
it skips the block that assigns the new `x` and updates `q`, leaving the old iterate
in place. Its progress test can then terminate on that old point.

A local branch witness uses `A = [[0,1]]`, `b = [1]`, input `[-0.1,1]` and nonnegative
bounds. The reference returns `[-0.1,1]` with "no sufficient progress", although
`[0,1]` satisfies both sets. This isolates an update defect; it is not a claim that
this particular input was produced by a bounded projected LP. The new generator
returns `[0,1]` and passes its feasibility checks on the same witness.

The new generator already corrected this branch and recomputes errors at the current
iterate. That repair is a documented deviation from the public code and reference.

### 4. The reference AARM routine is not the paper's composed-reflection operator

**Affected:** reference notebook cell 23, `retrieve_aarm`; public code around
lines 1868–1884.

Section 3.4 defines
`(1-alpha) I + alpha (2 beta P_S - I) o (2 beta P_U - I)`, with bounds set `S` and
affine set `U`. The notebook computes `x_A_reflected`, then never uses it. It projects
the bound-projected point and subtracts the original iterate in both reflections.
That is a different update.

Using the same affine approximation to isolate the composition, the diagnostic's
one-step notebook result is `[0.82,-0.01,0.83]`; the paper's composition gives
`[0.676,0.62,1.352]`. This comparison establishes different operators, not a claim
that either single iterate must be feasible. The current table generator does not
use AARM. Preserve the historical routine, but distinguish it from a corrected
paper-operator implementation in any algorithm comparison.

### 5. Both toy notebooks ignore an LP bound during retrieval and evaluation

**Affected:** `rp_lp_toy_example.ipynb`, cells 13/25 (`compute_metrics`/`apm_retrieval`);
`gurobi_toy_example.ipynb`, cells 19/31; their saved result directories.

The original and projected LPs impose `0 <= x <= 5`. Retrieval projects only onto
`x >= 0`; neither its stopping condition nor its metrics check `x <= 5`.

For `A = [[1,0,0],[0,1,1]]`, `b = [1,9]` and projected-feasible input `[1,5,0]`, both
notebooks return `[1,7,2]` and stop. Recorded equality and negativity errors are
both zero, while the upper-bound violation is **2**. The original is feasible,
for example at `[1,4.5,4.5]`.

The saved CSVs do not retain full solution vectors or upper-bound errors, so they
cannot establish whether each historical retrieval obeyed the complete LP bounds.
Any corrected experiment must enforce the full interval, record both bound errors,
and rerun affected results before making feasibility claims.

### 6. Saved reference-notebook conclusions need a precise feasibility qualifier

**Affected:** reference notebook saved outputs in cells 26–28 and commentary in cell 29.

The saved Dykstra values are approximately:

- Equality L2 error: `0.00210954`.
- Sum of bound violations: `0.55774542`.
- Combined error: `0.55985496`, exceeding the notebook's `0.02` criterion.
- Average bound error: `0.00011155`; average equality error: `0.00002379`.

Thus small average errors and failure of the notebook's combined feasibility test
coexist. Its comparison output correctly says `feasible=False`. Describing the point
as approximately feasible requires naming the average-error convention; it is not
a certificate under the notebook's stated stopping/feasibility tolerance.

The saved log also identifies an Apple M4/macOS Gurobi run. Those timings are not a
Windows rerun or the paper's Xeon/CPLEX/HiGHS experiment. Saved execution counts do
not establish that every output corresponds to a fresh execution of the saved source.

### 7. The old QR helper and toy experiments are separate methods

**Affected:** [src/quantile_regression.py](../src/quantile_regression.py) and both toy notebooks.

The helper's opening claim that retrieval algorithms are "preserved" is too broad:
its Dykstra routine uses affine-first ordering, a different initialization and
feasibility test; its AARM uses composed reflections; and its von Neumann routine
uses the line approximation. These differ from the designated reference notebook.
Its solver helper also leaves Gurobi's algorithm automatic instead of selecting
dual simplex for projected QR. It has no final bisection. Treat it as a separately
defined implementation, not a drop-in copy of the reference or new table package.

The toy notebooks already describe themselves as illustrative. Additional reasons
their tables do not reproduce the published QR/Diet tables are:

| Choice | Toy implementation | Published benchmark / new table generator |
|---|---|---|
| Problem | Dense random feasible LP, `n=2m`, sizes 20–60 | Structured QR and Diet, published sizes |
| Projection | Three-point Achlioptas-style; `k=max(4,round(.4m))` | Sparse Gaussian; epsilon-based dimension |
| Scaling | Nonzeros `+/-1/sqrt(k)` with total nonzero probability 1/3 | Gaussian nonzero variance `1/(k sigma)` |
| Approximate affine step | LSMR on `A A.T`, tolerances `1e-8`, cap 500 | LSMR on `A` and Equation (19) |
| Retrieval | Ordinary alternating projection; delta `1e-6` | Selected Dykstra; delta `0.01` |
| Repetitions | Regenerate the original instance per replication | Fixed instance per size, repeated projections |
| Metrics | Absolute relative objective gap, equality L2, negativity sum | Objective ratios and problem-specific average errors |
| Timing | Solve-only projected time; excludes projection construction | Projected construction, solve and retrieval |

The toy scaling gives entry variance `1/(3k)`, not the usual `1/k`. Multiplying all
of `T` by a nonzero scalar cancels from `TAx=Tb`, so this alone does not change the
exact projected LP feasible set; it does change an isometry claim and potentially
numerical scaling. Solving through `A A.T` also changes conditioning.

The Gurobi toy timer excludes model construction as well. Its speed ratios therefore
cannot be relabeled as the paper's full-pipeline ratios. Toy output paths depend on
the notebook working directory, and their CSVs lack the new pipeline's source,
solver-status and seed metadata. No existing toy CSV or plot was rewritten.

## What the new QR/Diet table generator gets right

The following were checked against the paper and/or the specifically identified
public-code choices. This is structural verification, not a guarantee of matching
published objective values or run times.

| Component | Verified implementation |
|---|---|
| QR LP | `D beta + u_plus - u_minus = b`, beta free, nonnegative residual variables, tau `0.2`, 399 predictors |
| QR data | 400 total columns, Uniform(-1,1) values, Bernoulli density `0.8` across predictors and response, as the authors' generator |
| Original Diet | Minimize food cost with `Dq >= b`, `q >= 0`; 400 foods, Uniform(0,1), matrix density `0.5` |
| TDP1 | `[TD,-T]`, nonnegative food/slack variables, penalty `mu=800` |
| TDP2 | `[TD,I,-I]`, nonnegative food/positive-negative residual variables, penalty `mu=800` |
| Projection dimension | Public-code rounding: `int(round((1/epsilon)**2) * log(n))`; QR `n=399+2m`; both Diet variants use `n=400+m`; no clipping at m |
| Projection | `sigma=.5*density(A)`, Gaussian scaling `1/sqrt(k*sigma)`; fixed-count support is explicitly identified as a code-level choice |
| Retrieval | Bounds-first Dykstra with correction vectors, LSMR line construction, max 30 iterations, delta `0.01`, documented repairs and minimization bisection |
| QR metrics | Negative residual-block sums divided by m; mean absolute equality error; free beta excluded from bound penalties |
| Diet metrics | Food cost excludes penalty; mean negative food quantity and mean original nutrient shortfall |
| Sweeps | Published row/epsilon ranges, five replications; one original solve per size per call; paired Diet variants |
| Aggregation | Mean of replication ratios; explicit failures, undefined ratios and separate feasibility counts; unrounded raw records |
| Timing/export | Process CPU and wall clocks separated; projected construction/solve/retrieval components; seeds, policies, statuses, source/artifact hashes retained |

The implementation remains a **Gurobi adaptation**, for these material reasons:

1. Section 3.7 uses original CPLEX and projected HiGHS for QR/Diet. This code uses
   Gurobi, explicit methods and one Gurobi thread. That does not force NumPy/BLAS
   to one thread, and the experiment loop itself is serial. These results do not
   measure a parallel-projection extension.
2. The paper describes independent Bernoulli projection support; the preserved
   code and reference use a fixed nonzero count. The generator follows the latter.
3. Seeded local generators and full-precision in-memory data differ from the
   authors' global generators and six-decimal AMPL files. The reference notebook
   also has a different QR data generator: fixed-count predictor sparsity and a
   dense response. Its seed-42 instance is not the new generator's benchmark data.
4. Corrected retrieval stopping, free QR bounds, the degenerate-line guard, sigma
   calculation and bisection can change iterates/results relative to public code.
   The preserved reference notebook has no final bisection step.
5. The paper includes instance reading in timing and calls its times CPU times;
   the inspected public code uses elapsed `time.time()`. This implementation uses
   process CPU, with shared generation recorded separately and no instance-file
   reading. The first lazy Gurobi import can also enter an original-solve timing.
6. TDP2 retrieval follows the authors' `Dq=b` retrieval space, which is generally
   inconsistent for these overdetermined random instances. Evaluation correctly
   uses the original `Dq>=b`. Retrieval failure is a possible measured outcome.

The inspected public QR path also has `highs-ipm`, epsilon `[0.5]` and one-run
overrides, unlike the paper. Its unused AMPL projected QR model has an additional
sign issue, and its QR retrieval call omits free lower bounds. These are reasons
to keep paper equations, active/inactive public code, and this adaptation distinct.
For the upstream Diet cache issue, distinguish the function overwriting passed
values from the driver's final ratios: the driver retains its first `fstar[df]`
for those denominators. The function-level defect does not by itself establish
incorrect published table ratios.

**A completed trial is not necessarily feasible.** In the existing small smoke
artifacts all 4 QR trials and 8 Diet trials completed, but only 4/4 QR and 2/8 Diet
retrievals passed `original_feasible`. This audit verified those counts and artifact
checksums. Infeasible finite retrievals intentionally remain in feasibility-error
averages. Check counts and full-precision errors alongside the formatted table;
`0.0000` is rounded, not proof of exact feasibility.

## Safe fix made during this review

[scripts/validate_result_tables.py](../scripts/validate_result_tables.py) previously
executed every notebook cell before asserting `RUN_PUBLISHED is False`. Saving the
notebook with the flag true could therefore launch the entire published sweep
before the "smoke" check rejected it.

It now checks the saved assignment before executing any cell and refuses an enabled,
missing, duplicate or computed flag. Three focused tests in
[test_smoke_guard.py](../tests/test_smoke_guard.py) passed, including a check that
no cell executes when the flag is true. The saved example notebook also passed the
preflight without executing its cells. This changes only the validation entry point,
not any ongoing notebook run or numerical implementation.

## Evidence and follow-through

[scripts/audit_methodology.py](../scripts/audit_methodology.py) loads only selected
function definitions from notebooks, runs the tiny counterexamples, exercises the
actual runner with data/solve/export work replaced, and verifies reference hashes.
It does not run notebook top-level cells, call an LP solver, or write result files.
Its recorded output is [methodology-audit-evidence.json](methodology-audit-evidence.json).

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/audit_methodology.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_smoke_guard.py -v
```

The existing 28-test validation is documented in [result-table-validation.md](result-table-validation.md).
Those tests cover useful numerical and export properties but do not certify paper
fidelity; they lacked a cross-sweep instance-pairing check and do not test the legacy
notebooks. The full suite and solver smoke experiments were not rerun during this
audit. Only the new three guard tests and the solver-free diagnostic were run.

Before using results for thesis comparisons:

1. Preserve the current run with its metadata and this audit. Compare epsilons
   within each sweep; qualify cross-sweep QR comparisons at sizes 6000–9000.
2. Introduce a separately versioned instance-seed policy before collecting a
   same-instance comparison across Tables 5 and 6. Keep current outputs as history.
3. Preserve the designated reference notebook. Put corrected legacy algorithms
   in an explicitly identified comparison implementation, with counterexample
   checks, before making claims about the paper's three mathematical operators.
4. Fix full-bound retrieval/evaluation in a new toy result version and regenerate
   any toy results used to claim LP feasibility. Keep demonstrations clearly labeled.
5. Use the new tables' failure/feasibility counts and explain solver, data, retrieval
   and timing differences in the thesis. Collect final performance measurements
   under a stated workload/thread policy; an interactive run during other work is
   not a controlled hardware comparison with the published timings.

These are outstanding methodological changes or reporting requirements. They were
not silently applied to the active experiment, and the repository is not marked as
an exact reproduction by this review.

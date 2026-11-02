# Methodology and source audit

See the [repository-wide audit](methodology-audit.md) for historical findings and
[implemented corrections](methodology-fixes.md) for revision 1.1. Keyed seeds now
preserve instance identity across different grids with the same master seed/problem/size.

Primary scientific source: local `Liberti et al.pdf`, Sections 3.2–3.8 and 4.1–4.3;
Tables 2–8 use the reporting structure implemented here. Only the QR and Diet families
are implemented. The public source was inspected at commit
[`a847aa23f8ce597e181c1a1cea75c1cbfcb325c9`](https://github.com/leoliberti/rp4lp/tree/a847aa23f8ce597e181c1a1cea75c1cbfcb325c9).
Selected original files are preserved byte-for-byte in `references/rp4lp/` with the
upstream CC0 license, URLs, and Git blob hashes in `provenance.json`. These research
scripts are reference material and are never imported or executed by the new pipeline.

## Component mapping

| Reference | New component | Responsibility |
|---|---|---|
| `quantreggen.py`, `quantreg.mod`, `QuantReg` in `rp4lp.py` | `src/result_tables/quantile_regression.py` | One masked data instance, original QR, one projected QR replication |
| `dietgen.py`, `diet.mod`, `dietprj1.mod`, `dietprj2.mod`, `Diet` | `src/result_tables/diet_problem.py` | One Diet instance and either TDP variant |
| `sparse_gaussian` and dimension expressions | `projection.py` | Seeded sparse projection and exact rounding rule |
| `altProjRetrieval`, `bisectionSearch` | `retrieval.py` | Dykstra, LSMR equation (19), bisection, diagnostics |
| Original AMPL/CPLEX and projected SciPy/HiGHS calls | `gurobi_solver.py` | Direct Gurobi interface, status checking, explicit parameters |
| Problem-specific output metrics | `metrics.py`, `pipeline.py` | Ratios, feasibility, one replication's timing/results |
| Main `rp4lp.py` experiment loops | `runner.py` | Instances, cached originals, all repetitions, seeds |
| `maketables.py` | `reporting.py` | Means, grouped seven-column tables, three plot categories, exports |

`src/result_table_generator.py` supplies the public entry point. The original toy/QR
notebooks and helper are archived under `references/pre_methodology_fixes_v1/`.
Active notebooks use the corrected methods described in the revision notes. The
reference notebook retains its distinct data generator; historical outputs remain
in the archive and original result directories.

## Mathematical choices and verified discrepancies

**Quantile Regression.** The generator draws an m by 400 table from Uniform(-1,1),
then independently masks entries at density 0.8, including the response column. Its
first 399 columns are predictors and the last is b. This follows `quantreggen.py`
and Sections 3.6.3/4.3. The reference notebook instead selects an exact number of
predictor nonzeros and draws a dense response separately. The new generator uses
full-precision data, whereas upstream writes six decimal places to AMPL files.

The original and projected objectives are `[0, tau, 1-tau]` with tau=0.2; the matrices
are `[D,I,-I]` and `[TD,T,-T]`. Coefficients beta are free during **both optimization
and retrieval**. Upstream's active SciPy projected solve has correct free bounds,
but its call to `altProjRetrieval` at line 1241 omits `low=lb`, thus the default
`low=0` incorrectly applies to beta. The inactive AMPL `quantregprj.mod` subtracts
the already-negative final TA block; the new Gurobi model uses TA directly.

**Diet.** There are 400 foods. Nonzero D, costs and requirements are Uniform(0,1),
with Bernoulli density 0.5 on D. The original LP uses inequalities Dq>=b and q>=0.
As upstream does, no rejection/resampling alters an infeasible randomly drawn row;
a zero nutrient row with positive requirement is a failed original solve.

TDP1 uses `[TD,-T]` and costs `[c,mu*ones(m)]`. TDP2 uses `[TD,I_k,-I_k]` and
`[c,mu*ones(2k)]`, with mu=2*n=800. Although a top-level upstream constant is 10,
the active SciPy path at lines 995–999 uses 2*n, agreeing with Section 4.2.
The TDP1 `.mod` comment has a plus-sign typo; the constructed matrix and actual
constraints use the negative slack block. The new pipeline stores both the solver's
penalty-augmented objective and the food cost; table ratios use **food cost only**.

TDP1 retrieval retains `[q,r]`, the affine system `[D,-I]x=b`, nonnegative bounds,
and augmented c for the retrieval/bisection objective. TDP2 discards projected
residual variables and retrieves q against **Dq=b**, which is usually inconsistent
when m>400. This is the upstream choice at lines 1026–1041, not a new inequality
retrieval heuristic. After retrieval, both variants are evaluated on c*q, q>=0,
and Dq>=b. Hence the Diet `avgeq` column is mean nutrient shortfall, not mean
absolute standard-form equality residual. Retrieval diagnostics record the latter
separately. Dykstra stopping cannot certify feasibility for the inconsistent TDP2
system; the returned errors and flags must be inspected.

The upstream `Diet` function also replaces provided cached `fstar`/`xstar` with
`myInf`/ones when `runOrg=False` (lines 859–861). The new runner preserves one real
original solve per instance for all epsilon/replication/variant combinations.

**Projection dimension.** Both families use
`int(round((1 / epsilon)**2) * log(number_of_LP_variables))`:

- QR uses `399 + 2*m` (`cols=p-1+2*m` at upstream lines 1170/1176, where p=400).
- Both Diet variants use `400 + m` (upstream line 939), including TDP2 even though
  its projected model has `400 + 2*k` decision variables.

Rounding happens **before** multiplication by log, then int truncates the product.
For QR m=5000, k is 231 at epsilon=.2, 924 at .1, and 406 at .15. For Diet m=3000,
k=203 at epsilon=.2. C0=1. Dimensions exceeding m are executed and reported;
the corrected reference notebook and legacy helper also allow k>=m.

**Projection density and normalization.** Sigma is half the **actual** density of
`[D,I,-I]` or `[D,-I]`, even for TDP2. Upstream QR computes density using p=400 in
places where p-1=399 is required (line 1115). The new pipeline corrects this using
the sparse matrix's nonzero count. Conditional nonzeros have variance 1/(k*sigma).

Upstream `scipy.sparse.random`, like the notebook, selects a fixed nonzero count
`round(sigma*k*m)`; the paper describes independent Bernoulli entry selection.
The new sampler preserves the **fixed-count support** and Gaussian normalization,
using a local NumPy generator for both support and values. Support positions are
uniformly sampled without replacement; the stored matrix is CSR. Global NumPy and
Python random states are never used. Corresponding Diet variants reconstruct the
identical T from their shared seed; no timing credit from the other variant is used.

**Retrieval.** The initial point is `x_projected + lsmr(A,b-A@x_projected)[0]`.
The implemented Dykstra iteration first projects onto bounds, then applies the
LSMR line construction in equation (19). It retains both Dykstra correction vectors,
precomputes `lsmr(A,b)` once, limits iterations to 30, and uses tolerance 0.01.
This is an approximate affine projection, so exact Dykstra convergence theory is
not a guarantee for this finite heuristic. LSMR tolerances and every call's status,
iterations, residual and condition estimate are recorded.

Corrections to upstream retrieval are deliberate: current-iterate errors are
recomputed after each update; a no-bound-error exit also checks equality error;
zero-length lines are guarded; an affine-feasible bound projection no longer
leaves the previous iterate in place. Both projections execute with their correction
vectors when the current point still fails the individual feasibility checks.
The combined-error and small-progress stopping tests remain, but are not labeled
feasible unless final residual checks pass. The corrected notebook shares this
Dykstra implementation, with the pre-fix notebook preserved separately.

The paper describes bisection toward the projected point to improve objective
while retaining affine feasibility. Upstream `bisectionSearch` swaps objective
values twice without swapping endpoint vectors, and its objective-difference loop
uses the wrong orientation for typical minimization endpoints. We implement the
paper's stated minimization intent: start from an approximately feasible endpoint,
move only toward a lower objective, retain equality L2 and summed bound error <=.01,
and stop when the objective interval is <=.01 or after 60 bisections. A start that
does not meet those feasibility conditions is reported as skipped; no alternative
repair method is substituted. This can produce different results from the checked-in
research code. The corrected reference notebook now includes this bisection step;
the archived version did not.

## Solvers, repetitions, timing and reporting

Section 3.7 describes original CPLEX automatic, projected QR HiGHS revised simplex,
and Diet barrier without crossover. The checked-in active QR path is instead
`highs-ipm` with `ipm_optimality_tolerance=1e-5`. Its epsilon list and five runs are
overridden to `[0.5]` and one run. Those overrides do not represent Tables 3–6.

This implementation uses Gurobi Method=-1 for originals, Method=1 (dual simplex)
for projected QR, and Method=2/Crossover=0 for Diet, following the paper's method
choices as closely as practical. Threads=1 is a deliberate repeatability default;
it is not claimed to match the paper's thread use. Tolerances, seed, presolve,
numerical focus, crossover, time limit, threads, method, version, and actual model
parameters are retained. Gurobi status must be optimal with an available finite
solution before X or ObjVal is used. There is no fallback optimizer or silent retry
with different solver settings.

The default repetition count is five. One instance per size is reused throughout
the sweep; all repetitions are owned by the table generator. Revision 1.1 derives
uint64 seeds from SHA-256 keys containing the master seed, problem and row size,
plus epsilon and replication for projections. Thus grid length, other row sizes and
run counts do not change a shared instance or projection. Every seed is stored.

The paper calls its times CPU times, but upstream uses `time.time()`. This new
pipeline measures process CPU with `time.process_time_ns()` and separately measures
wall time. Gurobi is called in-process (no AMPL/solver subprocess), so its native
thread work is within the process timer. The timer is tested against process user+
system CPU and native numerical work in the smoke validation. Solver Runtime is
wall-clock information and is never the primary publication time. Matrix/formulation
construction and retrieval are included; formatting, plots, metadata and digests are
excluded. Random instance generation is outside both pipelines and recorded
separately. Since data are already in memory, no file read/serialization overhead is
included; this differs from Section 3.8's instance-reading scope.

For QR, `avgin` sums negative parts from both slack blocks and divides by m,
not 2m or the full variable count; `avgeq` is mean absolute equality residual.
Upstream's `maxinq` adds the two block maxima; this convention is retained as
`maxin` and the true maximum is separately stored as `max_bound_violation`.
Diet's definitions are mean negative food quantity and mean nutrient shortfall.
Means are taken over finite completed trials. Failure counts are explicit. An
approximately retrieved point with positive error remains in the reported sample,
as needed to reproduce the paper's feasibility-error measurements.

The reporting layout follows `maketables.py`: group by original size, sort epsilon,
use seven columns, print epsilon and CPU ratio to two decimals and other measures to
four. Raw and averaged CSVs retain full precision. The three PNG categories use
unrounded metrics. Failure counts are visible in the previews, LaTeX annotations,
plot titles, averaged dataset and separate count files.

API references checked during implementation:
[Gurobi parameters](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html),
[Gurobi statuses](https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html),
[Python process CPU clock](https://docs.python.org/3/library/time.html#time.process_time),
[SciPy sparse random](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html).

# Scientific source and experiment baseline

**Current revision (2026-11-02):** the user authorized implementation of the audit
fixes. See [methodology-fixes.md](methodology-fixes.md). The active reference notebook
retains its data setup but uses corrected shared VN/Dykstra/AARM routines, checks both
feasibility tolerances, and adds minimization bisection. Version 1.1 table experiments
use grid-independent keyed seeds. Pre-change source/notebooks and outputs are preserved;
the settings and observations below describe that historical baseline.

For the subsequent repository-wide review, confirmed counterexamples and outstanding
methodological issues, see [methodology-audit.md](methodology-audit.md) (2026-11-02).
That review inspected the preserved authors' code as well as the local paper; the
historical notes below retain their original baseline context.

Recorded on 2026-10-12 from the local PDF and the user-designated reference notebook. These notes provide persistent context; consult the PDF for complete arguments and assumptions.

## Source

Leo Liberti, Benedetto Manca, and Pierre-Louis Poirion (2023), *Random Projections for Linear Programming: An Improved Retrieval Phase*. ACM Journal of Experimental Algorithmics, volume 28, issue 1, article 2.2, 33 pages. DOI: https://doi.org/10.1145/3617506.

- Local paper: `Liberti et al.pdf` at the repository root.
- Authors' code location reported in Section 3.7: https://github.com/leoliberti/rp4lp. That external code was not inspected when these notes were created; notebook comments attributing behavior to it are not independent verification.
- Reference notebook: `problems/quantile_regression/Liberti_QuantileRegression_m5000_RP_Retrieval.ipynb`. The user explicitly designated this as the desired setup.
- Page references below are one-based PDF pages, matching the article's page suffixes.

## Research approach

The paper projects an LP's equality constraints using a random matrix T, solves the resulting smaller LP, then retrieves an approximately feasible solution for the original LP. The original objective and variable bounds are retained. For minimization, the projected feasible set contains the original feasible set, so its optimum is a lower bound when both problems have finite optima (Section 2, page 5).

Projected feasibility does not imply original feasibility. Evaluate retrieval against the original equations and bounds. A small average error, a stopping status, and satisfaction of a particular feasibility tolerance are distinct claims.

Theoretical error bounds have assumptions and constants that do not directly prescribe practical parameters. Sections 3.1–3.2 discuss this limitation; epsilon is not a blanket guarantee on the retrieved objective error.

## Quantile regression formulation

Section 3.6.3, page 15, defines:

```text
minimize tau * sum(u_plus) + (1 - tau) * sum(u_minus)
subject to D beta + u_plus - u_minus = b
           beta free; u_plus, u_minus >= 0

x = [beta, u_plus, u_minus]
A = [D | I | -I]
c = [0_p, tau * 1_m, (1 - tau) * 1_m]
n = p + 2m
```

The paper samples nonzero data values uniformly from [-1, 1], uses design density 0.8, and fixes tau = 0.2. Free beta components must not receive nonnegativity bounds. The constraint matrix density is `(0.8 * p + 2) / (p + 2m)` for the specified design density.

## Projection and retrieval in the paper

- Section 3.3, page 12: T has shape k by m; each entry is nonzero with probability sigma, with conditional Gaussian standard deviation `1 / sqrt(k * sigma)`. The study uses `sigma = 0.5 * density(A)`.
- Sections 1.1.2 and 3.2: projected dimension follows the empirical scale `epsilon^-2 * ln(n)` with the multiplicative constant chosen as 1. The text also uses m in a parameter discussion; record the actual dimension and rounding expression used by code rather than silently changing it.
- Sections 3.2–3.4, pages 12–13: retrieval compares von Neumann APM, Dykstra APM, and AARM; uses at most 30 iterations, delta = 0.01, a small-progress stopping test, and AARM alpha = beta = 0.9. Dykstra gave the best preliminary tradeoff and is the selected benchmark variant.
- Section 3.5, page 13: affine projection is approximated using LSMR. Equation (19) constructs a closest point on a line through two approximately affine-feasible points. This is a heuristic, not automatically the exact orthogonal projection onto the entire affine space.
- Section 3.5 also describes a final bisection step to improve the objective along a segment toward the projected solution while retaining approximate equality feasibility.

## Pre-fix notebook choices (historical)

These are observed implementation settings, not certification of correctness or exact reproduction:

| Setting | Reference notebook |
| --- | --- |
| Instance | m = 5000, p = 399, density(D) = 0.8, tau = 0.2 |
| Randomness | Local NumPy generators; instance seed 42 and projection seed 42 |
| Dimension | n = 10399; epsilon = 0.2; `k = int(round((1.0 / epsilon) ** 2) * np.log(float(n_variables)))`, giving k = 231 |
| Projection sparsity | Exactly `round(sigma * k * m)` selected positions without replacement; Gaussian values scaled by `1 / sqrt(k * sigma)` |
| Original solve | Gurobi automatic method, Method = -1 |
| Projected solve | Gurobi dual simplex, Method = 1 |
| Retrieval | Separate von Neumann, Dykstra, and AARM runs, max_iter = 30, delta = 0.01; AARM alpha = beta = 0.9 |
| Retrieval start | `x_projected + lsmr(A, b - A @ x_projected)[0]` |
| Feasibility check | Sum of bound violations plus L2 equality residual <= 0.02 |
| Average inequality error | Total negative-part violations of both residual blocks divided by m |
| Average equality error | Mean absolute component of `A @ x - b` |
| Timing | `time.perf_counter()` wall time; shared D, b, A generation excluded; full projected pipeline includes T generation, TA/Tb, solve, and retrieval |

For Dykstra and AARM, `solve_projection_system` implements the line construction associated with Equation (19). The von Neumann update and AARM reflection sequence have comments claiming correspondence to public code. Verify that source before treating those sequences as the mathematical algorithms in the paper. In particular, the notebook computes a first AARM reflected point but does not use it in the next projection, whereas Section 3.4 states a composition of two relaxed reflections.

## Benchmark comparison and differences to retain explicitly

- Section 3.7, page 16: the paper solves originals with CPLEX 22.1 defaults and projected quantile regression with HiGHS revised simplex. The notebook uses Gurobi for both.
- Section 3.7: the paper repeats each size/epsilon pair five times to account for projection randomness. The reference notebook is a single seeded instance/run.
- Table 5, page 21: quantile regression row sizes are 5000, 6000, 7000, 8000, and 9000; epsilon ranges from 0.2 through 0.9 in increments of 0.1. For m = 5000 and epsilon = 0.2, the published ratios are projected/original objective 0.0000, retrieved/original objective 1.0402, k/m = 0.0462, and projected pipeline/original time 0.19. These are published averages, not expected exact outputs for seed 42 or Gurobi.
- Sections 3.8 and 4, pages 17–18: report projected/original and retrieved/original objective ratios, average bound and equality errors, k/m, and projected pipeline/original time. The notebook additionally reports relative objective error, L2 equality residual, maximum bound violation, and the reciprocal time ratio (direct/Dykstra pipeline, where greater than 1 indicates speedup).
- Paper timing includes instance reading and formulation construction and is described as CPU time. Notebook timing is wall time with shared generation excluded. Compare only after stating the timing scope and ratio direction.
- Fixed-count projection sparsity in the notebook differs from independent entrywise nonzero selection described in Section 3.3.
- The reference notebook has no final bisection postprocessing step. Do not describe its retrieval pipeline as including that part of the paper.
- The inequality-error denominator above is observed in the notebook. Confirm the authors' metric implementation before claiming an exact match to Table 5's averaging convention.

## Guidance for future work

Use the reference notebook as the starting point for related experiment scripts and extensions. Preserve free beta bounds, original feasibility checks, seed reproducibility, projection scaling, and explicit timing scope. Keep single-run observations separate from averaged benchmark results. Treat existing saved outputs and markdown summaries as historical observations until rerun. Record solver versions, hardware, algorithm changes, seeds, repetitions, and any parallel execution policy when making new performance claims.

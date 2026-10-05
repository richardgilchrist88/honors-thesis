# Quantile regression timing diagnostics

Two sequential runs of each identical seeded instance, on this Windows machine,
with Gurobi 13.0.3, one thread, default automatic algorithm/presolve settings,
p=399, density=0.8, tau=0.2, k=20% of m, and data/projection/solver seeds of 42.
All times are wall-clock seconds. Shared data generation is excluded. Solver
logging was enabled; its overhead is included. Pending model updates are flushed
in setup so the optimize column measures the optimize() call separately.

The slow m=5000 solve repeats: 52.701 and 54.773 seconds. First model creation
(including default-environment initialization) takes only 0.011 seconds.
The logs show 5000 presolved rows and 10399 columns for m=5000, versus 399 rows
and m columns for the larger cases. The 5000 case takes 12058 iterations.
This is evidence of different solver/presolve behavior, not a timing arithmetic
error. Logs from the earlier 46.101-second run were not retained, so its precise
breakdown cannot be reconstructed.

## Original LP timing breakdown

Median of two runs; model creation is already included in setup.

| m | original_setup_seconds | original_optimize_seconds | original_extraction_cleanup_seconds | time_pipeline |
| --- | --- | --- | --- | --- |
| 5000 | 0.114 | 53.604 | 0.018 | 53.737 |
| 6000 | 0.084 | 6.179 | 0.021 | 6.285 |
| 7000 | 0.120 | 7.360 | 0.026 | 7.507 |
| 8000 | 0.118 | 8.147 | 0.030 | 8.297 |
| 9000 | 0.123 | 8.919 | 0.035 | 9.078 |

## Complete pipeline times

Median of two runs. RP totals include projection generation, TA/Tb multiplication,
projected model setup and solve, solution extraction/cleanup, and retrieval.

| m | Original LP | von Neumann APM | Dykstra APM | AARM |
| --- | --- | --- | --- | --- |
| 5000 | 53.737 | 16.144 | 16.221 | 16.875 |
| 6000 | 6.285 | 21.765 | 21.858 | 22.572 |
| 7000 | 7.507 | 33.754 | 33.760 | 34.639 |
| 8000 | 8.297 | 49.178 | 49.317 | 50.244 |
| 9000 | 9.078 | 70.960 | 70.906 | 71.983 |

AARM is infeasible at every size; Dykstra is infeasible at m=6000 at the specified
0.01 L2 tolerance. Their elapsed times do not represent successful feasible solutions.
Objectives and feasibility flags match the earlier run for all methods and sizes.
Original comparison.csv and comparison_summary.csv remain the earlier results.

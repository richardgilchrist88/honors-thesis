# Honors Thesis

Research code for my honors thesis on parallel randomized projections for large-scale linear programming.

## Project Structure

- `problems/` — individual LP problem setups and notebooks
- `src/` — reusable Python functions shared across problems
- `data/` — input data and benchmark instances
- `results/` — generated experiment outputs

## Result Table Generator

The reusable Gurobi pipeline runs Quantile Regression and Diet (TDP1/TDP2),
records every replication and seed, and exports paper-style tables and plots.
Start with [the usage guide](docs/result-table-generator.md) and
[example notebook](problems/result_table_generator.ipynb). The
[methodology audit](docs/result-table-methodology.md) documents differences between
the paper, the authors' implementation, and this Gurobi adaptation.

The [repository-wide methodology review](docs/methodology-audit.md) also covers
the reference and toy notebooks, confirmed retrieval/bound-checking issues, and
the limits of comparing separate QR epsilon sweeps with the original seed policy.
The [implemented corrections](docs/methodology-fixes.md) fix these issues, preserve
the old sources/results, and document validation and remaining adaptations.

Install `requirements-result-tables.txt` in the project environment, then run
`python -m unittest discover -s tests -v`. The notebook now has full sweeps enabled:
select `.venv`, restart the kernel, then **Run All** to generate both QR and Diet tables.

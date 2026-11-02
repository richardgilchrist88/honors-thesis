"""Reproduce methodology-audit observations without LP solves or notebook runs.

Run from the repository root:
    .venv\\Scripts\\python.exe scripts/audit_methodology.py

Prints JSON to stdout; does not modify experiments, notebooks, or result files.
These observe the current implementation. The pre-fix observations are preserved
in docs/methodology-audit-evidence.json. This is not certification of reproduction.
"""

from pathlib import Path
import ast
import hashlib
import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsmr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.result_tables import runner
from src.result_tables.common import LinearProgram, SolveResult
from src.result_tables.retrieval import retrieve

REFERENCE = Path('problems/quantile_regression/Liberti_QuantileRegression_m5000_RP_Retrieval.ipynb')
TOYS = [Path('problems/toy_example/rp_lp_toy_example.ipynb'),
        Path('problems/toy_example/gurobi_toy_example.ipynb')]


def notebook_functions(relative, names):
    """Load only named function definitions; never execute notebook top-level cells."""
    notebook = json.loads((ROOT / relative).read_text(encoding='utf-8'))
    scope = dict(np=np, pd=pd, time=time, lsmr=lsmr)
    locations = {}
    for index, cell in enumerate(notebook['cells'], 1):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        selected = [node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name in names]
        for node in selected:
            locations[node.name] = {'cell': index, 'line_in_cell': node.lineno}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(relative), 'exec'), scope)
    missing = set(names) - locations.keys()
    if missing:
        raise ValueError(f'Missing notebook functions: {missing}')
    return scope, locations


def reference_observations():
    names = {'retrieval_start', 'bound_error_sum', 'solve_projection_system',
             'retrieve_apm', 'retrieve_dykstra', 'retrieve_aarm'}
    f, locations = notebook_functions(REFERENCE, names)
    # A bounded projected-feasible input for T = [1, 0]. The original is feasible.
    A = sparse.csr_matrix([[1., 0., 0.], [0., 1., 1.]])
    b, x0 = np.array([1., 1.]), np.array([1., 0., 3.])
    lower, upper = np.zeros(3), np.full(3, np.inf)
    vn, iterations, status = f['retrieve_apm'](x0, A, b, lower, upper)
    vn_observation = dict(input=x0.tolist(), output=vn.tolist(), iterations=iterations,
                          status=status, equality_l2=float(np.linalg.norm(A @ vn - b)))

    # A local branch witness: the clipped point already satisfies the equations.
    # This isolates the stale-iterate branch; it is not a full projected-LP trial.
    branch_A, branch_b = sparse.csr_matrix([[0., 1.]]), np.array([1.])
    branch_x = np.array([-.1, 1.])
    old, iterations, status = f['retrieve_dykstra'](
        branch_x, branch_A, branch_b, np.zeros(2), np.full(2, np.inf))
    lp = LinearProgram(branch_A, branch_b, np.zeros(2), np.zeros(2), np.full(2, np.inf))
    new = retrieve(branch_x, lp)
    dykstra_observation = dict(input=branch_x.tolist(), reference_output=old.tolist(),
                               reference_status=status, reference_iterations=iterations,
                               reference_bound_error=float(np.maximum(-old, 0).sum()),
                               generator_output=new.x.tolist(),
                               generator_feasible=bool(new.diagnostics['retrieval_feasible']))

    # Section 3.4: S = bounds, U = affine, (1-alpha)I + alpha R_S R_U.
    # Use the same Eq. (19) approximation for U to isolate composition/order.
    alpha = beta = .9
    initial = f['retrieval_start'](x0, A, b)
    x_check = lsmr(A, b)[0]
    affine = f['solve_projection_system'](initial, A, b, x_check)
    reflected_affine = 2 * beta * affine - initial
    reflected_bounds = 2 * beta * np.clip(reflected_affine, lower, upper) - reflected_affine
    composed = (1 - alpha) * initial + alpha * reflected_bounds
    notebook, _, _ = f['retrieve_aarm'](x0, A, b, lower, upper, max_iter=1)
    aarm_observation = dict(notebook_one_step=notebook.tolist(),
                            composed_reflections_one_step=composed.tolist(),
                            difference_l2=float(np.linalg.norm(notebook - composed)))
    return dict(locations=locations, von_neumann=vn_observation,
                dykstra=dykstra_observation, aarm=aarm_observation)


def toy_observations():
    observations = {}
    # Original feasible point: [1, 4.5, 4.5]; projected feasible point: [1, 5, 0]
    # with T = [1, 0]. Both obey 0 <= x <= 5.
    A, b = np.array([[1., 0., 0.], [0., 1., 1.]]), np.array([1., 9.])
    x0 = np.array([1., 5., 0.])
    for path in TOYS:
        f, locations = notebook_functions(path, {'compute_metrics', 'apm_retrieval',
                                                 'exact_equality_projection_update'})
        x, _, history = f['apm_retrieval'](A, b, x0)
        observations[path.as_posix()] = dict(
            locations=locations, input=x0.tolist(), output=x.tolist(),
            recorded_metrics=f['compute_metrics'](A, b, np.ones(3), x),
            upper_bound=5, measured_max_upper_violation=float(np.maximum(x - 5, 0).max()),
            stop_stage=history.iloc[-1]['stage'])
    return observations


def sweep_instance_seeds():
    """Exercise the actual runner with all data, solve, and export work replaced."""
    def fake_instance(size, seed):
        return SimpleNamespace(m=size, n=399, instance_seed=seed,
                               D=sparse.csr_matrix((size, 399)))

    def fake_trial(instance, epsilon, seed, original, formulation_variant=None):
        return dict(problem_type='Quantile Regression', row_size=instance.m, epsilon=epsilon,
                    instance_seed=instance.instance_seed, projection_seed=seed,
                    formulation_variant=formulation_variant, success=True,
                    projected_status='NOT_SOLVED_AUDIT', projection_digest='NOT_BUILT_AUDIT')

    arguments = dict(problem_type='Quantile Regression', row_min=5000, row_max=9000,
                     row_increment=1000, num_runs=5, master_seed=2026,
                     output_dir=ROOT / 'results' / 'audit-not-written')
    with patch.object(runner.quantile_regression, 'generate_instance', side_effect=fake_instance), \
         patch.object(runner.quantile_regression, 'solve_original', return_value=SolveResult('NOT_SOLVED_AUDIT')), \
         patch.object(runner.quantile_regression, 'solve_one_replication', side_effect=fake_trial), \
         patch.object(runner, 'aggregate_trials', return_value=None), \
         patch.object(runner, 'make_result', side_effect=lambda raw, averaged, meta, path: SimpleNamespace(metadata=meta)), \
         patch.object(runner, 'environment_metadata', return_value={}), \
         patch.object(runner, 'save_outputs'), patch.object(Path, 'mkdir'):
        table5 = runner.result_table_generator(**arguments, epsilon_min=.2, epsilon_max=.9, epsilon_increment=.1)
        table6 = runner.result_table_generator(**arguments, epsilon_min=.1, epsilon_max=.15, epsilon_increment=.05)
    return [dict(row_size=a['row_size'], table5_instance_seed=a['instance_seed'],
                 table6_instance_seed=b['instance_seed'], same_instance_seed=a['instance_seed'] == b['instance_seed'])
            for a, b in zip(table5.metadata['instances'], table6.metadata['instances'])]


def provenance():
    manifest = json.loads((ROOT / 'references/rp4lp/provenance.json').read_text())
    verified = {}
    for name, record in manifest['files'].items():
        data = (ROOT / 'references/rp4lp' / name).read_bytes()
        digest = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        verified[name] = digest == record['git_blob_sha']
    paths = [Path('Liberti et al.pdf'), Path('src/quantile_regression.py'),
             Path('src/retrieval_methods.py'), Path('src/toy_retrieval.py'), REFERENCE, *TOYS,
             Path('problems/result_table_generator.ipynb'), Path('scripts/validate_result_tables.py')]
    paths += [p.relative_to(ROOT) for p in sorted((ROOT / 'src/result_tables').glob('*.py'))]
    return dict(authors_commit=manifest['commit'], authors_file_hashes_match=verified,
                reviewed_input_sha256={p.as_posix(): hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                       for p in paths})


def main():
    evidence = dict(reference_notebook=reference_observations(), toy_notebooks=toy_observations(),
                    cross_sweep_seeds=sweep_instance_seeds(), provenance=provenance())
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

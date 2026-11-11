"""Generate small seeded trace examples; never launch full published sweeps.

Runs the reference QR comparison with 450 rows in memory (saved size stays 5000),
plus one QR and one paired Diet projection through the ordinary table pipeline.
Outputs: results/retrieval_logging_sample. Historical files are untouched.
"""

from pathlib import Path
import contextlib
import io
import json
import sys
import numpy as np
import nbformat
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import retrieval_methods as methods
from src.result_tables.common import LinearProgram
from src.result_tables.retrieval import retrieve
from src.result_tables.runner import result_table_generator


def check_numerical_invariance():
    baseline_path = ROOT / 'results/retrieval_logging_validation/before.json'
    baseline = (json.loads(baseline_path.read_text())['records']
                if baseline_path.exists() else [])
    comparisons = []
    for seed in (7, 19, 42):
        rng = np.random.default_rng(seed)
        A = sparse.csr_matrix(rng.normal(size=(6, 12)))
        b = A @ rng.uniform(0, 1, 12)
        start = rng.normal(size=12)
        lower, upper = np.zeros(12), np.ones(12)
        for name, method in [('VN', methods.retrieve_apm),
                             ('Dykstra', methods.retrieve_dykstra),
                             ('AARM', methods.retrieve_aarm)]:
            history = []
            x, iterations, status = method(start, A, b, lower, upper, trace=history)
            untraced = method(start, A, b, lower, upper)
            np.testing.assert_array_equal(x, untraced[0])
            assert (iterations, status) == untraced[1:]
            old = next((r for r in baseline if r['seed'] == seed and r['method'] == name), None)
            if old is not None:
                np.testing.assert_array_equal(x, old['x'])
                assert (iterations, status) == (old['iterations'], old['status'])
            comparisons.append(dict(seed=seed, method=name, iterations=iterations,
                                    status=status, matches_untraced=True,
                                    matches_prechange=True if old is not None else None))
        lp = LinearProgram(A, b, rng.normal(size=12), lower, upper)
        traced, untraced = retrieve(start, lp), retrieve(start, lp, collect_trace=False)
        np.testing.assert_array_equal(traced.x, untraced.x)
        old = next((r for r in baseline if r['seed'] == seed and r['method'] == 'pipeline_Dykstra'), None)
        for key, value in untraced.diagnostics.items():
            if key not in ('retrieval_trace', 'retrieval_explanation'):
                assert traced.diagnostics[key] == value, key
        if old is not None:
            np.testing.assert_array_equal(traced.x, old['x'])
            for key, value in old['diagnostics'].items():
                assert traced.diagnostics[key] == value, key
        comparisons.append(dict(seed=seed, method='pipeline_Dykstra',
                                matches_untraced=True, matches_prechange=True if old else None))
    return comparisons


def main():
    output = ROOT / 'results/retrieval_logging_sample'
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(full_published_sweeps_executed=False,
                   numerical_invariance=check_numerical_invariance(), experiments={})
    reference_output = output / 'reference-qr-m450'
    reference_output.mkdir(exist_ok=True)
    notebook_path = ROOT / 'problems/quantile_regression/Liberti_QuantileRegression_m5000_RP_Retrieval.ipynb'
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    scope = {'__name__': '__main__', 'TRACE_OUTPUT_DIR': reference_output}
    console = io.StringIO()
    try:
        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == 'code':
                    source = cell.source.replace('generate_quantile_instance(m=5000, p=399',
                                                 'generate_quantile_instance(m=450, p=399')
                    exec(compile(source, f'{notebook_path.name}:cell{index}', 'exec'), scope)
    finally:
        (reference_output / 'console.log').write_text(console.getvalue(), encoding='utf-8')
    assert scope['m'] == 450
    for name in ('apm', 'dykstra', 'aarm'):
        assert scope[name + '_trace'][-1]['iteration'] == scope[name + '_iterations']
    summary['experiments']['reference'] = dict(output=str(reference_output),
        configured_rows=5000, validation_rows=450,
        statuses={name: scope[name + '_status'] for name in ('apm', 'dykstra', 'aarm')})
    for name, problem, size in [('qr', 'Quantile Regression', 450), ('diet', 'Diet Problem', 80)]:
        console = io.StringIO()
        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
            result = result_table_generator(problem, .2, .2, .1, size, size, 1,
                num_runs=1, master_seed=2026, experiment_name=f'{name}-trace-sample', output_dir=output)
        (result.output_path / 'console.log').write_text(console.getvalue(), encoding='utf-8')
        assert result.raw_trials.success.all()
        trace = result.retrieval_trace
        assert not trace.empty
        saved = pd.read_csv(result.files['retrieval_trace.csv'],
                            dtype={'instance_seed': str, 'projection_seed': str})
        assert len(saved) == len(trace)
        for key in ('instance_seed', 'projection_seed'):
            assert saved[key].tolist() == trace[key].map(str).tolist()
        assert saved.stopping_branch.tolist() == trace.stopping_branch.tolist()
        assert result.files['run.log'].read_text(encoding='utf-8') == console.getvalue()
        saved_config = json.loads(result.files['config.json'].read_text(encoding='utf-8'))
        assert saved_config['master_seed'] == 2026
        for _, group in trace.groupby(['formulation_variant', 'replication']):
            assert group.iloc[-1].stopping_branch != 'continue'
        summary['experiments'][name] = dict(output=str(result.output_path),
            trials=len(result.raw_trials), trace_rows=len(trace),
            statuses=result.raw_trials.retrieval_status.tolist())
    (output / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

"""Run the example notebook's smoke cells and check the native process CPU timer.

Usage from repository root: python scripts/validate_result_tables.py
Use --smoke-only when the saved notebook enables full sweeps. This overrides the
guard in memory before execution, without editing the saved notebook.
Outputs go to results/result_table_generator_smoke.
The source notebook stays unexecuted and lightweight in version control.
"""

from pathlib import Path
import ast
import argparse
import json
import os
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.result_tables.common import Timer
from src.result_tables.config import IMPLEMENTATION_VERSION


def smoke_cell_sources(notebook, force_smoke=False):
    """Check the saved sweep guard before executing even the first notebook cell."""
    sources = [(i, ''.join(cell['source'])) for i, cell in enumerate(notebook['cells'])
               if cell['cell_type'] == 'code']
    assignments = []
    trees = []
    for i, source in sources:
        tree = ast.parse(source, filename=f'result_table_generator.ipynb:cell{i}')
        trees.append((i, tree))
        assignments.extend(node for node in tree.body
                           if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == 'RUN_PUBLISHED'
                                   for target in node.targets))
    if (len(assignments) != 1 or not isinstance(assignments[0].value, ast.Constant)
            or type(assignments[0].value.value) is not bool
            or (assignments[0].value.value and not force_smoke)):
        raise ValueError('Smoke validation requires the saved notebook to contain '
                         'RUN_PUBLISHED = False. No notebook cells were executed.')
    if force_smoke:
        assignments[0].value = ast.Constant(value=False)
        return [(i, ast.unparse(tree)) for i, tree in trees]
    return sources


def main(smoke_only=False):
    notebook = json.loads((ROOT / 'problems/result_table_generator.ipynb').read_text(encoding='utf-8'))
    # Execute the ordinary Python cells in order; no kernel or full experiment run required.
    # Suppress rich display here; HTML/PNG artifacts are independently written by the generator.
    scope = {'__name__': '__main__'}
    for i, source in smoke_cell_sources(notebook, force_smoke=smoke_only):
        exec(compile(source, f'result_table_generator.ipynb:cell{i}', 'exec'), scope)
        if 'display' in scope:
            scope['display'] = lambda *args, **kwargs: None
    assert scope['RUN_PUBLISHED'] is False
    summary = {}
    for name in ('qr_smoke', 'diet_smoke'):
        result = scope[name]
        assert result.raw_trials.success.all(), result.raw_trials[['original_status', 'projected_status', 'message']]
        raw = result.raw_trials
        np.testing.assert_allclose(raw.t_projected, raw.projection_cpu + raw.projected_solve_cpu + raw.retrieval_cpu)
        np.testing.assert_allclose(raw['t_projected / t_original'], raw.t_projected / raw.t_original)
        if name == 'diet_smoke':
            assert (raw.groupby(['epsilon', 'replication']).projection_digest.nunique() == 1).all()
        summary[name] = {
            'trials': len(raw), 'successful': int(raw.success.sum()),
            'original_feasible': int(raw.original_feasible.sum()),
            'original_objective': float(raw.f_original.iloc[0]),
            'max_avgeq': float(raw.avgeq.max()),
            'max_avgin': float(raw.avgin.max()),
            'output': str(result.output_path),
        }
    # Native NumPy work executes in the same process, like Gurobi. Cross-check
    # process_time_ns against OS user+system CPU, not elapsed time.
    matrix = np.random.default_rng(19).normal(size=(800, 800))
    before = os.times()
    with Timer() as native:
        for _ in range(12):
            product = matrix @ matrix.T
    after = os.times()
    os_cpu = after.user + after.system - before.user - before.system
    assert np.isfinite(product).all()
    assert native.cpu > 0
    assert abs(native.cpu - os_cpu) <= .05
    with Timer() as sleeping:
        time.sleep(.08)
    assert sleeping.wall >= .07
    assert sleeping.cpu < sleeping.wall
    summary['clock_check'] = dict(native_cpu=native.cpu, os_user_system_cpu=os_cpu,
                                  native_wall=native.wall, sleep_cpu=sleeping.cpu, sleep_wall=sleeping.wall)
    output = ROOT / f'results/result_table_generator_smoke/validation-v{IMPLEMENTATION_VERSION}.json'
    output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only', action='store_true',
                        help='Force RUN_PUBLISHED=False in memory; never run full sweeps')
    main(smoke_only=parser.parse_args().smoke_only)

"""Run corrected toy notebooks and a 450-row reference QR smoke in memory.

The saved reference remains m=5000; saved notebook outputs are not modified.
Toy outputs go to their v2 directories. No published table sweeps are launched.
"""

from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import sys
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    old = {p: hashlib.sha256(p.read_bytes()).hexdigest()
           for folder in ['toy_example', 'gurobi_toy_example']
           for p in (ROOT / 'results' / folder).iterdir() if p.is_file()}
    summary = {'full_published_sweeps_executed': False, 'notebooks': {}}
    for path in (ROOT / 'problems').rglob('*.ipynb'):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, str(path), 'exec')
        if path.name == 'result_table_generator.ipynb':
            continue
        scope, log = {'__name__': '__main__'}, io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                for index, cell in enumerate(notebook.cells, 1):
                    if cell.cell_type != 'code':
                        continue
                    source = cell.source
                    if path.name.endswith('Retrieval.ipynb'):
                        source = source.replace('generate_quantile_instance(m=5000, p=399',
                                                'generate_quantile_instance(m=450, p=399')
                    exec(compile(source, f'{path.name}:cell{index}', 'exec'), scope)
        except Exception:
            print(log.getvalue()[-4000:])
            raise
        record = {'all_code_cells_executed': True}
        if 'toy_example' in str(path):
            output = scope['OUTPUT_DIR']
            raw, trace = pd.read_csv(output / 'all_runs.csv'), pd.read_csv(output / 'apm_trace.csv')
            assert len(raw) == 50
            assert raw.apm_max_bound_violation.max() <= 1e-10
            assert 'apm_upper_violation_sum' in raw and 'upper_violation_sum' in trace
            record.update(trials=len(raw), apm_feasible=int(raw.apm_feasible.sum()),
                          max_apm_bound_violation=float(raw.apm_max_bound_violation.max()), output=str(output))
        else:
            assert scope['m'] == 450
            record.update(validation_row_size=450, configured_row_size=5000,
                          bisection_statuses={name: scope[name + '_bisection_status']
                                              for name in ['apm', 'dykstra', 'aarm']},
                          feasibility={name: scope['retrieval_feasibility'](
                              scope['x_' + name], scope['A'], scope['b'], scope['lb'], scope['ub'])
                              for name in ['apm', 'dykstra', 'aarm']})
        summary['notebooks'][str(path.relative_to(ROOT))] = record
        print(path.name, json.dumps(record))
    for path, digest in old.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, str(path)
    (ROOT / 'docs/methodology-fixes-validation.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('Notebook validation passed. Historical toy files unchanged.')


if __name__ == '__main__':
    main()

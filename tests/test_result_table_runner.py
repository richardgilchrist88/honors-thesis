"""Experiment orchestration, failure aggregation, export, overwrite and rollback."""

from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import openpyxl
from src.result_tables.common import SolveResult
from src.result_tables import runner, reporting
from src.result_tables.projection import projection_dimension
from src.result_tables.metrics import TABLE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def fake_original(instance):
    return SolveResult('OPTIMAL', status_code=2, x=np.zeros(instance.n), objective=2., cpu_seconds=4.)


def fake_replication(instance, epsilon, projection_seed, original_result, formulation_variant=None):
    objective = float(projection_seed % 10 + 1)
    n = instance.n + (2 * instance.m if formulation_variant == 'QR' else instance.m)
    k = projection_dimension(epsilon, n)
    return dict(row_size=instance.m, epsilon=epsilon, instance_seed=instance.instance_seed,
                projection_seed=projection_seed, formulation_variant=formulation_variant, k=k,
                success=True, original_feasible=False, retrieval_feasible=False,
                f_original=2., f_projected=objective, f_retrieved=objective + 1,
                avgin=.1, avgeq=.2, projected_status='OPTIMAL', projection_digest=str(projection_seed),
                **{'k / m': k / instance.m, 'f_projected / f_original': objective / 2,
                   'f_retrieved / f_original': (objective + 1) / 2, 't_projected / t_original': .5})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='table-test-', dir=ROOT / 'results')
        self.addCleanup(self.temporary.cleanup)
        self.arguments = dict(problem_type='Quantile Regression', epsilon_min=.2, epsilon_max=.3,
                              epsilon_increment=.1, row_min=10, row_max=20, row_increment=10,
                              num_runs=3, master_seed=123, output_dir=self.temporary.name,
                              experiment_name='test')

    def test_loops_cache_original_seed_records_and_reproducibility(self):
        q = runner.quantile_regression
        with patch.object(q, 'generate_instance', wraps=q.generate_instance) as generate, \
             patch.object(q, 'solve_original', side_effect=fake_original) as original, \
             patch.object(q, 'solve_one_replication', side_effect=fake_replication) as single, \
             patch.object(runner, 'save_outputs'):
            first = runner.result_table_generator(**self.arguments)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(original.call_count, 2)
            self.assertEqual(single.call_count, 12)
            second = runner.result_table_generator(**self.arguments)
        pd.testing.assert_frame_equal(first.raw_trials, second.raw_trials)
        self.assertEqual(len(first.averaged_results), 4)
        self.assertEqual(first.raw_trials.projection_seed.nunique(), 12)
        self.assertEqual(first.raw_trials.instance_seed.nunique(), 2)
        self.assertTrue((first.averaged_results.successful_runs == 3).all())
        self.assertTrue((first.averaged_results.failed_runs == 0).all())
        self.assertEqual(len(first.metadata['trials']), 12)
        for keys, group in first.raw_trials.groupby(['row_size', 'epsilon']):
            averaged = first.averaged_results.query('row_size == @keys[0] and epsilon == @keys[1]')
            self.assertAlmostEqual(averaged['f_projected / f_original'].iloc[0],
                                   group['f_projected / f_original'].mean())

    def test_diet_pairs_share_seeds_and_original_instance(self):
        d = runner.diet_problem
        with patch.object(d, 'solve_original', side_effect=fake_original) as original, \
             patch.object(d, 'solve_one_replication', side_effect=fake_replication) as single, \
             patch.object(runner, 'save_outputs'):
            result = runner.result_table_generator(**(self.arguments | {'problem_type': 'Diet Problem'}))
        self.assertEqual(original.call_count, 2)
        self.assertEqual(single.call_count, 24)
        for _, pair in result.raw_trials.groupby(['row_size', 'epsilon', 'replication']):
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair.projection_seed.nunique(), 1)
            self.assertEqual(pair.instance_seed.nunique(), 1)
        self.assertEqual(set(result.tables), {'TDP1', 'TDP2'})

    def test_instances_and_overlapping_projections_survive_grid_and_run_changes(self):
        q = runner.quantile_regression
        with patch.object(q, 'solve_original', side_effect=fake_original), \
             patch.object(q, 'solve_one_replication', side_effect=fake_replication), \
             patch.object(runner, 'save_outputs'):
            first = runner.result_table_generator(**self.arguments)
            changed = runner.result_table_generator(**(self.arguments | {
                'epsilon_min': .3, 'epsilon_max': .4, 'row_min': 20, 'num_runs': 2}))
            disjoint = runner.result_table_generator(**(self.arguments | {
                'epsilon_min': .1, 'epsilon_max': .1, 'num_runs': 1}))
        original_seeds = {r['row_size']: r['instance_seed'] for r in first.metadata['instances']}
        self.assertEqual(original_seeds, {r['row_size']: r['instance_seed'] for r in disjoint.metadata['instances']})
        self.assertEqual(changed.metadata['instances'][0]['instance_seed'], original_seeds[20])
        old = first.raw_trials.query('row_size == 20 and epsilon == .3 and replication <= 2')
        new = changed.raw_trials.query('epsilon == .3')
        self.assertEqual(old.projection_seed.tolist(), new.projection_seed.tolist())
        self.assertEqual(old.f_projected.tolist(), new.f_projected.tolist())

    def test_keyed_seeds_separate_problem_instance_and_projection_identities(self):
        from src.result_tables.seeding import experiment_seed
        values = [experiment_seed(42, 'Quantile Regression', 5000),
                  experiment_seed(42, 'Diet Problem', 5000),
                  experiment_seed(43, 'Quantile Regression', 5000),
                  experiment_seed(42, 'Quantile Regression', 6000),
                  experiment_seed(42, 'Quantile Regression', 5000, epsilon=.2, replication=1),
                  experiment_seed(42, 'Quantile Regression', 5000, epsilon=.2, replication=2),
                  experiment_seed(42, 'Quantile Regression', 5000, epsilon=.3, replication=1)]
        self.assertEqual(len(set(values)), len(values))

    def test_failed_and_zero_objective_runs_visible_in_averages(self):
        q = runner.quantile_regression
        with patch.object(q, 'solve_original', side_effect=fake_original), \
             patch.object(q, 'solve_one_replication', side_effect=fake_replication), \
             patch.object(runner, 'save_outputs'):
            result = runner.result_table_generator(**self.arguments)
        raw = result.raw_trials.copy()
        group = (raw.row_size == 10) & (raw.epsilon == .2)
        indices = raw.index[group]
        raw.loc[indices[0], 'success'] = False
        raw.loc[indices[0], 'avgin'] = 9999  # Must not contaminate successful averages.
        raw.loc[indices[1:], 'f_projected / f_original'] = np.nan
        raw.loc[(raw.row_size == 20) & (raw.epsilon == .3), 'success'] = False
        averaged = reporting.aggregate_trials(raw)
        first = averaged.query('row_size == 10 and epsilon == .2').iloc[0]
        self.assertEqual(first.successful_runs, 2)
        self.assertEqual(first.failed_runs, 1)
        self.assertEqual(first.avgin, .1)
        self.assertTrue(np.isnan(first['f_projected / f_original']))
        self.assertEqual(first.projected_objective_ratio_runs, 0)
        failed = averaged.query('row_size == 20 and epsilon == .3').iloc[0]
        self.assertEqual(failed.failed_runs, 3)
        self.assertTrue(np.isnan(failed.avgeq))
        self.assertGreater(failed['k / m'], 0)

    def test_random_master_seed_stored_and_configuration_paths_distinct(self):
        q = runner.quantile_regression
        with patch.object(q, 'solve_original', side_effect=fake_original), \
             patch.object(q, 'solve_one_replication', side_effect=fake_replication), \
             patch.object(runner, 'save_outputs'), patch.object(runner.secrets, 'randbits', return_value=991):
            result = runner.result_table_generator(**(self.arguments | {'master_seed': None}))
            replay = runner.result_table_generator(**(self.arguments | {'master_seed': 991}))
            other = runner.result_table_generator(**(self.arguments | {'epsilon_max': .4}))
        self.assertEqual(result.metadata['master_seed'], 991)
        pd.testing.assert_frame_equal(result.raw_trials, replay.raw_trials)
        self.assertNotEqual(other.output_path, replay.output_path)

    def test_real_exports_overwrite_and_preserve_seed_precision(self):
        q = runner.quantile_regression
        args = self.arguments | {'row_max': 10}
        with patch.object(q, 'solve_original', side_effect=fake_original), \
             patch.object(q, 'solve_one_replication', side_effect=fake_replication):
            first = runner.result_table_generator(**args)
            unrelated = first.output_path / 'keep-me.txt'
            unrelated.write_text('unrelated', encoding='utf-8')
            names = {p.name for p in first.output_path.iterdir()}
            # Existing raw file has stale rows; next run must replace all of it.
            with first.files['raw_trials.csv'].open('a') as handle:
                handle.write('stale data\n')
            second = runner.result_table_generator(**args)
        self.assertEqual(first.output_path, second.output_path)
        self.assertEqual({p.name for p in second.output_path.iterdir()}, names)
        self.assertEqual(unrelated.read_text(), 'unrelated')
        raw = pd.read_csv(second.files['raw_trials.csv'], dtype={'instance_seed': str, 'projection_seed': str})
        self.assertEqual(len(raw), 6)
        self.assertEqual(raw.projection_seed.tolist(), second.raw_trials.projection_seed.map(str).tolist())
        self.assertEqual(list(second.tables['QR']), TABLE_COLUMNS)
        self.assertIn('successful', second.files['preview_QR.html'].read_text(encoding='utf-8'))
        self.assertIn(r'\multicolumn{7}', second.files['table_QR.tex'].read_text())
        workbook = openpyxl.load_workbook(second.files['results.xlsx'], read_only=True)
        try:
            rows = list(workbook['raw_trials'].values)
            seed_column = rows[0].index('projection_seed')
            self.assertEqual([r[seed_column] for r in rows[1:]], raw.projection_seed.tolist())
            self.assertTrue({'QR', 'QR_counts', 'raw_trials', 'averaged_results'} <= set(workbook.sheetnames))
        finally:
            workbook.close()
        meta = json.loads(second.files['metadata.json'].read_text())
        for name, digest in meta['artifacts'].items():
            self.assertEqual(hashlib.sha256((second.output_path / name).read_bytes()).hexdigest(), digest)
        for name in ['QR_objectives.png', 'QR_feasibility.png', 'QR_size_cpu.png']:
            self.assertTrue(second.files[name].read_bytes().startswith(b'\x89PNG'))
        old_metadata = second.files['metadata.json'].read_bytes()
        old_raw = second.files['raw_trials.csv'].read_bytes()
        with patch.object(reporting, 'save_plots', side_effect=RuntimeError('plot failed')):
            with self.assertRaisesRegex(RuntimeError, 'plot failed'):
                reporting.save_outputs(second)
        self.assertEqual(second.files['metadata.json'].read_bytes(), old_metadata)
        self.assertEqual(second.files['raw_trials.csv'].read_bytes(), old_raw)
        self.assertFalse((second.output_path / '.result_table_generator.lock').exists())

        real_replace = reporting.os.replace
        attempts = 0

        def fail_during_commit(source, destination):
            nonlocal attempts
            attempts += 1
            if attempts == 3:
                raise OSError('commit interrupted')
            return real_replace(source, destination)

        with patch.object(reporting.os, 'replace', side_effect=fail_during_commit):
            with self.assertRaisesRegex(OSError, 'commit interrupted'):
                reporting.save_outputs(second)
        self.assertEqual(second.files['metadata.json'].read_bytes(), old_metadata)
        for name, digest in meta['artifacts'].items():
            self.assertEqual(hashlib.sha256((second.output_path / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(unrelated.read_text(), 'unrelated')

    def test_all_failed_experiment_exports_undefined_metrics_and_counts(self):
        args = self.arguments | {'row_max': 10, 'epsilon_max': .2, 'num_runs': 2}
        with patch.object(runner.quantile_regression, 'solve_original',
                          return_value=SolveResult('GUROBI_UNAVAILABLE', message='test unavailable')):
            with self.assertWarnsRegex(RuntimeWarning, '2 trial'):
                result = runner.result_table_generator(**args)
        self.assertEqual(len(result.raw_trials), 2)
        self.assertTrue(result.raw_trials.avgeq.isna().all())
        self.assertTrue(result.raw_trials.f_projected.isna().all())
        self.assertEqual(result.averaged_results.failed_runs.iloc[0], 2)
        self.assertEqual(result.averaged_results.successful_runs.iloc[0], 0)
        self.assertTrue(np.isnan(result.tables['QR']['f_retrieved / f_original'].iloc[0]))
        metadata = json.loads(result.files['metadata.json'].read_text())
        self.assertEqual(metadata['failed_runs'], 2)
        self.assertEqual(len(metadata['trials']), 2)
        self.assertTrue(result.files['results.xlsx'].exists())
        self.assertTrue(pd.read_csv(result.files['retrieval_trace.csv']).empty)
        self.assertIn('failure stage: original', result.files['run.log'].read_text(encoding='utf-8'))
        saved_config = json.loads(result.files['config.json'].read_text(encoding='utf-8'))
        self.assertEqual(saved_config['parameters']['num_runs'], 2)


if __name__ == '__main__':
    unittest.main()

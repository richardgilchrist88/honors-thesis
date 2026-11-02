"""Numerical/contract tests that do not require a Gurobi installation or license."""

import math
from decimal import Decimal
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from src.result_tables import config, quantile_regression as qr, diet_problem as diet
from src.result_tables.common import Instance, LinearProgram, SolveResult, Timer
from src.result_tables.metrics import quantile_metrics, diet_metrics, ratio
from src.result_tables.projection import projection_dimension, sparse_gaussian, projection_fingerprint
from src.result_tables.retrieval import retrieve, affine_line_projection, bisect_toward_projected
from src.result_tables.gurobi_solver import solve_lp
from src.result_tables.runner import result_table_generator


class ValidationTests(unittest.TestCase):
    def test_inclusive_grids(self):
        self.assertEqual(config.epsilon_grid(.2, .9, .1), [.2, .3, .4, .5, .6, .7, .8, .9])
        self.assertEqual(config.epsilon_grid(.1, .15, .05), [.1, .15])
        self.assertEqual(config.epsilon_grid(Decimal('.1'), Decimal('.15'), Decimal('.05')), [.1, .15])
        self.assertEqual(config.row_grid(5000, 9000, 1000), [5000, 6000, 7000, 8000, 9000])
        self.assertEqual(config.row_grid(5, 5, 2), [5])

    def test_invalid_grids(self):
        for args in ((0, .5, .1), (.1, 1, .1), (.5, .1, .1), (.1, .5, 0),
                     (.1, .5, -.1), (.1, .45, .1), (True, .5, .1), (float('nan'), .5, .1),
                     (.1, float('inf'), .1), ('0.1', .5, .1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                config.epsilon_grid(*args)
        for args in ((0, 5, 1), (5, 1, 1), (1, 5, 0), (1., 5, 1), (1, 6, 2), (True, 5, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                config.row_grid(*args)

    def test_generator_argument_validation_before_any_solve(self):
        arguments = dict(problem_type='Quantile Regression', epsilon_min=.2, epsilon_max=.2,
                         epsilon_increment=.1, row_min=10, row_max=10, row_increment=1)
        invalid = [('num_runs', 0), ('num_runs', 1.5), ('num_runs', True), ('master_seed', -1),
                   ('master_seed', True), ('problem_type', 'quantile regression'), ('problem_type', []),
                   ('experiment_name', ''), ('experiment_name', 42), ('output_dir', '')]
        with patch.object(qr, 'solve_original') as solve:
            for key, value in invalid:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    result_table_generator(**(arguments | {key: value}))
            solve.assert_not_called()
        for problem in ('Basis Pursuit', 'Max Flow'):
            with self.assertRaisesRegex(NotImplementedError, problem):
                result_table_generator(**(arguments | {'problem_type': problem}))


class InstanceProjectionTests(unittest.TestCase):
    def test_published_instance_dimensions_distributions_and_seed(self):
        q = qr.generate_instance(5000, 42)
        self.assertEqual(q.D.shape, (5000, 399))
        self.assertEqual(q.b.shape, (5000,))
        self.assertEqual(q.tau, .2)
        self.assertTrue(np.all((-1 <= q.D.data) & (q.D.data <= 1)))
        self.assertAlmostEqual(q.D.nnz / (5000 * 399), .8, delta=.002)
        self.assertAlmostEqual(np.count_nonzero(q.b) / 5000, .8, delta=.025)
        self.assertAlmostEqual(float(q.D.data.mean()), 0, delta=.005)
        same = qr.generate_instance(5000, 42)
        self.assertEqual((q.D != same.D).nnz, 0)
        np.testing.assert_array_equal(q.b, same.b)
        d = diet.generate_instance(3000, 17)
        self.assertEqual(d.D.shape, (3000, 400))
        self.assertAlmostEqual(d.D.nnz / (3000 * 400), .5, delta=.002)
        for values in (d.D.data, d.b, d.cost):
            self.assertTrue(np.all((0 <= values) & (values <= 1)))
            self.assertAlmostEqual(float(values.mean()), .5, delta=.04)

    def test_dimension_rounds_before_log_and_does_not_clip(self):
        self.assertEqual(projection_dimension(.2, 10399), 231)
        self.assertEqual(projection_dimension(.1, 10399), 924)
        self.assertEqual(projection_dimension(.15, 10399), 406)
        self.assertEqual(projection_dimension(.2, 3400), 203)
        k = projection_dimension(.3, 10399)
        self.assertEqual(k, int(11 * math.log(10399)))
        self.assertNotEqual(k, round((1 / .3) ** 2 * math.log(10399)))
        self.assertGreater(projection_dimension(.1, 419), 10)

    def test_projection_support_scaling_and_seed(self):
        A = sparse.eye(400, format='csr') + sparse.csr_matrix(np.ones((400, 400)))
        T, sigma = sparse_gaussian(A, 600, 123)
        self.assertEqual(T.shape, (600, 400))
        self.assertEqual(T.nnz, round(sigma * 600 * 400))
        self.assertEqual(sigma, .5)
        self.assertAlmostEqual(float(T.data.mean()), 0, delta=.002)
        self.assertAlmostEqual(float(T.data.var()), 1 / (600 * sigma), delta=.00006)
        other, _ = sparse_gaussian(A, 600, 123)
        different, _ = sparse_gaussian(A, 600, 124)
        self.assertEqual(projection_fingerprint(T), projection_fingerprint(other))
        self.assertNotEqual(projection_fingerprint(T), projection_fingerprint(different))

    def test_paired_diet_projection_and_lp_blocks(self):
        instance = diet.generate_instance(15, 10)
        first, retrieval1, T1, sigma1 = diet.build_projected(instance, 7, 11, 'TDP1')
        second, retrieval2, T2, sigma2 = diet.build_projected(instance, 7, 11, 'TDP2')
        np.testing.assert_array_equal(T1.toarray(), T2.toarray())
        self.assertEqual(sigma1, sigma2)
        self.assertEqual(first.A.shape, (7, 415))
        self.assertEqual(second.A.shape, (7, 414))
        np.testing.assert_allclose(first.A[:, :400].toarray(), (T1 @ instance.D).toarray())
        np.testing.assert_allclose(first.A[:, 400:].toarray(), -T1.toarray())
        np.testing.assert_array_equal(second.A[:, 400:407].toarray(), np.eye(7))
        np.testing.assert_array_equal(second.A[:, 407:].toarray(), -np.eye(7))
        np.testing.assert_array_equal(first.c[400:], np.full(15, 800))
        np.testing.assert_array_equal(second.c[400:], np.full(14, 800))
        self.assertEqual(retrieval1.A.shape, (15, 415))
        self.assertEqual(retrieval2.A.shape, (15, 400))

    def test_quantile_free_bounds_and_slack_signs(self):
        instance = Instance(sparse.csr_matrix([[1.], [2.]]), np.array([-1., 3.]), None, 1)
        lp = qr.build_original(instance)
        self.assertTrue(np.isneginf(lp.lower[0]))
        np.testing.assert_array_equal(lp.lower[1:], np.zeros(4))
        np.testing.assert_allclose(lp.c, [0, .2, .2, .8, .8])
        projected, _, T, _ = qr.build_projected(instance, 3, 10)
        np.testing.assert_allclose(projected.A[:, 1:3].toarray(), T.toarray())
        np.testing.assert_allclose(projected.A[:, 3:].toarray(), -T.toarray())


class RetrievalMetricTests(unittest.TestCase):
    def test_metrics_use_correct_spaces_and_denominators(self):
        q = Instance(sparse.csr_matrix([[1.], [1.]]), np.array([1., 2.]), None, 1)
        x = np.array([-2., -1., 3., -4., 1.])
        lp = qr.build_original(q)
        metrics = quantile_metrics(q, x, lp)
        self.assertEqual(metrics['avgin'], 2.5)
        self.assertEqual(metrics['maxin'], 5.)
        self.assertEqual(metrics['max_bound_violation'], 4.)
        self.assertEqual(metrics['avgeq'], np.abs(lp.A @ x - q.b).mean())
        d = Instance(sparse.csr_matrix([[1., 0.], [0., 2.]]), np.array([1., 1.]), np.ones(2), 2)
        metrics = diet_metrics(d, np.array([2., -.5, 999.]))
        self.assertEqual(metrics['avgin'], .25)
        self.assertEqual(metrics['avgeq'], 1.)
        self.assertEqual(metrics['maxeq'], 2.)
        self.assertTrue(math.isnan(ratio(1, 0)))
        self.assertTrue(math.isnan(ratio(1, 1e-12)))
        self.assertEqual(ratio(4, 2), 2.)

    def test_dykstra_respects_free_coefficients_and_diagnoses_inconsistency(self):
        lp = LinearProgram(sparse.eye(2, format='csr'), np.array([-1., 2.]), np.ones(2),
                           np.array([-np.inf, 0.]), np.full(2, np.inf))
        result = retrieve(np.zeros(2), lp)
        np.testing.assert_allclose(result.x, [-1., 2.], atol=1e-10)
        self.assertTrue(result.diagnostics['retrieval_feasible'])
        self.assertLessEqual(result.diagnostics['retrieval_iterations'], 30)
        inconsistent = LinearProgram(sparse.csr_matrix([[1.], [1.]]), np.array([0., 10.]),
                                     np.ones(1), np.zeros(1), np.full(1, np.inf))
        answer = retrieve(np.zeros(1), inconsistent)
        self.assertFalse(answer.diagnostics['retrieval_feasible'])
        self.assertGreater(answer.diagnostics['retrieval_equality_l2'], 1)

    def test_bisection_improves_minimization_without_losing_feasibility(self):
        lp = LinearProgram(sparse.csr_matrix([[1., 1.]]), np.array([1.]),
                           np.array([1., 0.]), np.zeros(2), np.full(2, np.inf))
        result, steps, _ = bisect_toward_projected(np.array([1., 0.]), np.array([0., .9]), lp, config.RETRIEVAL)
        self.assertGreater(steps, 0)
        self.assertLess(float(lp.c @ result), 1.)
        self.assertLessEqual(float(np.linalg.norm(lp.A @ result - lp.b)), .01)
        self.assertTrue((result >= 0).all())

    def test_degenerate_affine_line_is_finite(self):
        A = sparse.eye(2)
        result = affine_line_projection(np.zeros(2), A, np.ones(2), np.ones(2),
                                        lambda A, rhs: rhs)
        np.testing.assert_array_equal(result, [1., 1.])

    def test_missing_gurobi_is_a_reportable_status(self):
        lp = qr.build_original(qr.generate_instance(2, 1))
        with patch.dict('sys.modules', {'gurobipy': None}):
            result = solve_lp(lp, config.ORIGINAL_SOLVER)
        self.assertEqual(result.status, 'GUROBI_UNAVAILABLE')
        self.assertIsNone(result.objective)
        self.assertIsNone(result.x)

    def test_failed_original_never_fabricates_trial_metrics(self):
        instance = qr.generate_instance(5, 1)
        with patch('src.result_tables.pipeline.solve_lp') as solve:
            trial = qr.solve_one_replication(instance, .2, 2, SolveResult('INFEASIBLE'))
        solve.assert_not_called()
        self.assertFalse(trial['success'])
        self.assertEqual(trial['failure_stage'], 'original')
        self.assertTrue(math.isnan(trial['avgeq']))
        self.assertTrue(math.isnan(trial['f_projected / f_original']))

    def test_timer_uses_process_and_wall_clocks_separately(self):
        with patch('src.result_tables.common.process_time_ns', side_effect=[10, 210]), \
             patch('src.result_tables.common.perf_counter_ns', side_effect=[100, 1100]):
            with Timer() as t:
                pass
        self.assertEqual(t.cpu, 200 / 1e9)
        self.assertEqual(t.wall, 1000 / 1e9)


if __name__ == '__main__':
    unittest.main()

"""Stopping precedence, iteration counts, exports, and numerical invariance."""

from dataclasses import replace
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import sparse
from src import retrieval_methods as methods
from src.result_tables import config
from src.result_tables.common import LinearProgram, SolveResult
from src.result_tables import quantile_regression as qr
from src.result_tables.retrieval import retrieve
from src.result_tables.reporting import iteration_table
from src.result_tables.retrieval_trace import TRACE_COLUMNS, stopping_explanation


class StoppingTraceTests(unittest.TestCase):
    def zero_lp(self, rhs=1.):
        return LinearProgram(sparse.csr_matrix((1, 1)), np.array([rhs]),
                             np.zeros(1), np.zeros(1), np.full(1, np.inf))

    def test_initial_feasibility_reports_zero_updates_and_no_step(self):
        lp = self.zero_lp(rhs=0.)
        answer = retrieve(np.zeros(1), lp)
        self.assertEqual(answer.diagnostics['retrieval_iterations'], 0)
        last = answer.diagnostics['retrieval_trace'][-1]
        self.assertEqual(last['iteration'], 0)
        self.assertEqual(last['stopping_branch'], 'feasible')
        self.assertIsNone(last['step_norm'])
        self.assertFalse(last['step_checked'])

    def test_combined_error_precedes_small_step_and_does_not_certify_feasibility(self):
        # Both combined-error and small-step inequalities hold, but equality fails.
        lp = self.zero_lp(rhs=.015)
        with patch('src.result_tables.retrieval.affine_line_projection', return_value=np.zeros(1)):
            answer = retrieve(np.array([-.005]), lp)
        last = answer.diagnostics['retrieval_trace'][-1]
        self.assertEqual(last['stopping_branch'], 'small_combined_error')
        self.assertLessEqual(last['step_norm'], last['step_tolerance'])
        self.assertFalse(last['step_checked'])
        self.assertFalse(last['feasible'])
        self.assertFalse(answer.diagnostics['retrieval_feasible'])
        self.assertIn('step check was not evaluated', answer.diagnostics['retrieval_explanation'])
        self.assertIn('Bisection: skipped_infeasible_start', answer.diagnostics['retrieval_explanation'])

    def test_stagnation_is_recorded_with_failed_combined_check(self):
        with patch('src.result_tables.retrieval.affine_line_projection', return_value=np.array([-1.])):
            answer = retrieve(np.array([-1.]), self.zero_lp())
        last = answer.diagnostics['retrieval_trace'][-1]
        self.assertEqual(last['stopping_branch'], 'stagnated')
        self.assertEqual(last['step_norm'], 0.)
        self.assertTrue(last['step_checked'])
        self.assertGreater(last['combined_error'], last['combined_tolerance'])

    def test_iteration_cap_does_not_override_an_earlier_exit(self):
        settings = replace(config.RETRIEVAL, max_iterations=1)
        with patch('src.result_tables.retrieval.affine_line_projection', return_value=np.ones(1)):
            answer = retrieve(np.array([-1.]), self.zero_lp(), settings)
        last = answer.diagnostics['retrieval_trace'][-1]
        self.assertEqual(last['stopping_branch'], 'iteration_limit')
        self.assertEqual(last['iteration'], 1)
        self.assertEqual(last['max_iterations'], 1)
        self.assertGreater(last['step_norm'], last['step_tolerance'])
        with patch('src.result_tables.retrieval.affine_line_projection', return_value=np.array([-1.])):
            answer = retrieve(np.array([-1.]), self.zero_lp(), settings)
        self.assertEqual(answer.diagnostics['retrieval_trace'][-1]['stopping_branch'], 'stagnated')

    def test_all_methods_preserve_seeded_solutions_counts_and_statuses(self):
        for seed in (7, 19, 42):
            rng = np.random.default_rng(seed)
            A = sparse.csr_matrix(rng.normal(size=(6, 12)))
            b = A @ rng.uniform(0, 1, 12)
            start = rng.normal(size=12)
            lower, upper = np.zeros(12), np.ones(12)
            for method in (methods.retrieve_apm, methods.retrieve_dykstra, methods.retrieve_aarm):
                with self.subTest(seed=seed, method=method.__name__):
                    expected = method(start, A, b, lower, upper)
                    history = []
                    actual = method(start, A, b, lower, upper, trace=history)
                    np.testing.assert_array_equal(actual[0], expected[0])
                    self.assertEqual(actual[1:], expected[1:])
                    self.assertEqual(history[-1]['iteration'], actual[1])
                    self.assertNotEqual(history[-1]['stopping_branch'], 'continue')
            lp = LinearProgram(A, b, rng.normal(size=12), lower, upper)
            traced, untraced = retrieve(start, lp), retrieve(start, lp, collect_trace=False)
            np.testing.assert_array_equal(traced.x, untraced.x)
            for key in untraced.diagnostics:
                if key not in ('retrieval_trace', 'retrieval_explanation'):
                    self.assertEqual(traced.diagnostics[key], untraced.diagnostics[key])

    def test_vn_and_aarm_record_initial_feasibility_stagnation_and_cap(self):
        for method in (methods.retrieve_apm, methods.retrieve_aarm):
            with self.subTest(method=method.__name__):
                lp = self.zero_lp(rhs=0.)
                history = []
                _, iterations, _ = method(np.zeros(1), lp.A, lp.b, lp.lower, lp.upper, trace=history)
                self.assertEqual(iterations, 0)
                self.assertEqual(history[-1]['stopping_branch'], 'feasible')
                lp = self.zero_lp()
                history = []
                method(np.zeros(1), lp.A, lp.b, lp.lower, lp.upper, trace=history)
                self.assertEqual(history[-1]['stopping_branch'], 'stagnated')
                history = []
                projected = np.array([2. if method is methods.retrieve_apm else -2.])
                with patch.object(methods, 'solve_projection_system', return_value=projected):
                    method(np.zeros(1), lp.A, lp.b, lp.lower, lp.upper, max_iter=1, trace=history)
                self.assertEqual(history[-1]['stopping_branch'], 'iteration_limit')

    def test_flattened_trace_retains_trial_identity_and_uint64_seeds(self):
        answer = retrieve(np.zeros(1), self.zero_lp(rhs=0.))
        history = answer.diagnostics['retrieval_trace']
        large_seed = 2**64 - 9
        trial = dict(problem_type='Diet Problem', row_size=80, epsilon=.2,
                     replication=3, formulation_variant='TDP2', instance_seed=large_seed,
                     projection_seed=large_seed - 1, retrieval_trace=json.dumps(history))
        frame = iteration_table(pd.DataFrame([trial, trial | {'retrieval_trace': '[]'}]))
        self.assertEqual(len(frame), 1)
        self.assertEqual(int(frame.instance_seed.iloc[0]), large_seed)
        self.assertEqual(int(frame.projection_seed.iloc[0]), large_seed - 1)
        self.assertEqual(frame.formulation_variant.iloc[0], 'TDP2')
        self.assertEqual(frame.replication.iloc[0], 3)
        self.assertTrue(set(TRACE_COLUMNS) <= set(frame.columns))
        empty = iteration_table(pd.DataFrame([trial | {'retrieval_trace': '[]'}]))
        self.assertTrue(empty.empty)
        self.assertEqual(list(frame.columns), list(empty.columns))

    def test_numerical_failure_is_distinguished_from_a_skipped_retrieval(self):
        instance = qr.generate_instance(2, 1)
        size = instance.n + 2 * instance.m
        solved = SolveResult('OPTIMAL', x=np.zeros(size), objective=1.)
        with patch('src.result_tables.pipeline.solve_lp', return_value=solved), \
             patch('src.result_tables.pipeline.retrieve', side_effect=FloatingPointError('test nonfinite')):
            trial = qr.solve_one_replication(instance, .2, 2, solved)
        self.assertFalse(trial['success'])
        self.assertEqual(trial['retrieval_status'], 'NUMERICAL_ERROR')
        self.assertIn('failed before returning a complete trace', trial['retrieval_explanation'])
        self.assertEqual(trial['retrieval_trace'], '[]')


if __name__ == '__main__':
    unittest.main()

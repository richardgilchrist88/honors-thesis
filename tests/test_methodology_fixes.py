"""Regression witnesses from the methodology audit, without optimization calls."""

import unittest
import numpy as np
from scipy import sparse
from scripts.audit_methodology import notebook_functions, REFERENCE, TOYS
from src import retrieval_methods as methods
from src import toy_retrieval as toy
from src.result_tables.common import LinearProgram
from src.result_tables.retrieval import retrieve


class RetrievalFixTests(unittest.TestCase):
    def test_von_neumann_returns_original_feasible_point_in_audit_witness(self):
        f, _ = notebook_functions(REFERENCE, {'retrieve_apm'})
        A = sparse.csr_matrix([[1., 0., 0.], [0., 1., 1.]])
        b = np.array([1., 1.])
        x, _, status = f['retrieve_apm'](np.array([1., 0., 3.]), A, b, np.zeros(3), np.full(3, np.inf))
        self.assertLessEqual(np.linalg.norm(A @ x - b), .01)
        self.assertTrue((x >= 0).all())
        self.assertEqual(status, 'feasible')

    def test_dykstra_keeps_feasible_clipped_point_and_matches_table_implementation(self):
        f, _ = notebook_functions(REFERENCE, {'retrieve_dykstra'})
        A, b = sparse.csr_matrix([[0., 1.]]), np.array([1.])
        start, lower, upper = np.array([-.1, 1.]), np.zeros(2), np.full(2, np.inf)
        x, _, _ = f['retrieve_dykstra'](start, A, b, lower, upper)
        np.testing.assert_allclose(x, [0., 1.])
        expected = retrieve(start, LinearProgram(A, b, np.zeros(2), lower, upper))
        np.testing.assert_array_equal(x, expected.x)

    def test_aarm_uses_the_papers_composed_reflections(self):
        f, _ = notebook_functions(REFERENCE, {'retrieve_aarm'})
        A = sparse.csr_matrix([[1., 0., 0.], [0., 1., 1.]])
        x, _, _ = f['retrieve_aarm'](np.array([1., 0., 3.]), A, np.array([1., 1.]),
                                     np.zeros(3), np.full(3, np.inf), max_iter=1)
        # Hand-derived from P_U, then R_U, then P_S, R_S, averaging (alpha=beta=.9).
        np.testing.assert_allclose(x, [.676, .62, 1.352], atol=1e-10)

    def test_feasibility_requires_both_original_sets_and_preserves_free_beta(self):
        A = sparse.eye(2, format='csr')
        lower, upper = np.array([-np.inf, 0.]), np.full(2, np.inf)
        for method in (methods.retrieve_apm, methods.retrieve_dykstra, methods.retrieve_aarm):
            with self.subTest(method=method.__name__):
                x, _, _ = method(np.zeros(2), A, np.array([-2., 1.]), lower, upper)
                np.testing.assert_allclose(x, [-2., 1.])
        self.assertFalse(methods.retrieval_feasibility(np.array([1., 1.]), A, np.array([1., 3.]), lower, upper))
        self.assertFalse(methods.retrieval_feasibility(np.array([1., -.1]), A, np.array([1., -.1]), lower, upper))


class ToyFixTests(unittest.TestCase):
    def test_notebooks_enforce_upper_bound_and_report_remaining_infeasibility(self):
        A, b = np.array([[1., 0., 0.], [0., 1., 1.]]), np.array([1., 9.])
        for path in TOYS:
            f, _ = notebook_functions(path, {'apm_retrieval', 'compute_metrics'})
            for mode in ('exact', 'approx_lsmr'):
                with self.subTest(path=path, mode=mode):
                    x, _, trace = f['apm_retrieval'](A, b, np.array([1., 5., 0.]), mode=mode, upper_bound=5.)
                    self.assertTrue((x >= 0).all() and (x <= 5).all())
                    self.assertLessEqual(np.linalg.norm(A @ x - b), 1e-6)
                    self.assertTrue(trace.iloc[-1].feasible)
                    metrics = f['compute_metrics'](A, b, np.ones(3), np.array([1., 7., 2.]), upper_bound=5.)
                    self.assertEqual(metrics['upper_violation_sum'], 2.)
                    self.assertFalse(metrics['feasible'])
                    limited, _, trace = f['apm_retrieval'](A, b, np.array([1., 5., 0.]), mode=mode,
                                                         upper_bound=5., max_iter=1)
                    self.assertFalse(trace.iloc[-1].feasible)
                    self.assertGreater(np.linalg.norm(A @ limited - b), 1e-6)

    def test_custom_bound_is_respected_and_lsmr_handles_dependent_equations(self):
        A, b = np.array([[1., 1.], [2., 2.]]), np.array([3., 6.])
        x, _, trace = toy.apm_retrieval(A, b, np.array([2., 0.]), mode='approx_lsmr', upper_bound=2.)
        self.assertTrue(trace.iloc[-1].feasible)
        self.assertLessEqual(x.max(), 2.)

    def test_three_point_projection_has_correct_second_moment(self):
        matrix = toy.achlioptas_projection_matrix(200, 1000, np.random.default_rng(12))
        self.assertAlmostEqual(float(np.mean(matrix**2)), 1/200, delta=.0001)


if __name__ == '__main__':
    unittest.main()

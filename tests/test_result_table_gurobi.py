"""Licensed integration tests, explicitly skipped when Gurobi cannot optimize."""

import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from src.result_tables import config, quantile_regression as qr, diet_problem as diet
from src.result_tables.common import Instance, SolveResult
from src.result_tables.gurobi_solver import solve_lp


class GurobiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import gurobipy as gp
        except ImportError as exc:
            raise unittest.SkipTest(f'Gurobi unavailable: {exc}')
        cls.gp = gp
        try:
            cls.env = gp.Env(empty=True)
            cls.env.setParam('OutputFlag', 0)
            cls.env.start()
            with gp.Model(env=cls.env) as model:
                model.addVar(obj=1)
                model.optimize()
        except gp.GurobiError as exc:
            if hasattr(cls, 'env'):
                cls.env.dispose()
            raise unittest.SkipTest(f'No working Gurobi license: {exc}')

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def test_qr_original_negative_coefficient_matches_known_quantile(self):
        instance = Instance(sparse.csr_matrix(np.ones((3, 1))), np.array([-2., -1., 4.]), None, 1)
        answer = qr.solve_original(instance)
        self.assertEqual(answer.status, 'OPTIMAL')
        self.assertAlmostEqual(answer.x[0], -2.)
        self.assertAlmostEqual(answer.objective, 1.4)
        self.assertEqual(answer.parameters['Method'], -1)

    def test_projected_qr_matches_independent_gurobi_model(self):
        gp = self.gp
        instance = Instance(sparse.csr_matrix([[1.], [2.], [1.]]), np.array([-2., -1., 4.]), None, 1)
        lp, _, T, _ = qr.build_projected(instance, 5, 10)
        answer = solve_lp(lp, config.QUANTILE_SOLVER)
        with gp.Model(env=self.env) as direct:
            beta = direct.addMVar(1, lb=-gp.GRB.INFINITY)
            up, um = direct.addMVar(3), direct.addMVar(3)
            direct.addConstr((T @ instance.D) @ beta + T @ up - T @ um == T @ instance.b)
            direct.setObjective(.2 * up.sum() + .8 * um.sum())
            direct.optimize()
            self.assertEqual(direct.Status, gp.GRB.OPTIMAL)
            self.assertAlmostEqual(answer.objective, direct.ObjVal, places=7)
        self.assertEqual(answer.parameters['Method'], 1)

    def test_diet_original_and_variants_match_direct_models(self):
        gp = self.gp
        instance = Instance(sparse.csr_matrix([[1., 0.], [0., 1.], [1., 1.]]),
                            np.array([1., 2., 2.]), np.array([2., 1.]), 1)
        original = diet.solve_original(instance)
        self.assertTrue(original.optimal)
        self.assertAlmostEqual(original.objective, 4.)
        for variant in ('TDP1', 'TDP2'):
            lp, _, T, _ = diet.build_projected(instance, 2, 17, variant)
            answer = solve_lp(lp, config.DIET_SOLVER)
            with gp.Model(env=self.env) as direct:
                q = direct.addMVar(2)
                if variant == 'TDP1':
                    r = direct.addMVar(3)
                    direct.addConstr((T @ instance.D) @ q - T @ r == T @ instance.b)
                    direct.setObjective(instance.cost @ q + 4 * r.sum())
                else:
                    plus, minus = direct.addMVar(2), direct.addMVar(2)
                    direct.addConstr((T @ instance.D) @ q + plus - minus == T @ instance.b)
                    direct.setObjective(instance.cost @ q + 4 * (plus.sum() + minus.sum()))
                direct.optimize()
                self.assertEqual(direct.Status, gp.GRB.OPTIMAL)
                self.assertAlmostEqual(answer.objective, direct.ObjVal, places=5)
            self.assertEqual(answer.parameters['Method'], 2)
            self.assertEqual(answer.parameters['Crossover'], 0)
            self.assertAlmostEqual(diet.objective(instance, answer.x), instance.cost @ answer.x[:2])

    def test_penalty_objective_is_not_food_cost(self):
        # No food can satisfy projected requirements; only penalized residuals can.
        instance = Instance(sparse.csr_matrix((2, 1)), np.array([1., 2.]), np.ones(1), 5)
        lp, _, _, _ = diet.build_projected(instance, 2, 2, 'TDP2')
        result = solve_lp(lp, config.DIET_SOLVER)
        self.assertTrue(result.optimal)
        self.assertGreater(result.objective, 0)
        self.assertAlmostEqual(diet.objective(instance, result.x), 0., places=6)

    def test_infeasibility_has_no_primal_or_objective(self):
        instance = Instance(sparse.csr_matrix([[0.]]), np.array([1.]), np.ones(1), 0)
        result = diet.solve_original(instance)
        self.assertIn(result.status, ('INFEASIBLE', 'INF_OR_UNBD'))
        self.assertIsNone(result.x)
        self.assertIsNone(result.objective)

    def test_one_call_one_solve_and_k_greater_than_m(self):
        instance = qr.generate_instance(6, 10)
        original = qr.solve_original(instance)
        with patch('src.result_tables.pipeline.solve_lp', wraps=solve_lp) as solve:
            result = qr.solve_one_replication(instance, .1, 11, original)
        self.assertEqual(solve.call_count, 1)
        self.assertGreater(result['k'], instance.m)
        self.assertTrue(result['success'])
        self.assertTrue(np.isnan(result['f_projected / f_original']))  # zero baseline
        self.assertEqual(result['projection_seed'], 11)

    def test_projected_failure_is_recorded_without_retrieval(self):
        instance = qr.generate_instance(6, 10)
        original = qr.solve_original(instance)
        with patch('src.result_tables.pipeline.solve_lp', return_value=SolveResult('TIME_LIMIT', status_code=9)), \
             patch('src.result_tables.pipeline.retrieve') as retrieval:
            result = qr.solve_one_replication(instance, .2, 11, original)
        retrieval.assert_not_called()
        self.assertFalse(result['success'])
        self.assertEqual(result['projected_status'], 'TIME_LIMIT')
        self.assertTrue(np.isnan(result['avgeq']))


if __name__ == '__main__':
    unittest.main()

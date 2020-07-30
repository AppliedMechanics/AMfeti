"""Test routines for nonlinear_solvers module"""

from unittest import TestCase
from amfeti.nonlinear_solvers import NewtonRaphson
from ..tools import CustomDictAssertTest
import numpy as np
from numpy.testing import assert_array_almost_equal
from copy import copy


class TestProblem:
    def __init__(self):
        self.x_sol = np.array([0.0])
        self.res = copy(self.x_sol)

    def nonlinear_func(self, x):
        self.x_sol = copy(x)
        self.res = np.array([0.25 * x[0] ** 3 + 0.15 * x[0] ** 2 - 2.0 * x[0] - 2.1])
        return np.linalg.norm(self.res)

    def deriv_nonlinear_func(self, x):
        return np.array([3 * 0.25 * x ** 2 + 0.3 * x - 2.0])

    def solve_linear(self):
        info_dict = {'iterations': None}
        K = self.deriv_nonlinear_func(self.x_sol)
        return np.linalg.inv(K) @ -self.res, info_dict


class NewtonRaphsonTest(TestCase):
    def setUp(self):
        self.nonlinear_solver = NewtonRaphson()
        self.custom_asserter = CustomDictAssertTest()

    def tearDown(self):
        pass

    def test_set_config(self):
        config_desired = {'atol': 1.0e-08,
                         'rtol': None,
                         'max_iter': 10,
                         'log_iterations': False}

        self.custom_asserter.assert_dict_almost_equal(self.nonlinear_solver._config_dict, config_desired)

        config_test = {'atol': 1e-9,
                       'rtol': 1e-6,
                       'additional_opt': True}

        config_desired = {'atol': 1.0e-09,
                         'rtol': 1e-6,
                         'max_iter': 10,
                         'log_iterations': False,
                         'additional_opt': True}

        self.nonlinear_solver.set_config(config_test)
        self.custom_asserter.assert_dict_almost_equal(self.nonlinear_solver._config_dict, config_desired)

    def test_solve(self):
        self.nonlinear_solver.set_config({'log_iterations': True})

        test_problem = TestProblem()

        x0 = np.array([4.0])
        x, info_dict_actual = self.nonlinear_solver.solve(test_problem.solve_linear, test_problem.nonlinear_func, x0)

        x_sol_desired = np.array([3.0])

        assert_array_almost_equal(test_problem.x_sol, x_sol_desired)

        info_dict_desired = {'residual': np.array([8.3, 1.628192, 0.1394360, 1.409852e-03, 1.493825e-07, 0.0]),
                             'iterations': 5}

        assert_array_almost_equal(info_dict_actual['residual'], info_dict_desired['residual'])
        self.assertEqual(info_dict_actual['iterations'], info_dict_desired['iterations'])

        test_problem2 = TestProblem()
        self.nonlinear_solver.set_config({'max_iter': 2})
        x0 = np.array([4.0])
        x, info_dict_actual = self.nonlinear_solver.solve(test_problem2.solve_linear, test_problem2.nonlinear_func, x0)

        x_sol_desired = np.array([3.00025])

        assert_array_almost_equal(test_problem2.x_sol, x_sol_desired)

        info_dict_desired = {'residual': np.array([8.3, 1.628192, 0.1394360]),
                             'iterations': 2}

        assert_array_almost_equal(info_dict_actual['residual'], info_dict_desired['residual'])
        self.assertEqual(info_dict_actual['iterations'], info_dict_desired['iterations'])

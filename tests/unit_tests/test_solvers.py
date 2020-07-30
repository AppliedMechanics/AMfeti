"""Test routines for solver module"""

from unittest import TestCase
from amfeti.solvers import PCPGsolver, GMRESsolver
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from numpy.testing import assert_array_almost_equal
import numpy as np


class PCPGsolverTest(TestCase):
    def setUp(self):
        self.solver = PCPGsolver()
        self.A = csr_matrix(np.array([[5, -2, 0, 0], [-2, 5, -2, 0], [0, -2, 5, -2], [0, 0, -2, 3]]))
        self.b = np.array([0.5, -0.3, 1.35, 0.18])

    def tearDown(self):
        pass

    def test_solve(self):
        def F_callback(v):
            return self.A @ v

        def residual_callback(v):
            return -self.A @ v + self.b

        def precondition(v):
            return csr_matrix(np.array([[1/5, 0, 0, 0], [0, 1/5, 0, 0], [0, 0, 1/5, 0], [0, 0, 0, 1/3]])) @ v

        x0 = np.array([0.0, 0.0, 0.0, 0.0])

        solution_actual, info_dict_pure_cg = self.solver.solve(F_callback, residual_callback, x0)

        solution_desired = spsolve(self.A, self.b)
        assert_array_almost_equal(solution_actual, solution_desired)

        # Full reorthogonalization
        self.solver.set_config({'save_history': True,
                                'full_reorthogonalization': True})

        solution_actual, info_dict_fReOrth_cg = self.solver.solve(F_callback, residual_callback, x0)

        assert_array_almost_equal(solution_actual, solution_desired)
        self.assertTrue(info_dict_fReOrth_cg['PCPG_iterations'] <= info_dict_pure_cg['PCPG_iterations'])

        # Preconditioner
        self.solver.set_config({'save_history': True,
                                'full_reorthogonalization': False,
                                'precondition': precondition
                                })

        solution_actual, info_dict_precond_cg = self.solver.solve(F_callback, residual_callback, x0)

        assert_array_almost_equal(solution_actual, solution_desired)
        self.assertTrue(info_dict_precond_cg['PCPG_iterations'] <= info_dict_pure_cg['PCPG_iterations'])

    def test_project(self):
        def project(v):
            return csr_matrix(np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])) @ v
        self.solver.set_config({'projection': project})

        solution_actual = self.solver._project(np.array([1, 2, 3, 4]))

        solution_desired = np.array([2, 4, 6, 8])

        assert_array_almost_equal(solution_actual, solution_desired)


class GMRESsolverTest(TestCase):
    def setUp(self):
        self.solver = GMRESsolver()
        self.A = csr_matrix(np.array([[5, -2, 0, 0], [-2, 5, -2, 0], [0, -2, 5, -2], [0, 0, -2, 3]]))
        self.b = np.array([0.5, -0.3, 1.35, 0.18])

    def tearDown(self):
        pass

    def test_solve(self):
        def F_callback(v):
            return self.A @ v

        def residual_callback(v):
            return -self.A @ v + self.b

        def precondition(v):
            return csr_matrix(np.array([[1/5, 0, 0, 0], [0, 1/5, 0, 0], [0, 0, 1/5, 0], [0, 0, 0, 1/3]])) @ v

        x0 = np.array([0.0, 0.0, 0.0, 0.0])

        solution_actual, info_dict_pure_cg = self.solver.solve(F_callback, residual_callback, x0)

        solution_desired = spsolve(self.A, self.b)
        assert_array_almost_equal(solution_actual, solution_desired)

        # Preconditioner
        self.solver.set_config({'save_history': True,
                                'precondition': precondition
                                })

        solution_actual, info_dict_precond_cg = self.solver.solve(F_callback, residual_callback, x0)

        assert_array_almost_equal(solution_actual, solution_desired)
        self.assertTrue(info_dict_precond_cg['GMRES_iterations'] <= info_dict_pure_cg['GMRES_iterations'])

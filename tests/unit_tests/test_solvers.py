"""Test routines for solver module"""

from unittest import TestCase
from amfeti.solvers import PCPGsolver, GMRESsolver, ORTHOMINsolver
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
        self.assertTrue(info_dict_fReOrth_cg['iterations'] <= info_dict_pure_cg['iterations'])

        # Preconditioner
        self.solver.set_config({'save_history': True,
                                'full_reorthogonalization': False,
                                'precondition': precondition
                                })

        solution_actual, info_dict_precond_cg = self.solver.solve(F_callback, residual_callback, x0)

        assert_array_almost_equal(solution_actual, solution_desired)
        self.assertTrue(info_dict_precond_cg['iterations'] <= info_dict_pure_cg['iterations'])

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
        self.assertTrue(info_dict_precond_cg['iterations'] <= info_dict_pure_cg['iterations'])


class OrthominSolverTest(TestCase):
    def setUp(self):
        self.solver =ORTHOMINsolver()
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

        """ Check to ensure if the actual residual and the ORTHOMIN residual are consistent """
        NormCheck =np.linalg.norm(self.b- F_callback(solution_actual))  - info_dict_pure_cg['residual']

        if NormCheck > 1e-8:
            print('The Orthomin residual and actual residual is not consistent')

        else:

            print('Orthomin test done sucessfully! ')








        # Preconditioner
        self.solver.set_config({'save_history': True,
                                'precondition': precondition
                                })

        solution_actual, info_dict_precond_cg = self.solver.solve(F_callback, residual_callback, x0)

        assert_array_almost_equal(solution_actual, solution_desired)
        self.assertTrue(info_dict_precond_cg['iterations'] <= info_dict_pure_cg['iterations'])

    def OrthonormalCheck(self, OrthonormalCheck):

        if np.linalg.norm(OrthonormalCheck) > 1e-4:
            print('The basis is not consistent')

        else:
            print('Basis check done successfully!')

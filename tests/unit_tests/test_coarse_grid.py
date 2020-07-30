"""Test routines for coarse-grid module"""

from unittest import TestCase
from amfeti.coarse_grid import *
from ..tools import CustomDictAssertTest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix


class CoarseProblemBaseTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


class NaturalCoarseGridTest(TestCase):
    def setUp(self):
        self.coarse_grid = NaturalCoarseGrid()
        self.custom_asserter = CustomDictAssertTest()
        B_dict_test = {'sub1': {'interface1': csr_matrix(np.array([[1, 0, 0, 0, 0],
                                                                   [0, 0, -1, 0, 0]])),
                                'interface2': csr_matrix(np.array([[-1, 0, 0, 0, 0]]))},
                       'sub2': {'interface1': csr_matrix(np.array([[0, 0, -1, 0],
                                                                   [0, 1, 0, 0]]))},
                       'sub3': {'interface2': csr_matrix(np.array([[0, 0, 1]]))}}

        R_dict_test = {'sub1': csr_matrix(np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])),
                       'sub2': csr_matrix(np.array([[1], [2], [3], [4]])),
                       'sub3': csr_matrix(np.array([[0.01, 0.1],
                                                    [0.02, 0.2],
                                                    [0.03, 0.3]]))}

        f_dict_test = {'sub1': np.array([0, 6, 0.6, 0, 0]),
                       'sub2': np.array([0, 0, 0, 0]),
                       'sub3': np.array([8, 0, 0])}

        self.BR_dict = dict()
        self.RTf_dict = dict()
        for problem_id, B_dict in B_dict_test.items():
            BR_int_dict = dict()
            if R_dict_test[problem_id].size == 0:
                for interface, B in B_dict.items():
                    BR_int_dict[interface] = np.array([])
            else:
                for interface, B in B_dict.items():
                    BR_int_dict[interface] = B @ R_dict_test[problem_id]
            self.BR_dict[problem_id] = BR_int_dict
            if R_dict_test[problem_id].size == 0:
                self.RTf_dict[problem_id] = np.array([])
            else:
                self.RTf_dict[problem_id] = R_dict_test[problem_id].T @ f_dict_test[problem_id]

    @staticmethod
    def _interfacedict2vector(intdict):
        vector = None
        interface2dofmap = {'interface1': np.array([2, 0]),
                            'interface2': np.array([1])}
        for interface, value in intdict.items():
            if vector is None:
                vector = csr_matrix(np.zeros((3, value.shape[1])))
            vector[interface2dofmap[interface], :] = value
        return vector

    def tearDown(self):
        pass

    def test_set_config(self):
        config_dict = {'tolerance': 1.0e-7}

        self.coarse_grid.set_config(config_dict)

        config_dict_desired = {'solution_method': 'spsolve',
                                'tolerance': 1.0e-7}

        self.custom_asserter.assert_dict_almost_equal(self.coarse_grid._config_dict, config_dict_desired)

        config_dict = {'solution_method': 'inv',
                       'tolerance': 1.0e-7}

        self.coarse_grid.set_config(config_dict)

        self.custom_asserter.assert_dict_almost_equal(self.coarse_grid._config_dict, config_dict)

    def test_update(self):
        self.coarse_grid.update(self.BR_dict, self.RTf_dict, self._interfacedict2vector)

        G_desired = np.array([[-0.3, 2, 0, 0],
                             [-0.1, 0, 0.03, 0.3],
                             [0.1, -3, 0, 0]])

        e_desired = np.array([1.38, 0, 0.08, 0.8])

        assert_array_almost_equal(self.coarse_grid.G.todense(), G_desired)
        assert_array_almost_equal(self.coarse_grid.e, e_desired)

        self.assertEqual(self.coarse_grid._dimension_interface, 3)

    def test_solve(self):
        self.coarse_grid.update(self.BR_dict, self.RTf_dict, self._interfacedict2vector)
        solution_actual = self.coarse_grid.solve()
        solution_desired = np.array([-7.057143, 2.666667,  -4.704762])

        assert_array_almost_equal(solution_actual, solution_desired)

        input_vec = np.array([0, 0.1, 0])
        solution_actual = self.coarse_grid.solve(input_vec)
        solution_desired = np.array([0, 0, 4.0, -6.666667e-02])

        assert_array_almost_equal(solution_actual, solution_desired)

    def test_project(self):
        self.coarse_grid.update(self.BR_dict, self.RTf_dict, self._interfacedict2vector)

        input_vec = np.array([-0.3, 0.1, -0.15])

        projected_vec_actual = self.coarse_grid.project(input_vec)

        projected_vec_desired = np.array([0, 0, 0])

        assert_array_almost_equal(projected_vec_actual, projected_vec_desired)

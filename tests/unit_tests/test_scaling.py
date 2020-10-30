"""Test routines for scaling module"""

from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix
from amfeti.scaling import MultiplicityScaling
from ..tools import CustomDictAssertTest


class MultiplicityScalingTest(TestCase):
    def setUp(self):
        self.scaling_dict = {'subA': MultiplicityScaling(),
                             'subB': MultiplicityScaling(),
                             'subC': MultiplicityScaling(),
                             'subD': MultiplicityScaling(),
                             'subE': MultiplicityScaling()}

        self.B_dict = {'subA': {'interfaceAD': csr_matrix(np.array([[0, 0, 0, 1, 0],
                                                               [0, 0, 0, 0, 1]])),
                           'interfaceAE': csr_matrix(np.array([[1, 0, 0, 0, 0],
                                                               [0, 0, 0, 1, 0]])),
                           'interfaceAB': csr_matrix(np.array([[0, 0, 0, 1, 0]]))},
                  'subB': {'interfaceBD': csr_matrix(np.array([[1, 0, 0, 0],
                                                               [0, 0, 0, 1]])),
                           'interfaceBC': csr_matrix(np.array([[0, 1, 0, 0],
                                                               [0, 0, 1, 0]])),
                           'interfaceAB': csr_matrix(np.array([[0, 0, 0, -1]])),
                           'interfaceBE': csr_matrix(np.array([[0, 0, 1, 0],
                                                               [0, 0, 0, 1]]))},
                  'subC': {'interfaceBC': csr_matrix(np.array([[0, 0, 0, -1],
                                                               [0, -1, 0, 0]])),
                           'interfaceCE': csr_matrix(np.array([[1, 0, 0, 0],
                                                               [0, 1, 0, 0]]))},
                  'subD': {'interfaceAD': csr_matrix(np.array([[-1, 0, 0, 0, 0],
                                                               [0, -1, 0, 0, 0]])),
                           'interfaceBD': csr_matrix(np.array([[0, 0, 0, 0, -1],
                                                               [-1, 0, 0, 0, 0]])),
                           'interfaceDE': csr_matrix(np.array([[1, 0, 0, 0, 0]]))},
                  'subE': {'interfaceAE': csr_matrix(np.array([[0, 0, 0, -1, 0, 0],
                                                               [0, 0, 0, 0, -1, 0]])),
                           'interfaceBE': csr_matrix(np.array([[0, -1, 0, 0, 0, 0],
                                                               [0, 0, 0, 0, -1, 0]])),
                           'interfaceCE': csr_matrix(np.array([[0, 0, 0, 0, 0, -1],
                                                               [0, -1, 0, 0, 0, 0]])),
                           'interfaceDE': csr_matrix(np.array([[0, 0, 0, 0, -1, 0]]))}}

        self.custom_asserter = CustomDictAssertTest()

    def tearDown(self):
        pass

    def test_update(self):
        scaling_mat_dict_desired = {'subA': {'interfaceAD': csr_matrix(np.array([[1/4, 0],
                                                                                 [0, 1/2]])),
                                             'interfaceAE': csr_matrix(np.array([[1/2, 0],
                                                                                 [0, 1/4]])),
                                             'interfaceAB': csr_matrix(np.array([[1/4]]))},
                                    'subB': {'interfaceBD': csr_matrix(np.array([[1/2, 0],
                                                                                 [0, 1/4]])),
                                             'interfaceBC': csr_matrix(np.array([[1/2, 0],
                                                                                 [0, 1/3]])),
                                             'interfaceAB': csr_matrix(np.array([[1/4]])),
                                             'interfaceBE': csr_matrix(np.array([[1/3, 0],
                                                                                 [0, 1/4]]))},
                                    'subC': {'interfaceBC': csr_matrix(np.array([[1/2, 0],
                                                                                 [0, 1/3]])),
                                             'interfaceCE': csr_matrix(np.array([[1/2, 0],
                                                                                 [0, 1/3]]))},
                                    'subD': {'interfaceAD': csr_matrix(np.array([[1/4, 0],
                                                                                 [0, 1/2]])),
                                             'interfaceBD': csr_matrix(np.array([[1/2, 0],
                                                                                 [0, 1/4]])),
                                             'interfaceDE': csr_matrix(np.array([[1/4]]))},
                                    'subE': {'interfaceAE': csr_matrix(np.array([[1/2, 0],
                                                                                 [0, 1/4]])),
                                             'interfaceBE': csr_matrix(np.array([[1/3, 0],
                                                                                 [0, 1/4]])),
                                             'interfaceCE': csr_matrix(np.array([[1/2, 0],
                                                                                 [0, 1/3]])),
                                             'interfaceDE': csr_matrix(np.array([[1/4]]))}}

        for subs, scaling in self.scaling_dict.items():
            scaling.update(self.B_dict[subs])

            scaling_mat_desired = scaling_mat_dict_desired[subs]
            for interface, int_scaling in scaling.scaling_dict.items():
                assert_array_almost_equal(int_scaling.todense(), scaling_mat_desired[interface].todense())

    def test_apply(self):
        gap_dict = {'subA': {'interfaceAD': np.array([2, 2]),
                             'interfaceAE': np.array([2, 2]),
                             'interfaceAB': np.array([2])},
                    'subB': {'interfaceBD': np.array([2, 2]),
                             'interfaceBC': np.array([2, 2]),
                             'interfaceAB': np.array([2]),
                             'interfaceBE': np.array([2, 2])},
                    'subC': {'interfaceBC': np.array([2, 2]),
                             'interfaceCE': np.array([2, 2])},
                    'subD': {'interfaceAD': np.array([2, 2]),
                             'interfaceBD': np.array([2, 2]),
                             'interfaceDE': np.array([2])},
                    'subE': {'interfaceAE': np.array([2, 2]),
                             'interfaceBE': np.array([2, 2]),
                             'interfaceCE': np.array([2, 2]),
                             'interfaceDE': np.array([2])}}

        gap_dict_desired = {'subA': {'interfaceAD': np.array([1/2, 1]),
                                     'interfaceAE': np.array([1, 1/2]),
                                     'interfaceAB': np.array([1/2])},
                            'subB': {'interfaceBD': np.array([1, 1/2]),
                                     'interfaceBC': np.array([1, 2/3]),
                                     'interfaceAB': np.array([1/2]),
                                     'interfaceBE': np.array([2/3, 1/2])},
                            'subC': {'interfaceBC': np.array([1, 2/3]),
                                     'interfaceCE': np.array([1, 2/3])},
                            'subD': {'interfaceAD': np.array([1/2, 1]),
                                     'interfaceBD': np.array([1, 1/2]),
                                     'interfaceDE': np.array([1/2])},
                            'subE': {'interfaceAE': np.array([1, 1/2]),
                                     'interfaceBE': np.array([2/3, 1/2]),
                                     'interfaceCE': np.array([1, 2/3]),
                                     'interfaceDE': np.array([1/2])}}

        for subs, scaling in self.scaling_dict.items():
            scaling.update(self.B_dict[subs])

            gap_dict_actual = scaling.apply(gap_dict[subs])

            self.custom_asserter.assert_dict_almost_equal(gap_dict_actual, gap_dict_desired[subs])

from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import issparse

__all__ = ['CustomDictAssertTest']


class CustomDictAssertTest(TestCase):
    """
    Customized methods to test dictionaries by iterating through their items.

    Attributes
    ----------
    _recursion_counter : int
        Counter for recursive method-calls
    max_recursions : int
        Limit-number of recursion-calls
    """
    def __init__(self):
        super().__init__()
        self._recursion_counter = 0
        self.max_recursions = 10
        self.decimals = 6
        self.tolerance = 1.0e-6

    def assert_dict_almost_equal(self, dict1, dict2):
        """
        Test if the entries in two dictionaries are almost equal. Especially recommended, if dictionaries contain
        floating-point numbers.

        Parameters
        ----------
        dict1 : dict
            First dictionary for comparison
        dict2 : dict
            Second dictionary for comparison

        Return
        ------
        None
        """
        self._recursion_counter += 1
        if self._recursion_counter <= self.max_recursions:
            for key, value in dict1.items():
                if isinstance(value, dict):
                    self.assert_dict_almost_equal(value, dict2[key])
                elif isinstance(value, np.ndarray):
                    assert_array_almost_equal(value, dict2[key], self.decimals)
                elif issparse(value):
                    assert_array_almost_equal(value.todense(), dict2[key].todense(), self.decimals)
                else:
                    self.assertAlmostEqual(value, dict2[key], self.decimals, self.tolerance)
            self._recursion_counter = 0
        else:
            raise RuntimeError('Recursion stopped to avoid infinite loops')

    def assert_dict_equal(self, dict1, dict2):
        """
        Test if the entries in two dictionaries are equal. Only recommended, if no floating-point-numbers are present.

        Parameters
        ----------
        dict1 : dict
            First dictionary for comparison
        dict2 : dict
            Second dictionary for comparison

        Return
        ------
        None
        """
        self._recursion_counter += 1
        if self._recursion_counter <= self.max_recursions:
            for key, value in dict1.items():
                if isinstance(value, dict):
                    self.assert_dict_equal(value, dict2[key])
                elif isinstance(value, np.ndarray):
                    assert_array_equal(value, dict2[key])
                elif issparse(value):
                    assert_array_equal(value.todense(), dict2[key].todense())
                else:
                    self.assertEqual(value, dict2[key])
            self._recursion_counter = 0
        else:
            raise RuntimeError('Recursion stopped to avoid infinite loops')

    def assert_dict_keys_equal(self, dict1, dict2):
        """
        Test if the keys of two dictionaries are equal. The content of the dictionaries is not tested.

        Parameters
        ----------
        dict1 : dict
            First dictionary for comparison
        dict2 : dict
            Second dictionary for comparison

        Return
        ------
        None
        """
        self._recursion_counter += 1
        if self._recursion_counter <= self.max_recursions:
            for key, value in dict1.items():
                self.assertTrue(key in dict2.keys())
                if isinstance(value, dict):
                    self.assert_dict_keys_equal(value, dict2[key])

            for key, value in dict2.items():
                self.assertTrue(key in dict1.keys())
                if isinstance(value, dict):
                    self.assert_dict_keys_equal(value, dict1[key])
            self._recursion_counter = 0
        else:
            raise RuntimeError('Recursion stopped to avoid infinite loops')
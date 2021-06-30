"""Test routines for solution module"""

from unittest import TestCase
from numpy.testing import assert_array_almost_equal
import numpy as np
from amfeti.tools import *
from os.path import join, dirname


class DummyClass():
    def __init__(self):
        pass


class ToolsTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_invert_dictionary(self):
        # char and int dictionary
        dummy_dict = {'a': 1, 'b': 2, 'c': 3}
        inv_dummy_dict = {1: 'a', 2: 'b', 3: 'c'}
        return_dict = invert_dictionary(dummy_dict)
        self.assertTrue(inv_dummy_dict == return_dict)

        # string and double dictionary
        dummy_dict = {"a_element": 1.1, "b_element": 2.2, "c_element": 3.3}
        inv_dummy_dict = {1.1: "a_element", 2.2: "b_element", 3.3: "c_element"}
        return_dict = invert_dictionary(dummy_dict)
        self.assertTrue(inv_dummy_dict == return_dict)

        # list elements and tuples dictionary
        iter_list = [1, 2, 3]
        dummy_dict = {iter_list[0]: (1, 1), iter_list[1]: (2, 2), iter_list[2]: (3, 3)}
        inv_dummy_dict = {(1, 1): iter_list[0], (2, 2): iter_list[1], (3, 3): iter_list[2]}
        return_dict = invert_dictionary(dummy_dict)
        self.assertTrue(inv_dummy_dict == return_dict)

        # lists dictionary
        dummy_dict = {'A': 1, 'B': 1, 'C': 2}
        inv_dummy_dict = {1: ['A', 'B'], 2: 'C'}
        return_dict = invert_dictionary(dummy_dict)
        self.assertTrue(inv_dummy_dict == return_dict)

    def test_invert_dictionary_with_iterables(self):
        # lists dictionary
        iterable_list = ['a', 'b', 'c', 'd']
        dummy_dict = {1: iterable_list, 2: iterable_list, 3: iterable_list, 4: iterable_list}
        inv_dummy_dict = {'a': [1, 2, 3, 4], 'b': [1, 2, 3, 4], 'c': [1, 2, 3, 4], 'd': [1, 2, 3, 4]}
        return_dict = invert_dictionary_with_iterables(dummy_dict)
        # print(return_dict)
        self.assertTrue(inv_dummy_dict == return_dict)

        # np-arrays dictionary
        np_array_A = np.array([1, 2, 3])
        np_array_B = np.array([2, 4, 6])
        np_array_C = np.array([1, 3, 5])
        dummy_dict = {'A': np_array_A, 'B': np_array_B, 'C': np_array_C}
        inv_dummy_dict = {1: np.array(['A', 'C'], dtype=object), 2: np.array(['A', 'B'], dtype=object),
                          3: np.array(['A', 'C'], dtype=object), 4: np.array(['B'], dtype=object),
                          6: np.array(['B'], dtype=object), 5: np.array(['C'], dtype=object)}
        return_dict = invert_dictionary_with_iterables(dummy_dict)
        # print(return_dict)
        self.assertTrue(inv_dummy_dict.keys() == return_dict.keys())
        for k in return_dict.keys():
            self.assertTrue(inv_dummy_dict[k].all() == return_dict[k].all())

        # tuples dictionary
        dummy_tuple_1 = (0, 1)
        dummy_tuple_2 = (2, 3)
        dummy_dict = {'A': dummy_tuple_1, 'B': dummy_tuple_1, 'C': dummy_tuple_2}
        inv_dummy_dict = {0: ('A', 'B'), 1: ('A', 'B'), 2: ('C',), 3: ('C',)}
        return_dict = invert_dictionary_with_iterables(dummy_dict)
        # print(return_dict)
        self.assertTrue(inv_dummy_dict == return_dict)

        # string dictionary
        dummy_string_1 = "ab"
        dummy_string_2 = "cd"
        dummy_dict = {1: dummy_string_1, 2: dummy_string_2}
        inv_dummy_dict = {'a': 1, 'b': 1, 'c': 2, 'd': 2}
        return_dict = invert_dictionary_with_iterables(dummy_dict)
        # print(return_dict)
        self.assertTrue(inv_dummy_dict == return_dict)

        # sets dictionary
        iterable_set = set(iterable_list)
        dummy_dict = {1: iterable_set, 2: iterable_set, 3: iterable_set, 4: iterable_set}
        inv_dummy_dict = {'d': 4, 'c': 4, 'a': 4, 'b': 4}
        return_dict = invert_dictionary_with_iterables(dummy_dict)
        # print("set_dict",return_dict)
        self.assertTrue(inv_dummy_dict == return_dict)

        # error expected
        dummy_object = DummyClass()
        dummy_dict = {iterable_list[0]: dummy_object}
        with self.assertRaises(ValueError):
            invert_dictionary_with_iterables(dummy_dict)

    def test_amfeti_dir(self):
        filename = 'testname'
        absolute_path = dirname(dirname(dirname(__file__)))
        absolute_path = join(absolute_path, 'amfeti')
        expected_name = join(absolute_path, filename.lstrip('/'))
        return_name = amfeti_dir(filename)
        self.assertTrue(expected_name == return_name)

"""Test routines for solution module"""

from unittest import TestCase
from numpy.testing import assert_array_almost_equal
import numpy as np
from amfeti.solution import SolutionBase, StandardSolution


class StandardSolutionTest(TestCase):
    def setUp(self):
        self.solution_base = SolutionBase()
        self.standard_solution = StandardSolution()

    def tearDown(self):
        pass

    def test_default_attributes(self):
        self.assertTrue(self.solution_base._dual_solution == None)
        self.assertTrue(self.solution_base._alpha_solution == None)
        self.assertTrue(self.solution_base._local_problems_dict == dict())
        self.assertTrue(self.solution_base.solver_information == dict())

        self.assertTrue(self.standard_solution._dual_solution == None)
        self.assertTrue(self.standard_solution._alpha_solution == None)
        self.assertTrue(self.standard_solution._local_problems_dict == dict())
        self.assertTrue(self.standard_solution.solver_information == dict())

    def test_updated_attributes(self):
        dummy_numpy_array = np.random.randint(1, 9, (1, 3))[0]
        dummy_alpha_numpy_array = np.random.randint(1, 9, (1, 3))[0]

        self.solution_base.update({"1":1, "2":2, "3":3}, dummy_numpy_array, dummy_alpha_numpy_array, {"0":0, "1":1, "2":2})
        self.assertTrue(self.solution_base._dual_solution.all() == dummy_numpy_array.all())
        self.assertTrue(self.solution_base._alpha_solution.all() == dummy_alpha_numpy_array.all())
        self.assertTrue(self.solution_base._local_problems_dict == {"1":1, "2":2, "3":3})
        self.assertTrue(self.solution_base.solver_information == {"0":0, "1":1, "2":2})

        self.standard_solution.update({"1":1, "2":2, "3":3}, dummy_numpy_array, dummy_alpha_numpy_array, {"0":0, "1":1, "2":2})
        self.assertTrue(self.standard_solution._dual_solution.all() == dummy_numpy_array.all())
        self.assertTrue(self.standard_solution._alpha_solution.all() == dummy_alpha_numpy_array.all())
        self.assertTrue(self.standard_solution._local_problems_dict == {"1":1, "2":2, "3":3})
        self.assertTrue(self.standard_solution.solver_information == {"0":0, "1":1, "2":2})

    def test_properties(self):
        dummy_numpy_array = np.random.randint(1, 9, (1, 3))[0]
        dummy_alpha_numpy_array = np.random.randint(1, 9, (1, 3))[0]

        self.standard_solution.update({"1":1, "2":2, "3":3}, dummy_numpy_array, dummy_alpha_numpy_array, {"0":0, "1":1, "2":2})
        self.assertTrue(self.standard_solution.local_problems == {"1":1, "2":2, "3":3})
        self.assertTrue(self.standard_solution.dual_solution.all() == dummy_numpy_array.all())
        self.assertTrue(self.standard_solution.kernel_modes.all() == dummy_alpha_numpy_array.all())

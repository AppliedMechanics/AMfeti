"""Test routines for linalg-datatypes module"""

from unittest import TestCase

import scipy

from amfeti.linalg.datatypes import Pseudoinverse, Matrix
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix
from copy import copy


class PseudoinverseTest(TestCase):
    def setUp(self):
        self.pinv = Pseudoinverse()

    def tearDown(self):
        pass

    def test_init(self):
        self.assertTrue(self.pinv._method == 'svd')
        self.assertTrue(self.pinv._pinv == None)
        self.assertTrue(self.pinv._pinv_callback == None)
        self.assertTrue(self.pinv._tolerance == 1.0E-8)
        self.assertTrue(self.pinv._dimension == 0)
        self.assertTrue(isinstance(self.pinv._null_space, np.ndarray))
        assert_array_almost_equal(self.pinv._null_space, np.array(([])))

    def test_properties(self):
        assert_array_almost_equal(self.pinv._null_space, self.pinv.null_space.all())
        expected_matrix = csr_matrix((self.pinv._dimension, self.pinv._dimension))
        eye_mat = scipy.sparse.eye(self.pinv._dimension)
        for col in range(self.pinv._dimension):
            expected_matrix[col, :] = self.pinv._pinv_callback(eye_mat[col, :])
        self.assertTrue((expected_matrix != self.pinv.matrix).nnz == 0)
        assert_array_almost_equal(expected_matrix.todense(), eye_mat.todense())

        self.pinv._pinv = None
        self.pinv._dimension = 3
        self.pinv._pinv_callback = lambda x: x
        expected_matrix = copy(self.pinv.matrix)
        assert_array_almost_equal(self.pinv.matrix.todense(), expected_matrix.todense())

    def test_set_solver(self):
        self.pinv.set_solver('svd')
        self.assertEqual(self.pinv._method, 'svd')
        self.pinv.set_solver('qr')
        self.assertEqual(self.pinv._method, 'qr')

        with self.assertRaises(Exception):
            self.pinv.set_solver("nonexistent_method", 1.0E-8)

    def test_compute(self):
        K = np.array([[5, -5, 0], [-5, 10, -5], [0, -5, 5]])
        K_inv_desired = np.linalg.pinv(K)

        self.pinv.set_solver("svd")
        self.pinv.compute(K)
        assert_array_almost_equal(self.pinv.matrix, K_inv_desired)
        assert_array_almost_equal((self.pinv.null_space.T @ K), np.zeros_like(self.pinv.null_space.T))
        assert_array_almost_equal((K @ self.pinv.matrix @ K), K)

        self.pinv.set_solver("qr")
        self.pinv.compute(csr_matrix(K))
        assert_array_almost_equal((self.pinv.null_space.T @ K), np.zeros_like(self.pinv.null_space.T))
        assert_array_almost_equal((K @ self.pinv.matrix @ K), K)

        K = np.array([[5, -5, 1], [-5, 10, -5], [0, -5, 5]])

        self.pinv.set_solver("svd")
        self.pinv.compute(K)
        assert_array_almost_equal((K @ self.pinv.matrix @ K), K)

        self.pinv.set_solver("qr")
        self.pinv.compute(csr_matrix(K))
        assert_array_almost_equal((K @ self.pinv.matrix @ K), K)

        K = np.array([[5, -5, 1], [-5, 10, -5], [0, -5, 5]])
        K = np.concatenate((K, np.array([2*K[2,:]-K[1,:]])), axis=0)

        self.pinv.set_solver("svd")
        self.pinv.compute(K)
        assert_array_almost_equal((K @ self.pinv.matrix @ K), K)

        self.pinv.set_solver("qr")
        self.pinv.compute(csr_matrix(K))
        assert_array_almost_equal((K @ self.pinv.matrix @ K), K)

        with self.assertRaises(ValueError):
            self.pinv._method = 'dummy_method'
            self.pinv.compute(K)

    def test_apply(self):
        self.pinv = Pseudoinverse()
        K = np.array([[-1, 3, 2], [1, 1, 0], [1, 1, 0]])
        self.pinv.set_solver("svd")
        self.pinv.compute(K)
        f = np.array([1, 2, 3])
        u = self.pinv.apply(f)
        assert_array_almost_equal(u, np.array([1.5, 1, -0.25]))
        f = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        u = self.pinv.apply(f)
        assert_array_almost_equal(u, np.array([[3.5, 4., 4.5],
                                               [2., 2.5, 3.],
                                               [-0.75, -0.75, -0.75]]))


class MatrixTest(TestCase):
    def setUp(self):
        self.K_init = np.random.randint(0, 10, (3, 3))
        self.matrix = Matrix(self.K_init)

    def tearDown(self):
        pass

    def test_init(self):
        assert_array_almost_equal(self.matrix._data, self.K_init)
        self.assertTrue(isinstance(self.matrix._pseudoinverse, Pseudoinverse))
        self.assertFalse(self.matrix._inverse_computed)

    def test_properties(self):
        self.assertEqual(self.matrix.matrix.all(), self.K_init.all())
        self.assertEqual(self.matrix.shape, self.K_init.shape)
        self.assertEqual(self.matrix.trace, np.trace(self.K_init))
        self.assertEqual(self.matrix.det, np.linalg.det(self.K_init))
        assert_array_almost_equal(self.matrix.eigenvalues, np.flip(np.sort(np.linalg.eig(self.K_init)[0])))
        local_pinv = Pseudoinverse('svd', 1.0E-8)
        local_pinv.compute(self.K_init)
        assert_array_almost_equal(self.matrix.kernel, local_pinv.null_space)
        self.assertTrue(self.matrix._inverse_computed)
        self.matrix = Matrix(
            self.K_init)  # re-initializing the matrix so that all lines of the inverse property will be covered
        assert_array_almost_equal(self.matrix.inverse, local_pinv.matrix)

    def test_update(self):
        new_K = np.eye(3)
        self.matrix.update(new_K)
        self.assertFalse(self.matrix._inverse_computed)
        assert_array_almost_equal(self.matrix._data, new_K)

    def test_dot(self):
        self.matrix.update(np.eye(3))
        assert_array_almost_equal(self.matrix.dot(np.array([1, 2, 3])), np.array([1, 2, 3]))

    def test_apply_inverse(self):
        self.matrix.update(np.eye(3))
        assert_array_almost_equal(self.matrix.apply_inverse(np.array([1, 2, 3])), np.array([1, 2, 3]))

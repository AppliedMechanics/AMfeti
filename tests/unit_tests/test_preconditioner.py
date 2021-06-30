from unittest import TestCase
from numpy.testing import assert_array_equal, assert_allclose
from amfeti.preconditioners import *
from amfeti.linalg import Matrix
import numpy as np
from scipy.sparse import csr_matrix


def create5dofExample():
    K = Matrix(csr_matrix(np.array([[2, -3, 0, 0, 0],
                                    [-1, 4, -3, 0, 0],
                                    [0, -1, 4, -3, 0],
                                    [0, 0, -1, 4, -3],
                                    [0, 0, 0, -1, 2]])))

    interface_dofs = np.array([0, 1, 4])

    return K, interface_dofs


class PreconditionerBaseTest(TestCase):
    def setUp(self):
        self.K_local, self.interface_dofs = create5dofExample()

    def tearDown(self):
        pass

    def test_identify_interior_dofs(self):
        preconditioner_test = PreconditionerBase()
        preconditioner_test.update(self.K_local, self.interface_dofs)
        interior_dofs_desired = np.array([2, 3])
        interface_dofs_desired = np.array([0, 1, 4])

        assert_array_equal(preconditioner_test.interior_dofs, interior_dofs_desired)
        assert_array_equal(preconditioner_test.interface_dofs, interface_dofs_desired)

    def test_update(self):
        preconditioner_test = PreconditionerBase()
        preconditioner_test.update(self.K_local, self.interface_dofs)

        newK = Matrix(csr_matrix(
            np.array([[4, -3, 0, 0], [-1, 4, -3, 0], [0, -1, 4, -3], [0, 0, -1, 2]])))
        new_interface_dofs = np.array([0, 3])

        preconditioner_test.update(newK, new_interface_dofs)

        assert_array_equal(preconditioner_test.K.matrix.todense(), newK.matrix.todense())
        assert_array_equal(preconditioner_test.interface_dofs, new_interface_dofs)


class LumpedPreconditionerTest(TestCase):
    def setUp(self):
        self.K_local, self.interface_dofs = create5dofExample()
        self.preconditioner_test = LumpedPreconditioner()
        self.preconditioner_test.update(self.K_local, self.interface_dofs)

    def tearDown(self):
        pass

    def test_set_Q(self):
        Q_desired = np.array([[2, -3, 0, 0, 0],
                              [-1, 4, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 2]])
        assert_array_equal(self.preconditioner_test.Q.todense(), Q_desired)

    def test_apply(self):
        vector = np.array([4, -0.5, 0, 0, 1.3])

        prec_vector_actual = self.preconditioner_test.apply(vector)
        prec_vector_desired = np.array([9.5, -6, 0, 0, 2.6])

        assert_array_equal(prec_vector_actual, prec_vector_desired)


class DirichletPreconditionerTest(TestCase):
    def setUp(self):
        self.K_local, self.interface_dofs = create5dofExample()
        self.preconditioner_test = DirichletPreconditioner()
        self.preconditioner_test.update(self.K_local, self.interface_dofs)

    def tearDown(self):
        pass

    def test_set_Q(self):
        Q_desired = np.array([[2, -3, 0, 0, 0],
                              [-1, 3.07692308, 0, 0, -2.07692308],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, -0.07692308, 0, 0, 1.07692308]])
        assert_allclose(self.preconditioner_test.Q.todense(), Q_desired)

    def test_apply(self):
        vector = np.array([4, -0.5, 0, 0, 1.3])

        prec_vector_actual = self.preconditioner_test.apply(vector)
        prec_vector_desired = np.array([9.5, -8.23846154, 0, 0, 1.43846154])

        assert_allclose(prec_vector_actual, prec_vector_desired)
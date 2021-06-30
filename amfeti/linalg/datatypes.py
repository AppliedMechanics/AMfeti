#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
import numpy as np
from scipy.sparse import csr_matrix
from .tools import pinv_and_null_space_svd, qrfullsps


__all__ = ['Pseudoinverse',
           'Matrix']


class Pseudoinverse:
    """
    This class provides functionality to solve singular linear systems

    Ku = f

    where K is singular, by building the null-space  R and the pseudoinverse. There are several options to build both
    available. Then the general solution is

    u = K_pinv * f + R * alpha

    Attributes
    ----------
    _pinv : ndarray
        pseudoinverse-matrix
    _pinv_callback : callable
        pseudoinverse-callback
    _null_space : ndarray
        nullspace of the singular system
    _method : str
        solver, that is used to compute the pseudoinverse and null-space
    _tolerance : float
        tolerance for computing the pseudoinverse and null-space
    _dimension : int
        dimension of linear system
    """
    def __init__(self, method='svd', tolerance=1.0E-8):
        """
        Parameters
        ----------
        method : str
            solver, that is used to compute the pseudoinverse and null-space
        tolerance : float
            tolerance for computing the pseudoinverse and null-space

        Returns
        -------
        None
        """
        self._method = None
        self._pinv = None
        self._pinv_callback = None
        self._null_space = np.array([])
        self._tolerance = None
        self._dimension = 0
        self.set_solver(method, tolerance)

    @property
    def null_space(self):
        """
        Null-space or kernel, that belongs to the pseudoinverse
        """
        return self._null_space

    @property
    def matrix(self):
        """
        Pseudoinverse in matrix-form
        """
        if self._pinv is None:
            self._pinv = csr_matrix((self._dimension, self._dimension))
            # eye_mat = eye(self._dimension)
            eye_mat = np.eye(self._dimension)
            for col in range(self._dimension):
                self._pinv[col, :] = self._pinv_callback(eye_mat[col, :])

        return self._pinv

    def set_solver(self, method, tolerance=1.0E-8):
        """
        This method sets the chosen method and its tolerance to compute the pseudoinverse and null-space.

        Parameters
        ----------
        method : str
            solver, that is used to compute the pseudoinverse and null-space
        tolerance : float
            tolerance for computing the pseudoinverse and null-space

        Returns
        -------
        None
        """
        list_of_solvers = ['svd', 'qr']
        if method not in list_of_solvers:
            raise ValueError('Selected solution method not available, please select one from the following list: ' +
                             str(list_of_solvers))
        self._method = method
        self._tolerance = tolerance

    def compute(self, K):
        """
        This method computes the pseudoinverse and null-space with the predefined solver.

        Parameters
        ----------
        K : ndarray
            linear semi-definite linear matrix

        Returns
        -------
        None
        """
        self._pinv = None
        if self._method == 'qr':
            K_inv, R = qrfullsps(K)
            self._pinv = K_inv
            self._pinv_callback = np.array(K_inv).dot

        elif self._method == 'svd':
            K_inv, R = pinv_and_null_space_svd(K, tol=self._tolerance)
            self._pinv = K_inv
            self._pinv_callback = np.array(K_inv).dot

        else:
            raise ValueError('Solver %s not implement. Check list_of_solvers.' % self._method)

        self._dimension = K.shape[0]

        if R.size == 0:
            self._null_space = np.array([])
        else:
            self._null_space = R

    def apply(self, f):
        """
        Method to apply a right-hand-side vector to the pseudoinverse and solve the system.

        Parameters
        ----------
        f : ndarray
            right-hand-side vector

        Returns
        -------
        u : ndarray
            partial solution of semi-definite system
        """
        if f.ndim == 1:
            u = self._pinv_callback(f)
        else:
            u = self.matrix @ f
        return u


class Matrix:
    """
    Wrapper class for linear system matrices

    Attributes
    ----------
    _data : ndarray
        actual matrix-content
    _pseudoinverse : Pseudoinverse
        inverse and null-space of the given matrix
    _inverse_computed : bool
        flag, that controls the computation of the inverse for efficiency reasons
    """
    def __init__(self, K, pseudoinverse_method='svd', pseudoinverse_tol=1.0E-8):
        """
        Parameters
        ----------
        K : ndarray
            linear system matrix

        pseudoinverse_method : str
            method to compute the pseudoinverse and null-space

        pseudoinverse_tol : float
            tolerance for the pseudoinverse-computation

        Returns
        -------
        None
        """
        self._data = K
        self._pseudoinverse = Pseudoinverse(pseudoinverse_method, pseudoinverse_tol)
        self._inverse_computed = False

    @property
    def matrix(self):
        """
        matrix-content
        """
        return self._data

    @property
    def shape(self):
        """
        dimensional shape of the matrix
        """
        return self._data.shape

    @property
    def trace(self):
        """
        trace of the matrix, i.e. sum of the diagonal entries
        """
        return self._data.diagonal().sum()

    @property
    def det(self):
        """
        determinant of the matrix
        """
        return np.linalg.det(self._data)

    @property
    def eigenvalues(self):
        """
        eigenvalues of the matrix
        """
        w, v = np.linalg.eig(self._data)
        return np.sort(w)[::-1]

    @property
    def kernel(self):
        """
        kernel or null-space of the matrix
        """
        if not self._inverse_computed:
            self._pseudoinverse.compute(self._data)
            self._inverse_computed = True

        return self._pseudoinverse.null_space

    @property
    def inverse(self):
        """
        inverse of the matrix, that can also be a pseudoinverse if matrix is singular
        """
        if not self._inverse_computed:
            self._pseudoinverse.compute(self._data)
            self._inverse_computed = True

        return self._pseudoinverse.matrix

    def update(self, K):
        """
        Updating-method for the matrix-content. This also resets the trigger for inverting the matrix.

        Parameters
        ----------
        K : ndarray
            linear system matrix

        Returns
        -------
        None
        """
        self._data = K
        self._inverse_computed = False

    def dot(self, x):
        """
        Applies a matrix-vector or matrix-matrix product and returns the result

        Parameters
        ----------
        x : ndarray
            vector or matrix, that is multiplied with this matrix

        Returns
        -------
        Kdotx : ndarray
            resulting vector or matrix
        """
        return self._data @ x

    def apply_inverse(self, b):
        """
        Multiplies this matrix' inverse with a given vector or matrix

        Parameters
        ----------
        b : ndarray
            vector or matrix, that is multiplied with this matrix

        Returns
        -------
        Kinvb : ndarray
            resulting vector or matrix
        """
        if not self._inverse_computed:
            self._pseudoinverse.compute(self._data)
            self._inverse_computed = True

        return self._pseudoinverse.apply(b)

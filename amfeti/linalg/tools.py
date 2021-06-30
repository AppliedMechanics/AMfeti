#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

from scipy.sparse.linalg import spsolve, spsolve_triangular
from scipy.sparse import csr_matrix, lil_matrix, eye, diags, issparse, vstack, hstack
from scipy.linalg import orth
import logging
import numpy as np
from numpy.linalg import cholesky
from copy import copy


__all__ = ['cal_schur_complement',
           'pinv_and_null_space_svd',
           'qrfullsps']


def cal_schur_complement(K_bi, K_ii, K_ib, K_bb):
    try:
        lu_Kib = spsolve(K_ii, K_ib)

        return K_bb - K_bi @ lu_Kib
    except MemoryError:
        logger = logging.getLogger(__name__)
        logger.error('Memory error during Schur-complement calculations')
        raise MemoryError('Memory error during Schur-complement calculations')
    except Exception as e:
        raise Exception('Exception occurs during Schur-complement calculations')


def qrfullsps(K):
    """
    Full QR-factorization for a sparse potentially singular mxn matrix, with m>=n.

    Parameters
    ----------
    K : ndarray
        matrix, that is to be inverted

    Returns
    -------
    Kinv : ndarray
        pseudoinverse of K
    R : ndarray
        left nullspace of K
    """
    R = copy(K)
    Q = csr_matrix(eye(K.shape[0]))
    for j in range(K.shape[1]):
        G = None
        for k in range(K.shape[0]-1, j, -1):
            vec_norm = np.linalg.norm(np.array([R[j, j], R[k, j]]))
            if not np.isclose(vec_norm, 0):
                c_j = R[j, j] / vec_norm
                s_j = R[k, j] / vec_norm
                G_help = csr_matrix(eye(K.shape[0]))
                G_help[j, j] = c_j
                G_help[k, j] = -s_j
                G_help[j, k] = s_j
                G_help[k, k] = c_j
                R = G_help @ R
                if G is None:
                    G = copy(G_help)
                else:
                    G = G_help @ G
        if G is not None:
            Q = Q @ G.T
    Q = -Q
    R = -R

    r_row_sum = np.sum(np.abs(R), axis=1)
    tol = 1.0e-12
    rank = K.shape[0] - len(np.where(r_row_sum < tol)[0])

    Q1 = Q[:, :rank]
    Q2 = Q[:, rank:]

    R1 = R[:rank, :rank]
    R2 = R[:rank, :rank]

    # Inverting R1 by back substitution
    if R1.shape[0] == R1.shape[1]:
        R_inv = csr_matrix(R1.shape)
        backward_substitution_failed = False
        for row in np.arange(R1.shape[0]-1, -1, -1):
            if row is not R1.shape[0]-1:
                for col in np.arange(row+1, R1.shape[1]):
                    R_inv[row, :] -= R1[row, col] * R_inv[col, :]
            R_inv[row, row] = 1
            if np.isclose(R1[row, row], 0):
                backward_substitution_failed = True
                break
            else:
                R_inv[row, :] = R_inv[row, :] / R1[row, row]
    else:
        backward_substitution_failed = True
    if backward_substitution_failed:
        R_inv = csr_matrix(np.linalg.pinv(R.todense()))

    K_inv = R_inv @ Q1.T
    if K_inv.shape[0] < K.shape[1]:
        K_inv = vstack((K_inv, lil_matrix((K.shape[1]-K_inv.shape[0], K.shape[1]))))
    if K_inv.shape[1] < K.shape[0]:
        K_inv = hstack((K_inv, lil_matrix((K_inv.shape[0], K.shape[0]-K_inv.shape[1]))))

    return K_inv, Q2


def pinv_and_null_space_svd(K, tol=1.0E-12):
    """
    calculate pseudoinverve and nullspace using SVD method

    Parameters
    ----------
    K : ndarray
        matrix, that is to be inverted
    tol : float
        tolerance for the SVD

    Returns
    -------
    Kinv : ndarray
        pseudoinverse of K
    R : ndarray
        left nullspace of K
    """
    if issparse(K):
        K = K.todense()

    U, vals, Vt = np.linalg.svd(K)

    inv_vals = np.array([])
    zero_idxs = np.array([], dtype=np.int)
    nonzero_idxs = np.array([], dtype=np.int)
    for idx, val in enumerate(vals):
        if val / vals[0] > tol:
            inv_vals = np.append(inv_vals, 1.0 / val)
            nonzero_idxs = np.append(nonzero_idxs, idx)
        else:
            zero_idxs = np.append(zero_idxs, idx)

    invvals_diag = diags(inv_vals)
    if invvals_diag.shape[1] < U.shape[1]:
        U_inv = U[np.ix_(np.arange(U.shape[0]), nonzero_idxs)]
    else:
        U_inv = U
    if invvals_diag.shape[0] < Vt.T.shape[1]:
        Vt_null = Vt[np.ix_(zero_idxs, np.arange(Vt.shape[1]))]
        Vt_null = Vt_null.T
        U_null = U[np.ix_(np.arange(U.shape[0]), zero_idxs)]
        Vt_inv = Vt[np.ix_(nonzero_idxs, np.arange(Vt.shape[1]))]
    else:
        Vt_null = np.array([])
        U_null = np.array([])
        Vt_inv = Vt

    Kinv = Vt_inv.T @ invvals_diag @ U_inv.T

    R = U_null

    return Kinv, R


def _calc_null_space_of_upper_trig_matrix(U, idf=None, orthonormal=True):
    ''' This function computes the null space of
    a Upper Triangular matrix which can be a singular
    matrix

    argument
        U : np.matrix
            Upper triangular matrix
        idf: list
            index to fixed if the matrix is singular
        orthonormal : Boolean, Default =True
            return a orthonormal bases

    return
        R : np.matrix
            null space of U

    '''

    # finding the null space pivots
    n, n = U.shape
    rank_null = len(idf)
    all = set(range(n))
    idp = list(all - set(idf))
    R = np.zeros([n, rank_null])
    if rank_null > 0:
        U_lil = U.tolil()

        Ur = U_lil[:, idf].A
        Ur[idf, :] = -np.eye(rank_null, rank_null)

        U_lil[idf, :] = 0.0
        U_lil[:, idf] = 0.0
        U_lil[idf, idf] = 1.0
        U_csr = U_lil.tocsr()
        R = -spsolve_triangular(U_csr, Ur, lower=False)

        # free space im memory
        del U_csr
        del U_lil
        del Ur

        if orthonormal:
            R = orth(R)

    return R


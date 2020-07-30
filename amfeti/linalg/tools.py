#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

from scipy.sparse.linalg import spsolve, splu, spsolve_triangular
from scipy.sparse import csc_matrix, diags, lil_matrix, issparse
from scipy.linalg import orth
import logging
import numpy as np


__all__ = ['cal_schur_complement',
           'splusps',
           'cholsps']


def cal_schur_complement(K_bi, K_ii, K_ib, K_bb):
    try:
        lu_Kib = spsolve(K_ii.data, K_ib.data)

        return K_bb.data - K_bi.dot(lu_Kib)
    except MemoryError:
        logger = logging.getLogger(__name__)
        logger.error('Memory error during Schur-complement calculations')
        raise MemoryError('Memory error during Schur-complement calculations')
    except Exception as e:
        raise Exception('Exception occurs during Schur-complement calculations')


def splusps(A, tol=1.0e-6):
    ''' This method return the upper traingular matrix based on superLU of A.
    This function works for positive semi-definite matrix.
    This functions also return the null space of the matrix A.
    Input:

        A -> positive semi-definite matrix
        tol -> tolerance for small pivots

    Ouputs:
        U -> upper triangule of Cholesky decomposition
        idp -> list of non-zero pivot rows
        R -> matrix with bases of the null space
    '''
    [n, m] = np.shape(A)

    if n != m:
        print('Matrix is not square')
        return

    idp = []  # id of non-zero pivot columns
    idf = []  # id of zero pivot columns

    if not isinstance(A, csc_matrix):
        A = csc_matrix(A)

    A_diag = A.diagonal()
    Atrace = A_diag.sum()
    avg_diag_A = A_diag / Atrace

    try:
        # apply small perturbation in diagonal
        lu = splu(A, options={'DiagPivotThresh': 0.0, 'SymmetricMode': True})
        logging.info('Standard SuperLU.')
    except:
        B = A.tolil()
        B += 1.0E-15 * avg_diag_A * diags(np.ones(n))
        A = B.tocsc()
        del B
        lu = splu(A, options={'DiagPivotThresh': 0.0, 'SymmetricMode': True})
        logging.info('Perturbed SuperLU.')

    U = lu.U
    Pc = lil_matrix((n, n))
    Pc[np.arange(n), lu.perm_c] = 1

    U_diag = U.diagonal()
    Utrace = U_diag.sum()
    diag_U = U_diag / Utrace

    idf = np.where(abs(diag_U) < tol)[0].tolist()

    if len(idf) > 0:
        R = _calc_null_space_of_upper_trig_matrix(U, idf)
        R = Pc.dot(R)
    else:
        R = np.array([])

    return lu, idf, R


def cholsps(A, tol=1.0e-8):
    ''' This method return the upper traingular matrix of cholesky decomposition of A.
    This function works for positive semi-definite matrix.
    This functions also return the null space of the matrix A.
    Input:

        A -> positive semi-definite matrix
        tol -> tolerance for small pivots

    Ouputs:
        U -> upper triangule of Cholesky decomposition
        idp -> list of non-zero pivot rows
        R -> matrix with bases of the null space

    '''
    [n, m] = np.shape(A)

    if n != m:
        print('Matrix is not square')
        return

    L = np.zeros([n, n])
    # L = sparse.lil_matrix((n,n),dtype=float)
    # A = sparse.csr_matrix(A)
    idp = []  # id of non-zero pivot columns
    idf = []  # id of zero pivot columns

    if issparse(A):
        Atrace = np.trace(A.A)
        A = A.todense()
    else:
        Atrace = np.trace(A)

    tolA = tol * Atrace / n

    for i in range(n):
        Li = L[i, :]
        Lii = A[i, i] - np.dot(Li, Li)
        if Lii > tolA:
            L[i, i] = np.sqrt(Lii)
            idp.append(i)
        elif abs(Lii) < tolA:
            L[i, i] = 0.0
            idf.append(i)

        elif Lii < -tolA:
            logging.debug('Matrix is not positive semi-definite.' + \
                          'Given tolerance = %2.5e' % tol)
            return L, [], None

        for j in range(i + 1, n):
            if L[i, i] > tolA:
                L[j, i] = (A[j, i] - np.dot(L[i, :], L[j, :])) / L[i, i]

    # finding the null space
    rank = len(idp)
    rank_null = n - rank

    U = L.T
    R = None
    if rank_null > 0:
        Im = np.eye(rank_null)

        # Applying permutation to get an echelon form

        PA = np.zeros([n, n])
        PA[:rank, :] = U[idp, :]
        PA[rank:, :] = U[idf, :]

        # creating block matrix
        A11 = np.zeros([rank, rank])
        A12 = np.zeros([rank, rank_null])

        A11 = PA[:rank, idp]
        A12 = PA[:rank, idf]

        R11 = np.zeros([rank, rank_null])
        R = np.zeros([n, rank_null])

        # backward substitution
        for i in range(rank_null):
            for j in range(rank - 1, -1, -1):
                if j == rank - 1:
                    R11[j, i] = -A12[j, i] / A11[j, j]
                else:
                    R11[j, i] = (-A12[j, i] - np.dot(R11[j + 1:rank, i], A11[j, j + 1:rank])) / A11[j, j]

        # back to the original bases
        R[idf, :] = Im
        R[idp, :] = R11

        logging.debug('Null space size = %i' % len(idf))

    return U, idf, R


def _calc_null_space_of_upper_trig_matrix(U, idf=None, orthonormal=True):
    ''' This function computer the Null space of
    a Upper Triangule matrix which is can be a singular
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


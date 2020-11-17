#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Preconditioning module for AMfeti and linearized systems, that have a K-matrix-like structure similar to the linear
static local problem.
"""
from scipy.sparse import csr_matrix
from amfeti.linalg import cal_schur_complement, Matrix
from .preconditioner_base import PreconditionerBase
import numpy as np


all = ['LumpedPreconditioner',
       'SuperLumpedPreconditioner',
       'DirichletPreconditioner',
       'LumpedDirichletPreconditioner']


class LumpedPreconditioner(PreconditionerBase):
    """
    Lumped preconditioner, that uses the stiffnesses on the boundary only
    """
    def __init__(self):
        super().__init__()

    @property
    def K_bb(self):
        """
        Part of the operator matrix on the interface
        """
        if self.interior_dofs is None:
            self.interior_dofs = self._identify_interior_dofs(self.interface_dofs)

        return Matrix(self.K.data[np.ix_(self.interface_dofs, self.interface_dofs)])

    def _set_Q(self):
        self.Q = csr_matrix(np.zeros(self.K.data.shape))
        self.Q[np.ix_(self.interface_dofs, self.interface_dofs)] = self.K_bb.data


class SuperLumpedPreconditioner(LumpedPreconditioner):
    """
    Similar preconditioner as the lumped preconditioner, but it utilizes the diagonal entries of the lumped
    preconditioner only.
    """
    def __init__(self):
        super().__init__()

    def _set_Q(self):
        self.Q = csr_matrix(np.zeros(self.K.data.shape))
        self.Q[np.ix_(self.interface_dofs, self.interface_dofs)] = np.diag(self.K_bb.data)


class DirichletPreconditioner(LumpedPreconditioner):
    """
    Dirichlet preconditioner, that uses a Schur-complement to estimate interface forces. Hence the most accurate,
    but expensive preconditioner
    """
    def __init__(self):
        super().__init__()

    @property
    def K_ii(self):
        """
        Interior part of the operator matrix, which is not related to any boundary-dof
        """
        if self.interior_dofs is None:
            self.interior_dofs = self._identify_interior_dofs(self.interface_dofs)

        return Matrix(self.K.data[np.ix_(self.interior_dofs, self.interior_dofs)])

    @property
    def K_ib(self):
        """
        Connecting part of the operator matrix between interior and interface-dofs
        """
        if self.interior_dofs is None:
            self.interior_dofs = self._identify_interior_dofs(self.interface_dofs)

        return Matrix(self.K.data[np.ix_(self.interior_dofs, self.interface_dofs)])

    @property
    def K_bi(self):
        """
        Connecting part of the operator matrix between interface and interior-dofs
        """
        if self.interior_dofs is None:
            self.interior_dofs = self._identify_interior_dofs(self.interface_dofs)

        return Matrix(self.K.data[np.ix_(self.interface_dofs, self.interior_dofs)])

    def schur_complement(self):
        """
        Caller for a Schur-complement calculation
        """
        return cal_schur_complement(self.K_bi, self.K_ii, self.K_ib, self.K_bb)

    def _set_Q(self):
        self.Q = csr_matrix(np.zeros(self.K.data.shape),dtype=complex)
        S = self.schur_complement()

        self.Q[np.ix_(self.interface_dofs, self.interface_dofs)] = S


class LumpedDirichletPreconditioner(DirichletPreconditioner):
    """
    Lumped Dirichlet preconditioner, that uses a Schur-complement, but with die diagonal entries of the interior
    stiffnesses.
    """
    def __init__(self):
        super().__init__()

    def schur_complement(self):
        """
        Caller for a Schur-complement calculation with an extraction of the lumped interior operator matrix
        """
        K_ii_diag = Matrix(np.diag(self.K_ii.data))
        return cal_schur_complement(Matrix(self.K_ib.data.T), K_ii_diag, self.K_ib, self.K_bb)

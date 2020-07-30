#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Basic preconditioning module for AMfeti
"""
import numpy as np
from scipy.sparse import csr_matrix


__all__ = ['PreconditionerBase']


class PreconditionerBase:
    """
    Base-class for local preconditioners

    Attributes
    ----------
    K : ndarray
        local linear operand, e.g. stiffness
    Q : ndarray
        preconditioning-matrix
    interior_dofs : ndarray
        local ids of dofs, that are not part of the subdomain's boundaries
    interface_dofs : ndarray
        local ids of dofs, that are part of the subdomain's boundaries
    """
    def __init__(self):
        """
        Parameters
        ----------
        None
        """
        self.K = None
        self.Q = None
        self.Q_exp = None
        self.interior_dofs = None
        self.interface_dofs = None

    def update(self, K, interface_dofs):
        """
        Updater for the local linear operation matrix and interface_dofs in case of changes during the solution
        procedure.

        Parameters
        ----------
        K : ndarray
            current local linear operand, e.g. stiffness
        interface_dofs : ndarray
            current local ids of dofs, that are part of the subdomain's interfaces
        """
        self.K = K
        self.Q = csr_matrix(np.zeros_like(self.K))
        self.interface_dofs = interface_dofs
        self.interior_dofs = self._identify_interior_dofs(self.interface_dofs)
        self._set_Q()

    def _identify_interior_dofs(self, interface_dofs):
        """
        Setter for interior dofs, after interface-dofs are set
        """
        interior_dofs = np.setdiff1d(np.arange(self.K.shape[0]), interface_dofs)
        interior_dofs.astype(dtype=int)
        return interior_dofs

    def _set_Q(self):
        """
        Setter for the preconditioner-matrix, that defines the preconditioner's behavior.
        """
        pass

    def apply(self, u_b):
        """
        Application method for the local preconditioner to a vector.

        Parameters
        ----------
        u_b : ndarray
            vector, the local preconditioner shall be applied to

        Returns
        -------
        f_b : ndarray
            preconditioned vector
        """
        return self.Q.dot(u_b)

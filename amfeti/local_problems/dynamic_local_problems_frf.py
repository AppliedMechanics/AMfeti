#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Static local problems, meaning local problems, that don't have a time-integrator or time-dependent properties.
"""

from .local_problem_base import LocalProblemBase
from amfeti.scaling import MultiplicityScaling
from amfeti.preconditioners import *
from amfeti.linalg.datatypes import Matrix
import logging
from copy import copy

__all__ = [
    'LinearStaticLocalProblem',
    'NonlinearStaticLocalProblem'
]


class LinearDynamicLocalProblemFRF(LocalProblemBase):
    """
    Most simple local problem, that represents a linear static system in the context of a structural mechanics problem.
    More general it is a local problem based on a linear system of equations.

    Attributes
    ----------
    dimension : int
        number of local degrees of freedom
    K : Matrix
        local stiffness matrix
    B : dict
        dictionary of local connection matrices
    f : ndarray
        local external loads or right-hand-side
    q : ndarray
        local solution
    _interface_dofs : ndarray
        dofs on the interfaces
    _interior_dofs : ndarray
        dofs, that are not part of any interface
    _config_dict : dict
        dictionary with all configuration-information
    """
    def __init__(self, global_id, Z, K, B, f, **kwargs):
        """
        Parameters
        ----------
        global_id : int, string
            global id of the local problem
        K : ndarray
            local stiffness matrix
        B : dict
            dictionary of local connection matrices
        f : ndarray
            local external loads or right-hand-side
        kwargs : dict
            optional configuration
        """
        super().__init__(global_id)
        self.dimension = Z.shape[0]
        self._config_dict = {'pseudoinverse_config': {'method': 'svd',
                                                      'tolerance': 1e-10},
                             'preconditioner': None,
                             'scaling': None}
        self.set_config(kwargs)

        if isinstance(Z, Matrix):
            self.Z = Z
        else:
            self.Z = Matrix(Z, pseudoinverse_kargs=self._config_dict['pseudoinverse_config'])

        if isinstance(K, Matrix):
            self.K = K
        else:
            self.K = Matrix(K, pseudoinverse_kargs=self._config_dict['pseudoinverse_config'])



        self.f = f
        self.B = dict()
        for interface, B_mat in B.items():
            if isinstance(B_mat, csr_matrix):
                self.B[interface] = B_mat
            else:
                self.B[interface] = csr_matrix(B_mat)

        self._interface_dofs, self._interior_dofs = self._get_interface_and_interior_dofs(self.B)
        self.q = None
        self.update_preconditioner_and_scaling()
        self.lamda = None

    @property
    def interfaces(self):
        """
        All interfaces, that are part of the local problem's connections to other local problems

        Parameters
        ----------
        None

        interfaces : list
            all interfaces of this local problem with other local problems
        -------
        None
        """
        return list(self.B.keys())

    @property
    def kernel(self):
        """
        Provides the kernel of the local stiffness matrix

        Parameters
        ----------
        None

        Returns
        -------
        kernel : ndarray
            kernel of the stiffness matrix
        """
        return self.Z.kernel

    def set_config(self, new_config_dict):
        """
        Updates the local configuration dictionary

        Parameters
        ----------
        new_config_dict : dict
            dictionary of new configuration-parameters

        Returns
        -------
        None
        """
        if isinstance(new_config_dict, dict):
            self._config_dict.update(new_config_dict)
        if self._config_dict['preconditioner'] is not None and self._config_dict['scaling'] is None:
            self._config_dict['scaling'] = MultiplicityScaling()
            logger = logging.getLogger(__name__)
            logger.debug('No scaling was given, while preconditioner was set. Setting scaling to Multiplicity')
            
    def update_local_matrices(self,Z):
        if isinstance(Z, Matrix):
            self.Z = Z
        else:
            #self.Z = Matrix(Z, pseudoinverse_kargs=self._config_dict['pseudoinverse_config'])
            self.Z.data = Z
            self.Z.inverse_computed = False

    def update_preconditioner_and_scaling(self):
        """
        Updater for preconditioner and scaling, which are defined by the config dictionary

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.preconditioner is None:
            self.preconditioner = self._config_dict['preconditioner']
        if self.scaling is None:
            self.scaling = self._config_dict['scaling']
        if self.preconditioner is not None and self.scaling is not None:
            self.preconditioner.update(self.Z, self._interface_dofs)
            self.scaling.update(self.B)
        else:
            logger = logging.getLogger(__name__)
            logger.debug('No preconditioner and/or no scaling was specified. Omitting the updating.')

    def solve(self, external_solution_dict, external_force_bool=False, rigid_body_modes=None):
        """
        Solution-method for the local problem

        Parameters
        ----------
        external_solution_dict : dict
            external solution, that acts on the local problem and is separated in interfaces according to the local
            B-matrices

        external_force_bool : Boolean
            trigger for taking external force into account

        rigid_body_modes : ndarray
            in case of singular local problems the kernel-solutions (rigid body modes in structural mechanics) can be
            added to the local solution

        Returns
        -------
        u_b : dict
            dictionary of boundary-solutions with interface-keys
        """
        f = self._local_right_hand_side(external_solution_dict, external_force_bool)

        q = self.Z.apply_inverse(f)
        if rigid_body_modes is not None and rigid_body_modes.size is not 0:
            q += self.Z.kernel.dot(rigid_body_modes)
        if external_force_bool:
            self.q = q

        return self._distribute_to_interfaces(q)

    def precondition(self, q_b_dict):
        """
        Precondition the external solution with local contributions

        Parameters
        ----------
        q_b_dict : dict
            predefined local solutions on the interfaces

        Returns
        -------
        external_solution_dict_scaled : dict
            preconditioned external solutions on the interfaces
        """
        if self.preconditioner is None:
            external_solution_dict_scaled = q_b_dict
        else:
            u_b_dict_scaled = self.scaling.apply(q_b_dict)
            u_b_expanded = self._expand_external_solution(u_b_dict_scaled)
            external_solution = self.preconditioner.apply(u_b_expanded)
            external_solution_dict = self._distribute_to_interfaces(external_solution)
            external_solution_dict_scaled = self.scaling.apply(external_solution_dict)

        return external_solution_dict_scaled

    def _local_right_hand_side(self, external_solution_dict, external_force_bool):
        if not external_force_bool:
            f = np.zeros_like(self.f)
        else:
            f = np.copy(self.f)

        lamda = self._expand_external_solution(external_solution_dict)
        self.lamda = copy(lamda)

        return f - lamda

    def _expand_external_solution(self, external_solution_dict):
        """
        Maps external solutions from interfaces to local degrees of freedom

        Parameters
        ----------
        external_solution_dict : dict
            dictionary of external solutions with interface-ids as keys

        Returns
        -------
        lamda : ndarray
            mapped external solution
        """
        lamda = None
        if external_solution_dict is not None:
            lamda = np.zeros(self.dimension)
            for interface_id, B in self.B.items():
                lamda = lamda + B.T.dot(external_solution_dict[interface_id])
        return lamda

    def _distribute_to_interfaces(self, q):
        """
        Maps local solutions with local degrees of freedom to interfaces

        Parameters
        ----------
        q : ndarray
            local solutions

        Returns
        -------
        ub_dict : dict
            dictionary of mapped interface-solutions (sign-changes possible according to signs in B-matrix)
        """
        qb_dict = {}
        for interface_id, B in self.B.items():
            qb_dict[interface_id] = B @ q
        return qb_dict

    def _get_interface_and_interior_dofs(self, B_dict):
        """
        Selects interface-dofs (dofs with non-zero entries) and interior-dofs from B-matrix and stores them

        Parameters
        ----------
        B_dict : dict
            B-matrices

        Returns
        -------
        interface_dofs : ndarray
            interface-dofs
        interior_dofs : ndarray
            interior_dofs
        """
        interface_dofs = np.array([], dtype=int)
        for interface, B in B_dict.items():
            interface_dofs = np.append(interface_dofs, B.indices)
        interface_dofs = np.unique(interface_dofs)
        interior_dofs = np.setdiff1d(np.arange(self.dimension), interface_dofs)

        return interface_dofs, interior_dofs




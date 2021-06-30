#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Dynamic local problems, meaning local problems, that need a time-integrator object and have time-dependent properties.
"""

from .static_local_problems import LinearStaticLocalProblem
from amfeti.scaling import MultiplicityScaling
from amfeti.preconditioners import *
from amfeti.linalg.datatypes import Matrix
import logging
from copy import copy

__all__ = [
    'LinearDynamicLocalProblem',
    'NonlinearDynamicLocalProblem'
]


class LinearDynamicLocalProblem(LinearStaticLocalProblem):
    """
    Local problem with linear time-dependent properties, that represents a linear dynamic system in the context of a
    structural mechanics problem. More general it is a local problem based on a linear system of equations with repeated
    right-hand-sides.

    Attributes
    ----------
    dimension : int
        number of local degrees of freedom
    K : Matrix
        local jacobian matrix
    B : dict
        dictionary of local connection matrices
    f : ndarray
        local force-residual
    t : ndarray
        local times
    q : dict
        local solutions at time-steps (e.g. displacements)
    dq : dict
        local solutions' first time-derivative (e.g. velocities) at time-steps
    ddq : dict
        local solutions' second time-derivative (e.g. acceleration) at time-steps
    _integrator : IntegratorBase
        local time-integration object describing the dynamic behavior of the local problem. For detailed specifications
        on the Integrator-class see `Basics of time-integration` or `Requirements to an Integrator-Class` and for the
        required API the :class:`~amfeti.local_problems.integrator_base.IntegratorBase`
    _t0 : double
        initial time
    _q0 : ndarray
        initial local solution
    _dq0 : ndarray
        initial first time-derivative of local solution
    _ddq0 : ndarray
        initial second time-derivative of local solution
    _dq_p : ndarray
        predicted first time-derivative of local solution
    _delta_dq : ndarray
        incremental correction for predicted first time-derivative of local solution
    _interface_dofs : ndarray
        dofs on the interfaces
    _interior_dofs : ndarray
        dofs, that are not part of any interface
    _config_dict : dict
        dictionary with all configuration-information
    """
    def __init__(self, global_id, integrator, B_dict, inital_solutions_dict, **kwargs):
        """
        Parameters
        ----------
        integrator : Integrator
            integrator for system's dynamics
        B_dict : dict
            dictionary of local connection matrices
        inital_solutions_dict : dict
            initial solutions and time
        kwargs : dict
            optional configuration
        """
        self._t0 = inital_solutions_dict['t0']
        self._q0 = inital_solutions_dict['q0']
        self._dq0 = inital_solutions_dict['dq0']
        self._ddq0 = inital_solutions_dict['ddq0']
        self._integrator = integrator
        self._integrator.set_prediction(self._q0, self._dq0, self._ddq0, self._t0)
        self._dq_p = self._integrator.dq_p
        self._delta_dq = np.zeros_like(self._dq_p)

        super().__init__(global_id, self._integrator.jacobian(self._dq_p), B_dict,
                         -self._integrator.residual_int(self._dq_p)+self._integrator.residual_ext(self._dq_p), **kwargs)

        self.t = np.array([])
        self.q = dict()
        self.dq = dict()
        self.ddq = dict()
        self._store_solutions(self._t0, self._q0, self._dq0, self._ddq0)

    def _store_solutions(self, t, q, dq, ddq):
        """
        Writes solution in solution-properties

        Parameters
        ----------
        t : double
            time
        q : ndarray
            final local solution at given time
        dq : ndarray
            final local solution's first time-derivative at given time
        ddq : ndarray
            final local solution's second time-derivative at given time

        Returns
        -------
        None
        """
        self.t = np.append(self.t, t)
        N_timesteps = len(self.t)-1
        self.q[N_timesteps] = copy(q)
        self.dq[N_timesteps] = copy(dq)
        self.ddq[N_timesteps] = copy(ddq)

    def update_system(self, external_solution_dict, update_input_dict=None):
        """
        Updates local states with given external solution and optionally some further input, e.g. a load-factor.

        Parameters
        ----------
        external_solution_dict : dict
            external interface-solutions
        update_input_dict : dict
            optional further input, e.g. load-factor

        Returns
        -------
        None
        """
        if update_input_dict['solver_start']:
            N_timesteps = len(self.t)-1
            self._integrator.set_prediction(self.q[N_timesteps], self.dq[N_timesteps], self.ddq[N_timesteps],
                                            self.t[N_timesteps])
            self.f = -self._integrator.residual_int(self._dq_p) - self._integrator.residual_ext(self._dq_p)
        else:
            self._dq_p += self._delta_dq
            self._integrator.set_correction(self._dq_p)
            self._store_solutions(self._integrator.t_p, self._integrator.q_p, self._integrator.dq_p, self._integrator.ddq_p)

    def dump_local_information(self):
        """
        Returns some local information, such as the current time.

        Parameters
        ----------
        None

        Returns
        -------
        info_dict : dict
            some local information
        """
        info_dict = {'t': self._integrator.t_p}
        return info_dict

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
        q_b : dict
            dictionary of boundary-solutions with interface-keys
        """
        if rigid_body_modes is not None and rigid_body_modes.size is not 0:
            logger = logging.getLogger(__name__)
            msg = ['Unexpected rigid-body-modes received in dynamic local problem ', self.id,
                   '. Please check the local Jacobian, if it is still non-singular.']
            logger.warning(msg)

        f = self._local_right_hand_side(external_solution_dict, external_force_bool)

        delta_dq = self.K.apply_inverse(f)
        if external_force_bool:
            self._delta_dq = delta_dq
            return self._distribute_to_interfaces(delta_dq + self._integrator.q_p)
        else:
            return self._distribute_to_interfaces(delta_dq)


class NonlinearDynamicLocalProblem(LinearDynamicLocalProblem):
    def __init__(self, global_id, integrator, B_dict, inital_solutions_dict, **kwargs):
        """
        Parameters
        ----------
        integrator : Integrator
            integrator for system's dynamics
        B_dict : dict
            dictionary of local connection matrices
        inital_solutions_dict : dict
            initial solutions and time
        kwargs : dict
            optional configuration
        """
        super().__init__(global_id, integrator, B_dict, inital_solutions_dict, **kwargs)
        self._load_factor = 1.0

    @property
    def residual(self):
        """
        Current local force-residual
        """
        return self._integrator.residual_int(self._dq_p) + \
            self._load_factor * self._integrator.residual_ext(self._dq_p) + self.lamda

    def update_system(self, external_solution_dict, update_input_dict=None):
        """
        Updates local states with given external solution and optionally some further input, e.g. a load-factor.

        Parameters
        ----------
        external_solution_dict : dict
            external interface-solutions
        update_input_dict : dict
            optional further input, e.g. load-factor

        Returns
        -------
        None
        """
        logger = logging.getLogger(__name__)
        if 'nonlinear_solver_finished' in update_input_dict.keys() and update_input_dict['nonlinear_solver_finished']:
            self._delta_dq = np.zeros_like(self._dq_p)
            self._store_solutions(self._integrator.t_p, self._integrator.q_p, self._integrator.dq_p,
                                  self._integrator.ddq_p)
            logger.debug('Local solutions stored')
        elif 'nonlinear_solver_start' in update_input_dict.keys() and update_input_dict['nonlinear_solver_start']:
            N_timesteps = len(self.t) - 1
            self._integrator.set_prediction(self.q[N_timesteps], self.dq[N_timesteps], self.ddq[N_timesteps],
                                            self.t[N_timesteps])
            self._dq_p = self._integrator.dq_p
        else:
            if update_input_dict is not None:
                self._load_factor = update_input_dict['load_factor']
            logger.debug('Local load-factor: ' + str(self._load_factor))
            self._dq_p += self._delta_dq
            self._integrator.set_correction(self._dq_p)
            self.lamda = self._expand_external_solution(external_solution_dict)
            self.f = -self.residual
            self.K.update(self._integrator.jacobian(self._dq_p))
            self.update_preconditioner_and_scaling()

    def dump_local_information(self):
        """
        Returns some local information, such as the current residual and time.

        Parameters
        ----------
        None

        Returns
        -------
        info_dict : dict
            some local information
        """
        info_dict = {'residual': np.linalg.norm(self.residual),
                     't': self._integrator.t_p}
        return info_dict

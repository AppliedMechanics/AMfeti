#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Module that defines the control-technique to follow the system's equilibrium path. This is especially required in case
of nonlinear systems.
"""

import numpy as np
from amfeti.config_base import ConfigBase
from .nonlinear_solver import NewtonRaphson

__all__ = ['LoadSteppingControl']


class ControlBase(ConfigBase):
    def __init__(self):
        super().__init__()

    def solve(self, solve_linear_problem, update_residual, x0):
        raise NotImplementedError('Solver function was not implemented for subclass')


class ArcLengthControl(ControlBase):
    """
    Loadpath-controlling algorithm, that implements an arc-length method to follow a nonlinear equilibrium-path and is
    able to pass critical points.

    Attributes
    ----------

    _rtol : float
        relative tolerance for the nonlinear solver
    _atol : float
        absolute tolerance for the nonlinear solver
    _default_rtol_scaling : float
        default scaling-factor for the relative tolerance, which is set to the Newton-solver's start-residual
    _external_rtol_scaling : float
        can be set by the user and overrides the default rtol_scaling
    """
    def __init__(self):
        self._q_control = None
        self._load_control = None
        self._kappa = None
        super().__init__()
        self.set_config({'arc_shape': 1.0,
                         'N_steps': 10,
                         'arc_length': 0.0,
                         'nonlinear_solver': NewtonRaphson(),
                         'nonlinear_solver_options': dict()})
        self._nonlinear_solver = self._config_dict['nonlinear_solver']
        self._load_factor = 0.0

    def solve(self, solve_linear_problem, update_residual, x0):
        self._nonlinear_solver.set_config(self._config_dict['nonlinear_solver_options'])
        info_dict = dict()

        def _update_newton_residual(q):
            return update_residual(q, {'load_factor': self._load_factor})

        x = x0
        for i_step in range(0, self._config_dict['N_steps']):
            # Start Newton Iterations
            self._update_loadfactor(x)
            x, newton_info = self._nonlinear_solver.solve(solve_linear_problem, _update_newton_residual, x)
            info_dict[i_step] = {'newton': newton_info}

        self._load_factor = 0.0
        return info_dict

    def _update_loadfactor(self, delta_q):
        raise ValueError('ArcLength not yet fully supported!')
        residual_load = self.arc_shape ** 2 * (self.load_factor - self.load_factor_n)**2
        jacobian_delta_q = np.array([0])
        residual_q = np.array([0])
        for prob_id, local_problem in delta_q['local_problems'].items():
            delta_q_n = local_problem.q - self.q_n[prob_id]
            jacobian_delta_q += 2 * np.dot(delta_q_n, local_problem.delta_q)
            residual_q += np.dot(delta_q_n, delta_q_n)
        self.load_factor += (-residual_q - residual_load - self.arc_length**2 - jacobian_delta_q) / \
                            (2 * self.arc_shape**2 * (self.load_factor - self.load_factor_n))


class LoadSteppingControl(ArcLengthControl):
    """
    Loadpath-controlling algorithm, that implements a load-stepping method to follow a nonlinear equilibrium-path until
    a limit-point appears.

    Attributes
    ----------
    """
    def __init__(self):
        super().__init__()
        self.set_config({'arc_length': 1.0/self._config_dict['N_steps']})

    def set_config(self, new_config_dict):
        super(LoadSteppingControl, self).set_config(new_config_dict)
        super(LoadSteppingControl, self).set_config({'arc_length': 1.0 / self._config_dict['N_steps']})
        self._load_factor = 0.0

    def _update_loadfactor(self, q):
        self._load_factor += self._config_dict['arc_length']

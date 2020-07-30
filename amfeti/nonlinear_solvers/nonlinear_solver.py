#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Solver-module to solve nonlinear boundary value problems.
"""

import logging
import numpy as np
from copy import copy
from amfeti.config_base import ConfigBase

__all__ = ['NewtonRaphson']

abort_statement = '''
###############################################################################
#### The current computation has been aborted.                             ####
#### No convergence was gained within the number of given iteration steps. ####
###############################################################################
'''


class NewtonRaphson(ConfigBase):
    """
    Class for solving nonlinear boundary value problem with a classical Newton-Raphson technique.
    It requires evaluation of the residuals' first derivative in every iteration.
    """
    def __init__(self):
        super().__init__()
        self.set_config({'atol': 1.0e-08,
                         'rtol': None,
                         'max_iter': 10,
                         'log_iterations': False})
        self._residual_0 = None

    def solve(self, solve_linear_problem, update_residual, x0):
        # Initialize
        info_dict = dict()
        info_dict['residual'] = np.array([])
        info_dict['linear_solver'] = {}
        linear_info_dict = {}
        iteration = 0
        x = copy(x0)
        res = update_residual(x)
        logger = logging.getLogger(__name__)
        logger.info('Start-iteration: {0:3d}, residual: {1:6.3E}'.format(iteration, res))
        if self._config_dict['rtol'] is not None:
            self._residual_0 = copy(res)

        while iteration <= self._config_dict['max_iter']:
            if self._config_dict['log_iterations']:
                info_dict['residual'] = np.append(info_dict['residual'], res)
                info_dict['iterations'] = iteration
                if iteration > 0:
                    info_dict['linear_solver'].update({iteration - 1: copy(linear_info_dict)})
                logger.info('Newton-Iteration: {0:3d}, residual: {1:6.3E}'.format(iteration, res))

            # check convergence
            if self._check_convergence(res):
                logger.info('Converged with iteration: {0:3d}, residual: {1:6.3E}'.format(iteration, res))
                break
            elif iteration == self._config_dict['max_iter']:
                print(abort_statement)
                logger.info(abort_statement)
                logger.info('Newton-Iteration: {0:3d}, residual: {1:6.3E}'.format(iteration, res))

            delta_x, linear_info_dict = solve_linear_problem()

            # correct variables
            x += delta_x

            # Update residual
            res = update_residual(x)

            iteration += 1
        return x, info_dict

    def _check_convergence(self, residual):
        if self._config_dict['rtol'] is not None and self._residual_0 is not None:
            return residual / self._residual_0 <= self._config_dict['rtol'] or residual <= self._config_dict['atol']
        else:
            return residual <= self._config_dict['atol']

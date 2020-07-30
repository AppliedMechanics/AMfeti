#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

from amfeti.config_base import ConfigBase
from amfeti.solver_managers import SerialSolverManager, ParallelSolverManager
import logging


class FetiSolverBase(ConfigBase):
    """
    Base-class for FETI-solver presets. These Solver-classes build a predefined solver according to limited
    user-specifications. For more sofisticated solvers, build them on your own from local problems and
    solver-managers.

    Attributes
    ----------
    _solver_manager : SolverManagerBase
        solver manager for the global problem
    _local_problems : LocalProblemBase
        local problem
    _config_dict : dict
        solver-configuration
    """
    def __init__(self):
        """
        Parameters
        ----------
        None
        """
        self._solver_manager = None
        self._local_problems = dict()
        super().__init__()
        self.set_config({'solver_manager': None})

    def update(self):
        """
        Sets and updates the solver-manager according to configuration.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logger = logging.getLogger(__name__)
        if self._config_dict['solver_manager'] is None:
            if self._config_dict['use_parallel']:
                self._solver_manager = ParallelSolverManager(self._local_problems, self._config_dict['global_solver'])
            else:
                self._solver_manager = SerialSolverManager(self._local_problems, self._config_dict['global_solver'])
        else:
            self._solver_manager = self._config_dict['solver_manager']
            logger.debug('Using predefined solver-manager')

        self._solver_manager.update()
        self._dual_solution_length = self._solver_manager.no_lagrange_multiplier

    def solve(self):
        pass

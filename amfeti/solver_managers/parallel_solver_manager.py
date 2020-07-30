#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Parallel solver managers, that set global options, control the global solution-process and call local problems
"""

import numpy as np
from copy import copy
from amfeti.solver_managers import SerialSolverManager
from amfeti.parallelization_managers import MPIManager


__all__ = ['ParallelSolverManager']


class ParallelSolverManager(SerialSolverManager):
    def __init__(self, local_problems_dict, solver):
        super().__init__(local_problems_dict, solver)
        self._local_problems_distributed = False
        self._parallelization_manager = None
        self.set_config({'parallelization_manager': MPIManager(),
                         'local_manager_config': dict()})

    @property
    def solution(self):
        solutions_dict = self._parallelization_manager.read_solutions()
        solver_information = dict()
        local_problems = dict()
        dual_solution = None
        kernel_modes = None
        for key, solution in solutions_dict.items():
            solver_information[key] = solution.solver_information
            for problem_id, problem in solution.local_problems.items():
                local_problems[problem_id] = problem
            dual_solution = solution.dual_solution
            kernel_modes = solution.kernel_modes
        self._solution.update(local_problems, dual_solution, kernel_modes, solver_information)
        return self._solution

    @property
    def no_lagrange_multiplier(self):
        return self._global_dof_dimension

    def update(self):
        if self._local_problems_distributed:
            raise ValueError('Call of distributed local problems not supported yet. After solving the system by the '
                             'parallel solver, local problems have to be updated serialized and updated again.')
        else:
            self._coarse_grid = self._config_dict['coarse_grid']
            self._solution = self._config_dict['solution']
            self._interface2dof_map = dict()
            self._global_dof_dimension = 0

            for problem_id, local_problem in self._local_problems_dict.items():
                for interface, B in local_problem.B.items():
                    if interface not in self._interface2dof_map:
                        new_global_dof_dimension = self._global_dof_dimension + B.shape[0]
                        self._interface2dof_map[interface] = np.arange(self._global_dof_dimension,
                                                                       new_global_dof_dimension)
                        self._global_dof_dimension = new_global_dof_dimension

            if self._parallelization_manager is None:
                self._parallelization_manager = copy(self._config_dict['parallelization_manager'])
            self._config_dict['local_manager_config'].update({'coarse_grid': self._coarse_grid,
                                                                'solution': self._solution,
                                                                'interface2dof_map': self._interface2dof_map,
                                                                'global_dof_dimension': self._global_dof_dimension})
            self._parallelization_manager.set_up_parallel_process(self._local_problems_dict, self.solver,
                                                                  self._config_dict['local_manager_config'])

    def solve(self):
        self._parallelization_manager.launch_parallel_process()
        self._local_problems_dict = self._parallelization_manager.load_local_problems()

    def update_local_problems(self, lambda_sol, update_input_dict=None, local_info_dict=None):
        """
        Updates local problems with a dual solution and optionally with further input. After the update an information-
        dump is collected from local problems.

        Parameters
        ----------
        lambda_sol : ndarray
            dual solution
        update_input_dict : dict
            optional additional update-information for local problems
        local_info_dict : dict
            optional dictionary for the info-dump

        Returns
        -------
        local_info_dict : dict
            optional dictionary containing the local info-dumps
        """
        self._local_problems_dict = self._parallelization_manager.load_local_problems()
        for problem_id, local_problem in self._local_problems_dict.items():
            local_problem.update_system(self._vector2interfacedict(lambda_sol), update_input_dict)
            if isinstance(local_info_dict, dict):
                local_info_dict.update({problem_id: local_problem.dump_local_information()})
        self._parallelization_manager.set_up_parallel_process(self._local_problems_dict, self.solver,
                                                              self._config_dict['local_manager_config'])
        return local_info_dict
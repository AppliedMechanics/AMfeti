#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Solution module, that stores the solver's solutions and information of the solution-process
"""

__all__ = ['StandardSolution']


class SolutionBase:
    def __init__(self):
        self._dual_solution = None
        self._alpha_solution = None
        self._local_problems_dict = dict()
        self.solver_information = dict()

    def update(self, local_problems_dict, dual_solution, alpha_solution, solver_info_dict):
        self._dual_solution = dual_solution
        self._alpha_solution = alpha_solution
        self._local_problems_dict.update(local_problems_dict)
        self.solver_information = solver_info_dict


class StandardSolution(SolutionBase):
    def __init__(self):
        super().__init__()

    @property
    def local_problems(self):
        return self._local_problems_dict

    @property
    def dual_solution(self):
        return self._dual_solution

    @property
    def kernel_modes(self):
        return self._alpha_solution

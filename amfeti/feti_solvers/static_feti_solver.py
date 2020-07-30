#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
A collection of FETI-solvers for static linear and nonlinear problems
"""

from .feti_solver_base import FetiSolverBase
from amfeti.local_problems import LinearStaticLocalProblem, NonlinearStaticLocalProblem
from amfeti.solvers import PCPGsolver
from amfeti.preconditioners import DirichletPreconditioner, LumpedPreconditioner
from amfeti.scaling import MultiplicityScaling
from amfeti.nonlinear_solvers import LoadSteppingControl
import numpy as np
import logging


__all__ = ['LinearStaticFetiSolver',
           'NonlinearStaticFetiSolver']


class LinearStaticFetiSolver(FetiSolverBase):
    """
    FETI-solver for linear static problems

    Attributes
    ----------
    _solver_manager : SolverManagerBase
        solver manager for the global problem
    _local_problems : LocalProblemBase
        local problem
    _config_dict : dict
        solver-configuration
    _dual_solution_length : int
        number of global dual degrees of freedom
    """
    def __init__(self, K_dict, B_dict, f_dict, **kwargs):
        """
        Parameters
        ----------
        K_dict : dict
            stiffness-matrices
        B_dict : dict
            dictionary of connectivities per interface-ids
        f_dict : dict
            external forces
        kwargs : dict
            optional arguments for solver-configuration
        """
        super().__init__()

        self.set_config({'K_dict': K_dict,
                         'B_dict': B_dict,
                         'f_dict': f_dict,
                         'use_parallel': False,
                         'global_solver': PCPGsolver()})
        self.set_config(kwargs)
        self._create_local_problems()
        self._dual_solution_length = None

    def _create_local_problems(self):
        """
        Creates linear static local problems and configures them

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        B_dict = self._config_dict['B_dict']
        f_dict = self._config_dict['f_dict']
        for problem_id, K in self._config_dict['K_dict'].items():
            self._local_problems[problem_id] = LinearStaticLocalProblem(problem_id, K, B_dict[problem_id],
                                                                        f_dict[problem_id])
            self._local_problems[problem_id].set_config({'preconditioner': DirichletPreconditioner(),
                                                         'scaling': MultiplicityScaling()})
            self._local_problems[problem_id].update_preconditioner_and_scaling()

    def solve(self):
        """
        Runs the solver-manager's solve-method.

        Parameters
        ----------
        None

        Returns
        -------
        solution : StandardSolution
            solution-object
        """
        self._solver_manager.solve()
        return self._solver_manager.solution


class NonlinearStaticFetiSolver(LinearStaticFetiSolver):
    """
    FETI-solver for nonlinear static problems

    Attributes
    ----------
    _nonlinear_solver : ControlBase
        nonlinear global solver
    _solver_manager : SolverManagerBase
        solver manager for the global problem
    _local_problems : LocalProblemBase
        local problem
    _config_dict : dict
        solver-configuration
    _dual_solution_length : int
        number of global dual degrees of freedom
    """
    def __init__(self, K_dict, B_dict, f_int_dict, f_ext_dict, q_0_dict, **kwargs):
        """
        Parameters
        ----------
        K_dict : dict
            callbacks for stiffness-matrices
        B_dict : dict
            dictionary of connectivities per interface-ids
        f_int_dict : dict
            callbacks for internal forces
        f_ext_dict : dict
            callbacks for external forces
        q_0_dict : dict
            start values for local solutions (e.g. displacements)
        kwargs : dict
            optional arguments for solver-configuration
        """
        kwargs_dict = {'loadpath_controller': LoadSteppingControl(),
                       'loadpath_controller_options': {'nonlinear_solver_options': {'log_iterations': True}},
                       'f_int_dict': f_int_dict,
                       'q_0_dict': q_0_dict}
        kwargs_dict.update(kwargs)
        super().__init__(K_dict, B_dict, f_ext_dict, **kwargs_dict)
        self._nonlinear_solver = self._config_dict['loadpath_controller']
        self._nonlinear_solver.set_config(self._config_dict['loadpath_controller_options'])
        self._dual_solution_length = None

    def solve(self):
        self._solver_manager.update_local_problems(np.zeros(self._dual_solution_length),
                                                   {'load_factor': self._nonlinear_solver._load_factor}, {})
        lambda_0 = self._solver_manager.initialize_lambda()
        nonlinear_info_dict = self._nonlinear_solver.solve(self._solve_linear_problem_callback, self._update_residual, lambda_0)
        solution = self._solver_manager.solution
        solution.solver_information = nonlinear_info_dict
        return solution

    def _create_local_problems(self):
        B_dict = self._config_dict['B_dict']
        f_dict = self._config_dict['f_dict']
        f_int_dict = self._config_dict['f_int_dict']
        q_0_dict = self._config_dict['q_0_dict']
        for problem_id, K in self._config_dict['K_dict'].items():
            self._local_problems[problem_id] = NonlinearStaticLocalProblem(problem_id, K, B_dict[problem_id],
                                                                           f_int_dict[problem_id], f_dict[problem_id],
                                                                           q_0_dict[problem_id])
            self._local_problems[problem_id].set_config({'preconditioner': DirichletPreconditioner(),
                                                         'scaling': MultiplicityScaling()})
            self._local_problems[problem_id].update_preconditioner_and_scaling()

    def _solve_linear_problem_callback(self):
        solver_information = self._solver_manager.solve()
        solution = self._solver_manager.solution
        delta_lambda = solution.dual_solution
        return delta_lambda, solver_information

    def _update_residual(self, lamda_alpha, update_input_dict):
        lamda = lamda_alpha
        local_info_dict = dict()
        local_info_dict = self._solver_manager.update_local_problems(lamda, update_input_dict, local_info_dict)
        residuals = np.array([])
        for local_id, info_dict in local_info_dict.items():
            residuals = np.append(residuals, info_dict['residual'])
        self._solver_manager.update()
        return np.max(residuals)

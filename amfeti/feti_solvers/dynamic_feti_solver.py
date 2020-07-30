#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
A collection of FETI-solvers for dynamic linear and nonlinear problems
"""

import numpy as np
from math import isclose
import logging
from .feti_solver_base import FetiSolverBase
from amfeti.solvers import PCPGsolver
from amfeti.preconditioners import DirichletPreconditioner
from amfeti.scaling import MultiplicityScaling
from amfeti.solver_managers import SerialSolverManager
from amfeti.local_problems import LinearDynamicLocalProblem, NonlinearDynamicLocalProblem
from amfeti.nonlinear_solvers import LoadSteppingControl
from copy import copy


__all__ = ['LinearDynamicFetiSolver',
           'NonlinearDynamicFetiSolver']


class LinearDynamicFetiSolver(FetiSolverBase):
    """
    FETI-solver for linear dynamic problems

    Attributes
    ----------
    _solver_manager : SolverManagerBase
        solver manager for the global problem
    _local_problem : LocalProblemBase
        local problem
    _config_dict : dict
        solver-configuration
    """
    def __init__(self, integrator_dict, B_dict, t0, t_end, q0_dict, dq0_dict, ddq0_dict, **kwargs):
        """
        Parameters
        ----------
        integrator_dict : dict
            integrator-objects describing the dynamic behavior of the local problems. For detailed specifications on the
            Integrator-class see `Basics of time-integration` or `Requirements to an Integrator-Class` and for the
            required API the :class:`~amfeti.local_problems.integrator_base.IntegratorBase`
        B_dict : dict
            dictionary of connectivities per interface-ids
        t0 : double
            start-time
        t_end : double
            end-time
        q0_dict : dict
            start-values for local solutions (e.g. displacements)
        dq0_dict : dict
            start-values for local solutions' first time-derivative (e.g. velocities)
        ddq0_dict : dict
            start-values for local solutions' second time-derivative (e.g. accelerations)
        kwargs : dict
            optional arguments for solver-configuration
        """
        super().__init__()

        self.set_config({'integrator_dict': integrator_dict,
                         'B_dict': B_dict,
                         't0': t0,
                         't_end': t_end,
                         'q0_dict': q0_dict,
                         'dq0_dict': dq0_dict,
                         'ddq0_dict': ddq0_dict,
                         'use_parallel': False,
                         'global_solver': PCPGsolver(),
                         'preconditioner': DirichletPreconditioner(),
                         'scaling': MultiplicityScaling()
                         })
        self.set_config(kwargs)
        self._create_local_problems()
        self._dual_solution_length = None

    def _create_local_problems(self):
        """
        Creates linear dynamic local problems and configures them

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        B_dict = self._config_dict['B_dict']
        t0 = self._config_dict['t0']
        q0 = self._config_dict['q0_dict']
        dq0 = self._config_dict['dq0_dict']
        ddq0 = self._config_dict['ddq0_dict']
        for problem_id, integrator in self._config_dict['integrator_dict'].items():
            inital_solutions_dict = {'t0': t0, 'q0': q0[problem_id], 'dq0': dq0[problem_id], 'ddq0': ddq0[problem_id]}
            if problem_id not in self._local_problems:
                self._local_problems[problem_id] = LinearDynamicLocalProblem(problem_id, integrator, B_dict[problem_id],
                                                                             inital_solutions_dict)
            self._local_problems[problem_id].set_config({'preconditioner': copy(self._config_dict['preconditioner']),
                                                         'scaling': copy(self._config_dict['scaling'])})
            self._local_problems[problem_id].update_preconditioner_and_scaling()

    def solve(self):
        """
        Runs the time-stepping and solver manager's solver method.

        Parameters
        ----------
        None

        Returns
        -------
        solution : StandardSolution
            solution-object
        """
        logger = logging.getLogger(__name__)
        t_step = 0
        solvers_info_dict = dict()
        while True:
            logger.info('Solving time-step ' + str(t_step))
            solver_information = self._solver_manager.solve()
            solvers_info_dict[t_step] = solver_information
            info_dict = self._solver_manager.update_local_problems(np.zeros(self._dual_solution_length), {}, {})
            problem_id0 = list(info_dict.keys())[0]
            t_array = np.array([])
            for problem_id, local_info_dict in info_dict.items():
                t = local_info_dict['t']
                t_array = np.append(t_array, t)
                if not isclose(info_dict[problem_id0]['t'], t):
                    logger.warning('Time-stepping in local problem ', problem_id, ' is not synchronized at final local '
                                                                                  'step with problem ', problem_id0,
                                   '.')
            t = np.min(t_array)
            t_step += 1
            if isclose(t, self._config_dict['t_end']) or t >= self._config_dict['t_end']:
                break
        solution = self._solver_manager.solution
        solution.solver_information = solvers_info_dict
        return solution


class NonlinearDynamicFetiSolver(LinearDynamicFetiSolver):
    """
    FETI-solver for nonlinear dynamic problems

    Attributes
    ----------
    _nonlinear_solver : ControlBase
        nonlinear global solver
    _solver_manager : SolverManagerBase
        solver manager for the global problem
    _local_problem : LocalProblemBase
        local problem
    _config_dict : dict
        solver-configuration
    _dual_solution_length : int
        number of global dual degrees of freedom
    """
    def __init__(self, integrator_dict, B_dict, t0, t_end, q0_dict, dq0_dict, ddq0_dict, **kwargs):
        """
        Parameters
        ----------
        integrator_dict : dict
            integrator-objects describing the dynamic behavior of the local problems. For detailed specifications on the
            Integrator-class see `Basics of time-integration` or `Requirements to an Integrator-Class` and for the
            required API the :class:`~amfeti.local_problems.integrator_base.IntegratorBase`
        B_dict : dict
            dictionary of connectivities per interface-ids
        t0 : double
            start-time
        t_end : double
            end-time
        q0_dict : dict
            start-values for local solutions (e.g. displacements)
        dq0_dict : dict
            start-values for local solutions' first time-derivative (e.g. velocities)
        ddq0_dict : dict
            start-values for local solutions' second time-derivative (e.g. accelerations)
        kwargs : dict
            optional arguments for solver-configuration
        """
        kwargs_dict = {'loadpath_controller': LoadSteppingControl(),
                       'loadpath_controller_options': {'nonlinear_solver_options': {'log_iterations': True,
                                                                                    'rtol': 1e-8},
                                                       'N_steps': 1}
                       }
        kwargs_dict.update(kwargs)
        super().__init__(integrator_dict, B_dict, t0, t_end, q0_dict, dq0_dict, ddq0_dict, **kwargs_dict)
        self._nonlinear_solver = self._config_dict['loadpath_controller']
        self._nonlinear_solver.set_config(self._config_dict['loadpath_controller_options'])
        self._dual_solution_length = None

    def _create_local_problems(self):
        """
        Creates linear dynamic local problems and configures them

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        B_dict = self._config_dict['B_dict']
        t0 = self._config_dict['t0']
        q0 = self._config_dict['q0_dict']
        dq0 = self._config_dict['dq0_dict']
        ddq0 = self._config_dict['ddq0_dict']
        for problem_id, integrator in self._config_dict['integrator_dict'].items():
            inital_solutions_dict = {'t0': t0, 'q0': q0[problem_id], 'dq0': dq0[problem_id], 'ddq0': ddq0[problem_id]}
            if problem_id not in self._local_problems:
                self._local_problems[problem_id] = NonlinearDynamicLocalProblem(problem_id, integrator,
                                                                                B_dict[problem_id],
                                                                                inital_solutions_dict)
            self._local_problems[problem_id].set_config({'preconditioner': copy(self._config_dict['preconditioner']),
                                                         'scaling': copy(self._config_dict['scaling'])})
            self._local_problems[problem_id].update_preconditioner_and_scaling()

    def solve(self):
        """
        Runs the time-stepping and solver manager's solver method.

        Parameters
        ----------
        None

        Returns
        -------
        solution : StandardSolution
            solution-object
        """
        logger = logging.getLogger(__name__)
        t_step = 0
        solvers_info_dict = dict()
        while True:
            logger.info('Solving time-step ' + str(t_step))
            lambda_0 = self._solver_manager.initialize_lambda()
            if t_step > 0:
                self._solver_manager.update_local_problems(np.zeros(self._dual_solution_length),
                                                                   {'load_factor': self._nonlinear_solver._load_factor,
                                                                    'nonlinear_solver_start': True}, {})
            nonlinear_info_dict = self._nonlinear_solver.solve(self._solve_linear_problem_callback,
                                                                  self._update_residual, lambda_0)
            solvers_info_dict[t_step] = nonlinear_info_dict
            info_dict = self._solver_manager.update_local_problems(np.zeros(self._dual_solution_length),
                                                                   {'load_factor': self._nonlinear_solver._load_factor,
                                                                    'nonlinear_solver_finished': True}, {})
            problem_id0 = list(info_dict.keys())[0]
            t_array = np.array([])
            for problem_id, local_info_dict in info_dict.items():
                t = local_info_dict['t']
                t_array = np.append(t_array, t)
                if not isclose(info_dict[problem_id0]['t'], t):
                    logger.warning('Time-stepping in local problem ', problem_id, ' is not synchronized at final local '
                                   'step with problem ', problem_id0, '.')
            t = np.min(t_array)
            t_step += 1
            if isclose(t, self._config_dict['t_end']) or t >= self._config_dict['t_end']:
                break
        solution = self._solver_manager.solution
        solution.solver_information = solvers_info_dict
        return solution

    def _solve_linear_problem_callback(self):
        """
        Caller for the solver-manager's linear solve-method, that returns the global incremental solution

        Parameters
        ----------
        None

        Returns
        -------
        delta_lambda : ndarray
            incremental global solution
        """
        solver_information = self._solver_manager.solve()
        solution = self._solver_manager.solution
        delta_lambda = solution.dual_solution
        return delta_lambda, solver_information

    def _update_residual(self, lamda_alpha, update_input_dict):
        """
        Updates the local residuals and selects the maximal local residual for the nonlinear solver

        Parameters
        ----------
        lamda_alpha : ndarray
            global solutions
        update_input_dict : dict
            additional input-parameters for the nonlinear solver and local problems, such as load-factor

        Returns
        -------
        residual_max : ndarray
            maximum norm of local residuals
        """
        lamda = lamda_alpha
        local_info_dict = dict()
        local_info_dict = self._solver_manager.update_local_problems(lamda, update_input_dict, local_info_dict)
        residuals = np.array([])
        for local_id, info_dict in local_info_dict.items():
            residuals = np.append(residuals, info_dict['residual'])
        self._solver_manager.update()
        return np.max(residuals)

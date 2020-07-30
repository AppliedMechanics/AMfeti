#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Local solver managers, that posses only one or a few local problems, perform global operations locally and use an
interface for exchange of information.
"""
from .serial_solver_manager import SerialSolverManager
import numpy as np
import logging


__all__ = ['LocalSolverManager']


class LocalSolverManager(SerialSolverManager):
    def __init__(self, local_problems_dict, solver):
        super().__init__(local_problems_dict, solver)
        self.set_config({'parallel_map': None,
                         'parallel_processor': None,
                         'parallel_processor_opt': None,
                         'local_problems_full': None})
        self._local_parallel_processor = None
        self._solution = self._config_dict['solution']

    @property
    def solution(self):
        self._solution.update(self._local_problems_dict, self._lambda_sol, self._alpha_sol, self._info_dict)
        return self._solution

    def update(self, local_parallel_processor):
        logger = logging.getLogger(__name__)
        logger.info('Updating local solver manager')
        self._local_parallel_processor = local_parallel_processor
        self._local_parallel_processor.set_config(self._config_dict['parallel_processor_opt'])
        self._local_parallel_processor.update()

        self._coarse_grid = self._config_dict['coarse_grid']
        self._interface2dof_map = self._config_dict['interface2dof_map']
        self._global_dof_dimension = self._config_dict['global_dof_dimension']

        BR_dict = dict()
        RTf_dict = dict()

        for problem_id, local_problem in self._local_problems_dict.items():
            BR_int_dict = dict()
            if local_problem.kernel.size == 0:
                for interface, B in local_problem.B.items():
                    BR_int_dict[interface] = np.array([])
            else:
                for interface, B in local_problem.B.items():
                    BR_int_dict[interface] = B @ local_problem.kernel
            BR_dict[problem_id] = BR_int_dict
            if local_problem.kernel.size == 0:
                RTf_dict[problem_id] = np.array([])
            else:
                RTf_dict[problem_id] = local_problem.kernel.T @ local_problem.f

        BR_neighbors = dict()
        RTf_neighbors = dict()
        self._local_parallel_processor.send_global_info(BR_dict)
        BR_prob = self._local_parallel_processor.receive_global_info(BR_dict)
        BR_neighbors.update(BR_prob)

        self._local_parallel_processor.send_global_info(RTf_dict)
        RTf_prob = self._local_parallel_processor.receive_global_info(RTf_dict)
        RTf_neighbors.update(RTf_prob)

        BR_dict.update(BR_neighbors)
        RTf_dict.update(RTf_neighbors)

        self._coarse_grid.update(BR_dict, RTf_dict, self._interfacedict2vector)

    def _F_action(self, v,  external_force=False):
        r"""
        Application of the input-vector on the FETI-solver's F-operator, such that
        .. math::
            \textbf{F}\vec{\lambda}

        is calculated

        Parameters
        ----------
        v : ndarray
            dual solution
        external_force : bool
            flag that switches on external loads in the local problems. During global iterations only the local-
            problems' reactions to changes of the dual solution is of interest. External loads are taken into account in
            the residual and the final solution.

        Returns
        -------
        -d : ndarray
            cumulated reaction of local problems on the interfaces (in structural mechanics viewed as gaps)
        """
        v_dict = self._vector2interfacedict(v)
        u_dict = {}
        for problem_id, local_problem in self._local_problems_dict.items():
            u_dict_local = local_problem.solve(v_dict, external_force)
            for interface, u_b in u_dict_local.items():
                if interface not in u_dict:
                    u_dict[interface] = {problem_id: u_b}
                else:
                    u_dict[interface].update({problem_id: u_b})

        # compute gap
        gap_dict = {}
        for interface_id, u_dict_interface in u_dict.items():
            for problem_id, u_b in u_dict_interface.items():
                if interface_id not in gap_dict:
                    gap_dict[interface_id] = u_b
                else:
                    gap_dict[interface_id] += u_b

            self._local_parallel_processor.send_interface_info(gap_dict[interface_id], interface_id)

        for interface_id, u_dict_interface in u_dict.items():
            neighbor_u_b = self._local_parallel_processor.receive_interface_info(gap_dict[interface_id], interface_id)
            if neighbor_u_b.size > 0:
                gap_dict[interface_id] += neighbor_u_b

        self._local_parallel_processor.send_global_info(gap_dict)
        gap_dict.update(self._local_parallel_processor.receive_global_info(gap_dict))

        d = self._interfacedict2vector(gap_dict)
        return -d

    def _apply_preconditioner(self, v):
        """
        Application of the local preconditioners to a given vector

        Parameters
        ----------
        v : ndarray
            global vector (in structural mechanics often viewed as gaps)

        Returns
        -------
        f : ndarray
            preconditioned dual solution
        """
        v_dict = self._vector2interfacedict(v)
        f_dict = {}
        for problem_id, local_problem in self._local_problems_dict.items():
            f_dict_local = local_problem.precondition(v_dict)
            for interface, f_b in f_dict_local.items():
                if interface not in f_dict:
                    f_dict[interface] = {problem_id: f_b}
                else:
                    f_dict[interface].update({problem_id: f_b})

        # compute interface force
        lambda_dict = {}
        for interface_id, f_dict_interface in f_dict.items():
            for problem_id, f_b in f_dict_interface.items():
                if interface_id not in lambda_dict:
                    lambda_dict[interface_id] = np.zeros_like(f_b)
                lambda_dict[interface_id] += f_b
            self._local_parallel_processor.send_interface_info(lambda_dict[interface_id], interface_id)

        for interface_id, f_dict_interface in f_dict.items():
            neighbors_lambda = self._local_parallel_processor.receive_interface_info(lambda_dict[interface_id], interface_id)
            if neighbors_lambda.size > 0:
                lambda_dict[interface_id] += neighbors_lambda

        self._local_parallel_processor.send_global_info(lambda_dict)
        lambda_dict.update(self._local_parallel_processor.receive_global_info(lambda_dict))

        f = self._interfacedict2vector(lambda_dict)
        return f

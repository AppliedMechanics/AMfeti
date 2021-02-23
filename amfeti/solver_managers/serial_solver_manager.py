#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Serial solver managers, that set global options, control the global solution-process and call local problems
"""
from .solver_manager_base import SolverManagerBase
from amfeti.coarse_grid import NaturalCoarseGrid
from amfeti.solution import StandardSolution
from scipy.sparse import csr_matrix, issparse
import numpy as np

__all__ = ['SerialSolverManager']


class SerialSolverManager(SolverManagerBase):
    """
    Solver-manager for a serial, not parallelized, solution process. This is recommended for testing- and prototyping-
    purposes or if no MPI-installation is available.

    References
    ----------
    For more theory on FETI solvers, coarse grids and preconditioning see
    [1] C. Farhat and F.X. Roux (1991): A method of Finite Element Tearing and Interconnecting and its parallel solution
        algorithm. International Journal for Numerical Methods in Engineering 32 1205--1227.
    [2] D.J. Rixen and C. Farhat (1999): A simple and efficient extension of a class of substructure based
        preconditioners to heterogeneous structural mechanics problems. International Journal for Numerical Methods in
        Engineering 44 489--516

    Attributes
    ----------
    solver : GlobalSolverBase
        linear iterative solver for the global problem
    _local_problems_dict : LocalProblemBase
        dictionary of local problems
    _coarse_grid : CoarseGridBase
        coarse grid for acceleration and/or stabilization of the solution process (e.g. for semidefinite problems at
        least a NaturalCoarseGrid is needed)
    _solution : SolutionBase
        solution-object, that stores all solutions and solver-information
    _global_dof_dimension : int
        number of overall dual degrees of freedom (e.g. Lagrange-Multipliers)
    _interface2dof_map : dict
        mapping of interface-names to dof-ids in the dual solution-vector
    _config_dict : dict
        configuration
    _lambda_sol : ndarray
        dual solution
    _alpha_sol : ndarray
        kernel solution
    _info_dict : dict
        solver-information
    """
    def __init__(self, local_problems_dict, solver):
        """
        Parameters
        ----------
        local_problems_dict : dict
            all local problem-objects
        solver : GlobalSolverBase
            linear iterative solver for the global problem

        Returns
        -------
        None
        """
        super().__init__(local_problems_dict)
        self._coarse_grid = None
        self._solution = None
        self._interface2dof_map = dict()
        self._global_dof_dimension = 0
        self.solver = solver
        self._config_dict = {'coarse_grid': NaturalCoarseGrid(),
                             'solution': StandardSolution()}
        self._lambda_sol = None
        self._alpha_sol = None
        self._info_dict = dict()

    @property
    def solution(self):
        self._solution.update(self._local_problems_dict, self._lambda_sol, self._alpha_sol, self._info_dict)
        return self._solution

    @property
    def no_lagrange_multiplier(self):
        return self._global_dof_dimension

    def update(self):
        """
        Updates coarse-grid- and solution-object based on the configuration and local problems

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._coarse_grid = self._config_dict['coarse_grid']
        self._solution = self._config_dict['solution']
        BR_dict = dict()
        RTf_dict = dict()
        self._interface2dof_map = dict()
        self._global_dof_dimension = 0

        for problem_id, local_problem in self._local_problems_dict.items():
            BR_int_dict = dict()
            for interface, B in local_problem.B.items():
                BR_int_dict[interface] = np.array([])

            BR_dict[problem_id] = BR_int_dict

            RTf_dict[problem_id] = np.array([])
            
            for interface, B in local_problem.B.items():
                if interface not in self._interface2dof_map:
                    new_global_dof_dimension = self._global_dof_dimension + B.shape[0]
                    self._interface2dof_map[interface] = np.arange(self._global_dof_dimension, new_global_dof_dimension)
                    self._global_dof_dimension = new_global_dof_dimension

        self._coarse_grid.update(BR_dict, RTf_dict, self._interfacedict2vector)

    def solve(self):
        """
        Solves the global linear problem

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        config_dict = {'projection': self._coarse_grid.project,
                  # 'precondition': self._apply_preconditioner}
                   'precondition': self._apply_multi_preconditioner}
                       #  'multiprecondition': self._apply_multi_preconditioner }
        lambda_rigid = self.initialize_lambda()

        self.solver.set_config(config_dict)
        lambda_sol, info_dict = self.solver.solve(self._F_action, self._residual, lambda_rigid)

        alpha_sol = self._coarse_grid.solve(self._F_action(lambda_sol, True))
        alpha_dict = self._coarse_grid.map_vector2localproblem(alpha_sol)

        for problem_id, local_problem in self._local_problems_dict.items():
            local_problem.solve(self._vector2interfacedict(lambda_sol), True, alpha_dict[problem_id])

        # gaps_dict = local_problem.computegaps()
        self._lambda_sol = lambda_sol
        self._alpha_sol = alpha_sol
        self._info_dict.update(info_dict)

    # def computegaps(self):
    #
    #     gaps_dict = {}
    #


        # return  gaps_dict

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
        for problem_id, local_problem in self._local_problems_dict.items():
            local_problem.update_system(self._vector2interfacedict(lambda_sol), update_input_dict)
            if isinstance(local_info_dict, dict):
                local_info_dict.update({problem_id: local_problem.dump_local_information()})
        return local_info_dict

    def initialize_lambda(self):
        """
        Initialization of dual solution based on the coarse grid.

        Parameters
        ----------
        None

        Returns
        -------
        lambda_sol : ndarray
            initialized dual solution
        """
        lambda_sol = self._coarse_grid.solve()
        if lambda_sol is None:
            lambda_sol = np.zeros(self._global_dof_dimension,dtype=complex)
        return lambda_sol

    def _residual(self, v):
        r"""
        FETI-solver residual, often referred to as
        .. math::
            \vec{d}

        Parameters
        ----------
        v : ndarray
            dual solution

        Returns
        -------
        -d : ndarray
            FETI-residual (in structural mechanics viewed as gaps)
        """
        d = self._F_action(v, True)
        return -d

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
        d = np.zeros_like(v, dtype = complex)
        for problem_id, local_problem in self._local_problems_dict.items():
            u_dict_local = local_problem.solve(v_dict, external_force)
            # for interface, u_b in u_dict_local.items():
            #     if interface not in u_dict:
            #         u_dict[interface] = {problem_id: u_b}
            #     else:
            #         u_dict[interface].update({problem_id: u_b})
            d = d+ self._interfacedict2vector(u_dict_local)
        # compute gap
        # gap_dict = {}
        # for interface_id, u_dict_interface in u_dict.items():
        #
        #     if interface_id.ndim == 1:
        #
        #         for problem_id, u_b in u_dict_interface.items():
        #             if interface_id not in gap_dict:
        #                 gap_dict[interface_id] = np.zeros_like(u_b,dtype=complex)
        #                 gap_dict[interface_id] += u_b
        #     else:
        #         for problem_id, u_b in u_dict_interface.items():
        #             if interface_id not in gap_dict:
        #                 for nRHS in range(2):
        #                     gap_dict[interface_id][:,nRHS] = np.zeros_like(u_b,dtype=complex)
        #                     gap_dict[interface_id][:,nRHS] += u_b







        # for problem_id, local_problem in self._local_problems_dict.items():
        #     u_dict_local = local_problem.solve(v_dict, external_force)
        #     for interface, u_b in u_dict_local.items():
        #         if interface not in u_dict:
        #             u_dict[interface] = {problem_id: u_b}
        #         else:
        #             u_dict[interface].update({problem_id: u_b})
        #     # d = d + self._interfacedict2vector(u_dict_local)
        # # compute gap
        # gap_dict = {}
        # for interface_id, u_dict_interface in u_dict.items():
        #     for problem_id, u_b in u_dict_interface.items():
        #         if interface_id not in gap_dict:
        #             gap_dict[interface_id] = np.zeros_like(u_b,dtype=complex)
        #         gap_dict[interface_id] += u_b
        # d2 = self._interfacedict2vector(gap_dict)

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

        f = self._interfacedict2vector(lambda_dict)
        return f






    def _apply_multi_preconditioner(self, v):
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
        f = np.zeros((len(v),self._local_problems_dict.__len__()), dtype= complex)
        icounter = 0
        f_dict_local = {}
        for problem_id, local_problem in self._local_problems_dict.items():
            f_dict_local = local_problem.precondition(v_dict)
            f[:,problem_id-1] = self._interfacedict2vector(f_dict_local)
        return f

    def _interfacedict2vector(self, intdict):
        """
        Assembles interface-quantities to a global vector

        Parameters
        ----------
        intdict : dict
            interface-quantities

        Returns
        -------
        vector : ndarray
            assembled vector
        """
        vector = None
        for interface, value in intdict.items():
            if vector is None:
                if issparse(value):
                    if value.ndim is 1:
                        vector = csr_matrix(self._global_dof_dimension,dtype=complex)
                    else:
                        vector = csr_matrix((self._global_dof_dimension, value.shape[1]), dtype=complex)
                else:
                    if value.ndim is 1:
                        vector = np.zeros(self._global_dof_dimension,dtype=complex)
                    else:
                        vector = np.zeros((self._global_dof_dimension, value.shape[1]), dtype=complex)
            if value.ndim is 1:
                vector[self._interface2dof_map[interface]] = value
            else:
                vector[self._interface2dof_map[interface], :] = value
        return vector

    def _vector2interfacedict(self, v):
        """
        Distribute a global vector to interfaces

        Parameters
        ----------
        v : ndarray
            global vector

        Returns
        -------
        interface_dict : dict
            interface-quantities
        """
        interface_dict = dict()
        for interface, dofs in self._interface2dof_map.items():
            if v.ndim is 1:
                interface_dict[interface] = v[dofs]
            else:
                interface_dict[interface] = v[dofs, :]
        return interface_dict

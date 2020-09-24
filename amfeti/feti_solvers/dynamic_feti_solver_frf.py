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
from amfeti.local_problems.static_local_problems_frf import LinearStaticLocalProblemFRF
from amfeti.solvers import PCPGsolver
from amfeti.preconditioners import DirichletPreconditioner, LumpedPreconditioner
from amfeti.scaling import MultiplicityScaling
from amfeti.nonlinear_solvers import LoadSteppingControl
import numpy as np
import logging
from copy import copy
import scipy
__all__ = ['LinearStaticFetiSolver',
           'NonlinearStaticFetiSolver']


class LinearDynamicFetiSolverFRF(FetiSolverBase):
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
    def __init__(self,K_dict, M_dict, B_dict, f_dict, w_list,  **kwargs):
        """
        Parameters
        ----------
        Z_dict : dict
            dynamic stiffness matrices
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
                         'M_dict': M_dict,
                         'B_dict': B_dict,
                         'f_dict': f_dict,
                         'frequency' : w_list,
                         'use_parallel': False,
                         'scaling': MultiplicityScaling(),
                         'global_solver': PCPGsolver()})
        self.update_preconditioner = True
        self.number_precon_updates = 0
        self.set_config(kwargs)
        self._dual_solution_length = None
        self._create_local_problems()

    def _update_local_problems(self):
        """
         updates the linear static local problems and configures them

         Parameters
         ----------
         None

         Returns
         -------
         None
         """
        Z_dict = self._config_dict['Z_dict']
        for problem_id, Z in self._config_dict['Z_dict'].items():
            self._local_problems[problem_id].update_local_matrices(Z)
            if self.update_preconditioner == True:
                if problem_id == 1:
                   print('updating Prconditioner')
                   self.number_precon_updates = self.number_precon_updates + 1
                self._local_problems[problem_id].update_preconditioner_and_scaling()

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
        K_dict = self._config_dict['K_dict']
        B_dict = self._config_dict['B_dict']
        f_dict = self._config_dict['f_dict']
        for problem_id, K in self._config_dict['K_dict'].items():
            self._local_problems[problem_id] = LinearDynamicLocalProblemFRF(problem_id, K_dict[problem_id], K, B_dict[problem_id],
                                                                        f_dict[problem_id])
            
            self._local_problems[problem_id].set_config({'preconditioner': copy(self._config_dict['preconditioner']),
                                                         'scaling': copy(self._config_dict['scaling'])})
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
        M_dict = self._config_dict['M_dict']
        K_dict = self._config_dict['K_dict']
        solution_dict =  dict()
        buildZ = lambda w, M, K, alpha=0.00005, beta=0.000001: -w ** 2 * M + K + 1J * w * (alpha * K + beta * M)
        def build_Z_dict(w, M_dict, K_dict, alpha=0.0005, beta=0.00001):
            Z_dict = {}
            for key, K in K_dict.items():
                Z_dict[key] = buildZ(w, M_dict[key], K, alpha, beta)
            return Z_dict
        w_list = self._config_dict['frequency']
        for i in range(2 * len(w_list)):
            w = w_list[i]
            Z_dict_ = build_Z_dict(w, M_dict, K_dict)
            self.set_config({'Z_dict': Z_dict_})
            self._update_local_problems()
            self._solver_manager.update()
            self._solver_manager.solve()
            solution_dict[w] = self._solver_manager.solution
            print("Frequency = %d : GMRES iteration %d"
                  %( w,solution_dict[w].solver_information['GMRES_iterations'] ))

            if solution_dict[w].solver_information['GMRES_iterations'] <190:
                self.update_preconditioner = False
            else:
                self.update_preconditioner = True
            if w == w_list[-1]:
                break
        return solution_dict
    
 

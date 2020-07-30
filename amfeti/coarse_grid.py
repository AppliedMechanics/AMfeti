#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Coarse problem module of AMfeti, that provides classes and methods to manage and solve operations on a global
coarse-grid, such as the natural coarse-grid resulting from the local problems' nullspaces
"""
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, hstack, eye
from copy import copy
from amfeti.config_base import ConfigBase
import logging

__all__ = ['NaturalCoarseGrid']


class CoarseGridBase(ConfigBase):
    def __init__(self):
        super().__init__()

    def update(self, *args):
        pass

    def solve(self):
        pass


class NaturalCoarseGrid(CoarseGridBase):
    """
    Natural coarse grid based on the null-spaces of local problems

    Attributes
    ----------
    G : ndarray
        globally assembled nullspace-operator
    e : ndarray
        globally assembled portion of local right-hand-sides in the local nullspaces
    G_dict : dict
        dictionary of local nullspace-operators
    e_dict : dict
        dictionary of local portion of local right-hand-sides in the local nullspaces
    """
    def __init__(self):
        super().__init__()
        self.G = None
        self.e = None
        self._projector = None
        self._dimension_interface = None
        self._config_dict = {'solution_method': 'spsolve'}
        self._vector2localproblem_map = {}

    def update(self, BR_dict, RTf_dict, interfacedict2vector):
        """
        Update the coarse-grid

        Parameters
        ----------
        B_dict : dict
            local connectivity matrices, that map the interface-solutions to the local degrees of freedom
        R_dict : dict
            local nullspaces, that map local degrees of freedom into the nullspace
        f_dict : dict
            local right-hand-sides
        interfacedict2vector : callable
            mapping from interface-dictionary to global degrees of freedom

        Returns
        -------
        None
        """
        logger = logging.getLogger(__name__)
        self._projector = None
        self.G = None
        self.e = None
        interfacedof_end = 0
        for problem_id, BR_int_dict in BR_dict.items():
            if RTf_dict[problem_id].size == 0:
                self._vector2localproblem_map[problem_id] = np.array([])
            else:
                G_prob = interfacedict2vector(BR_int_dict)
                e_prob = RTf_dict[problem_id]
                if self.G is None:
                    self.G = csr_matrix(G_prob)
                    self.e = e_prob
                    interfacedof_end = self.G.shape[1]
                    self._vector2localproblem_map[problem_id] = np.arange(interfacedof_end)
                else:
                    self.G = hstack((self.G, csr_matrix(G_prob)))
                    self.e = np.append(self.e, e_prob)
                    interfacedof_start = copy(interfacedof_end)
                    interfacedof_end += G_prob.shape[1]
                    self._vector2localproblem_map[problem_id] = np.arange(interfacedof_start, interfacedof_end)
        if self.G is not None:
            self._dimension_interface = self.G.shape[0]

    def solve(self, v=None):
        """
        Solver for coarse problem

        Parameters
        ----------
        v : ndarray
            righ-hand-side of coarse problem

        Returns
        -------
        coarse_solution : ndarray
            solution of coarse problem
        """
        if self.G is None:
            coarse_solution = v
        else:
            reproject = False
            GTG = self.G.T @ self.G
            if v is None:
                v = self.e
                reproject = True
            else:
                v = self.G.T @ v

            if self._config_dict['solution_method'] == 'spsolve':
                coarse_solution = spsolve(GTG, v)
                if reproject:
                    coarse_solution = self.G @ coarse_solution
            else:
                raise ValueError('Specified solution method, ' + self._config_dict['solution_method'] +
                                 ', for coarse grid unknown.')

        return coarse_solution

    def project(self, v):
        """
        Removes those portions from a vector, that are in the nullspaces and returns the remaining parts

        Parameters
        ----------
        v : ndarray
            vector that shall be projected

        Returns
        -------
        v_proj : ndarray
            projected vector
        """
        if self._dimension_interface is None:
            return v
        else:
            if self._projector is None:
                self._projector = eye(self._dimension_interface)
                if self.G is not None:
                    self._projector -= self.G @ self.solve(eye(self._dimension_interface))
            return self._projector @ v

    def map_vector2localproblem(self, v):
        v_dict = dict()
        for local_problem_id, dofmap in self._vector2localproblem_map.items():
            if dofmap.size is 0:
                v_dict[local_problem_id] = np.array([])
            else:
                v_dict[local_problem_id] = v[dofmap]
        return v_dict

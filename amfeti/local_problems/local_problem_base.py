#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

from amfeti.config_base import ConfigBase
import logging


class LocalProblemBase(ConfigBase):
    """
    Base-class for local problems.

    Attributes
    ----------
    id : int, string
        global id of the local problem
    scaling : ScalingBase
        Scaling, that adds to the preconditioner
    preconditioner : PreconditionerBase
        Preconditioner, that inverts the local solve-method
    """
    def __init__(self, global_id):
        """
        Parameters
        ----------
        global_id : int, string
            global id of the local problem
        """
        self.id = global_id
        self.scaling = None
        self.preconditioner = None
        super().__init__()

    def solve(self, external_solution):
        """
        Local solve-method, that solves the local problem based on the passed external solution. The external solution
        is determined by the global solver and passed to the local problem. After the local solves are performed, the
        local boundary-solutions are returned.

        Parameters
        ----------
        external_solution : dict
            external solution, that acts on the local problem. For example Lagrange-Multipliers in case of a FETI-solver

        Returns
        -------
        q_b : dict
            solutions on the local problem's boundary
        """
        q_b = external_solution
        return q_b

    def precondition(self, q_b):
        """
        Preconditioning-method, that takes predefined boundary-solutions of the local problem. From these boundary-
        solutions the external solutions are determined, that would generate these boundary-solutions.

        Parameters
        ----------
        q_b : dict
            solutions on the local problem's boundary

        Returns
        -------
        external_solution : dict
            external solution, that acts on the local problem. For example Lagrange-Multipliers in case of a FETI-solver
        """
        external_solution = q_b
        return external_solution

    def update_system(self, external_solution_dict, update_input_dict=None):
        """
        Updates local states with given external solution and optionally some further input, e.g. a load-factor.

        Parameters
        ----------
        external_solution_dict : dict
            external interface-solutions
        update_input_dict : dict
            optional further input, e.g. load-factor

        Returns
        -------
        None
        """
        logger = logging.getLogger(__name__)
        logger.warning('Local update-method called, but no updating is implemented.')

    def dump_local_information(self):
        """
        Returns some local information, such as the current residual.

        Parameters
        ----------
        None

        Returns
        -------
        info_dict : dict
            some local information
        """
        info_dict = dict()
        return info_dict

#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Scaling module for the preconditioning-step of AMfeti
"""
import numpy as np
from scipy.sparse import diags

__all__ = ['MultiplicityScaling',
           'SquarerootMultiplicityScaling']


class ScalingBase:
    """
    Base-class for scalings, used during preconditioning

    Attributes
    ----------
    scaling_dict : dict
        scaling-matrices for each interface
    """
    def __init__(self):
        """
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.scaling_dict = dict()

    def update(self, *args):
        pass

    def apply(self, gap_dict):
        """
        Application methods, that multiplies the scaling to each interface-gap or interface-force

        Parameters
        ----------
        gap_dict : dict
            interface-gaps or -forces

        Returns
        -------
        gap_dict : dict
            scaled interface-gaps or -forces
        """
        for interface, scaling in self.scaling_dict.items():
            gap_dict[interface] = scaling @ gap_dict[interface]
        return gap_dict


class MultiplicityScaling(ScalingBase):
    """
    Multiplicity-Scaling, that averages interface-quantities based on the number of gaps belonging to each local
    degree of freedom

    References
    ----------
       [1]  C. Farhat and D.J. Rixen (1999): A simple and efficient extension of a class of substructure based
       preconditioners to heterogeneous structural mechanics problems. International Journal for Numerical Methods in
       Engineering 44 489--516.
    """
    def __init__(self):
        super().__init__()

    def update(self, B_dict):
        """
        Updating-method for Multiplicity-scaling, that extracts the multiplicity-factors for each interface-gap from the
        local B-matrices

        Parameters
        ----------
        B_dict : dict
            B-matrices of each interface

        Returns
        -------
        None
        """
        self.scaling_dict = dict()
        scaling_values = None
        for key, B in B_dict.items():
            if scaling_values is None:
                scaling_values = np.ones(B.shape[1])

            scaling_values[B.indices] += 1

        for key, B in B_dict.items():
            self.scaling_dict[key] = diags(1.0 / scaling_values[B.indices])

class SquarerootMultiplicityScaling(ScalingBase):
    """
    Multiplicity-Scaling, that scales the interface-quantities based on the square root number of gaps belonging 
    to each local degree of freedom    
    """
    def __init__(self):
        super().__init__()

    def update(self, B_dict):
        """
        Updating-method for Multiplicity-scaling, that extracts the multiplicity-factors for each interface-gap from the
        local B-matrices

        Parameters
        ----------
        B_dict : dict
            B-matrices of each interface

        Returns
        -------
        None
        """
        self.scaling_dict = dict()
        scaling_values = None
        for key, B in B_dict.items():
            if scaling_values is None:
                scaling_values = np.ones(B.shape[1])

            scaling_values[B.indices] += 1

        for key, B in B_dict.items():
            self.scaling_dict[key] = diags(1.0 / np.sqrt(scaling_values[B.indices]))
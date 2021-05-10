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


__all__ = ['MultiplicityScaling', 'Klumpedscaling']


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
        self.stiffness_k_dict = dict()

    def update(self, *args):
        pass


    def local_stiffness_interface_compute(self, *args):

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

#
class Klumpedscaling(ScalingBase):
    """
    K-Scaling, that averages interface-quantities based on the local stiffness of all the different subdomains it is connected to

    References
    ----------
       [1]  C. Farhat and D.J. Rixen (1999): A simple and efficient extension of a class of substructure based
       preconditioners to heterogeneous structural mechanics problems. International Journal for Numerical Methods in
       Engineering 44 489--516.
    """
    def __init__(self):
        super().__init__()



    def update(self, B_dict,K_dict, local_problem_id):
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




        scaling_dict = dict()
        local_stiffness_k_dict = self.stiffness_k_dict[local_problem_id]

        for interface_id, B_local in B_dict[local_problem_id].items():


            scaling_values_numerator = local_stiffness_k_dict[interface_id]
            scaling_values_denominator = local_stiffness_k_dict[interface_id]

            # You run the loop over all the subomdains to collect stiffness
            for problem_id, local_problem in B_dict.items():

            # you make sure you do not append stiffness of the current subdomain again!
                if problem_id == local_problem_id:
                       print('only collect stiffness from neighbours')
                else:
                    for interface_id_neighbour, B_local_interface_neighbour in B_dict[problem_id].items():
                        if interface_id  == interface_id_neighbour:
                            scaling_values_denominator = scaling_values_denominator+ self.stiffness_k_dict[problem_id][interface_id_neighbour]

            scaling_dict[interface_id] = diags(scaling_values_numerator/scaling_values_denominator)


        return scaling_dict



    def local_stiffness_interface_compute(self,B_dict,  K_dict):

        self.stiffness_k_dict = dict()

        for local_problem, B_dict_local in B_dict.items():
            stiffness_k_dict_local = dict()
            for interface_id, B_dict_local_interface in B_dict_local.items():
                stiffness_k_dict_local[interface_id] = K_dict[local_problem].data[[B_dict_local_interface.indices]]



            self.stiffness_k_dict[local_problem] = stiffness_k_dict_local








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
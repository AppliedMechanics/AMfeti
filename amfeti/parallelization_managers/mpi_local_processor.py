#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Local MPI processor, that performs communication between MPI-nodes

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
WARNING: no MPI-scripts may be imported before the shell-script containing the mpiexec-command is executed. Otherwise
the caller of the shell-script crashes!
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

import numpy as np
#from amfeti.config_base import ConfigBase
from amfeti.parallelization_managers.tools import load_object, save_object
import sys
import os
import logging

from mpi4py import MPI


__all__ = ['MPILocalProcessor']

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def send_info(local_var, rank_id, tag):
    """
    Method, that exchanges information with a specified MPI-rank

    Parameters
    ----------
    local_var : ndarray, dict, int, double, list, str, tuple
        object, that shall be sent to the specified MPI-rank

    rank_id : int
        unique ID of the MPI-rank, with which information shall be exchanged

    tag : str, int, tuple
        unique tag, that ensures exchange of correct messages

    Returns
    -------
    var_nei : ndarray, dict, int, double, list, str, tuple
        object, that is received from the other MPI-rank and is of the same type, as the sent object
    """
    if isinstance(local_var, np.ndarray):
        # sending message to neighbors
        comm.Send(local_var, dest=rank_id, tag=tag)
    else:
        # sending message to neighbors
        comm.send(local_var, dest=rank_id, tag=tag)


def receive_info(local_var, rank_id, tag):
    if isinstance(local_var, np.ndarray):
        # receiving messages from neighbors
        var_nei = np.empty(local_var.shape)
        comm.Recv(var_nei, source=rank_id, tag=tag)
    else:
        # receiving messages from neighbors
        var_nei = comm.recv(source=rank_id, tag=tag)
    return var_nei


class MPILocalProcessor:
    """
    Local processor, that handles all communication-tasks between MPI-ranks.

    Attributes
    ----------
    _config_dict : dict
        dictionary with all configuration-information

    _interface2rank_map : dict
        map of an interface-name to a MPI-rank, that has that interface-name as well

    _problem2rank_map : dict
        map of a local-problem-id to a MPI-rank
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
        self._config_dict = {'interface2rank_map': None,
                             'problem2rank_map': None,
                             'total_ranks_list': []}
        self._interface2rank_map = None
        self._problem2rank_map = None

    @staticmethod
    def _convert_to_integer(input):
        new_integer = 0
        if isinstance(input, str):
            new_integer = sum([ord(c) for c in input])
        elif isinstance(input, int):
            new_integer = input
        else:
            raise ValueError('Datatype ', type(input), ' not supported for conversion to integer')

        return new_integer

    def set_config(self, new_config_dict):
        """
        Updates the configuration dictionary

        Parameters
        ----------
        new_config_dict : dict
            dictionary of new configuration-parameters

        Returns
        -------
        None
        """
        if isinstance(new_config_dict, dict):
            self._config_dict.update(new_config_dict)

    def update(self):
        """
        Updating rank-maps

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._interface2rank_map = self._config_dict['interface2rank_map']
        self._problem2rank_map = self._config_dict['problem2rank_map']

    def send_interface_info(self, local_var, interface):
        """
        Method, that sends information to MPI-ranks based on a given interface-name. Objects are sent only
        to ranks, that have the specified interface-name.

        Parameters
        ----------
        local_var : ndarray, dict, int, double, list, str, tuple
            object, that shall be sent to the specified MPI-rank

        interface : str, int, tuple
            interface-name

        Returns
        -------
        None
        """
        if interface in self._interface2rank_map:
            if isinstance(interface, tuple):
                tag = 0
                for entry in interface:
                    tag += self._convert_to_integer(entry)
            else:
                tag = self._convert_to_integer(interface)

            send_info(local_var, self._interface2rank_map[interface], tag)

    def receive_interface_info(self, local_var, interface):
        """
        Method, that receives information from MPI-ranks based on a given interface-name. Objects are received only
        from ranks, that have the specified interface-name.

        Parameters
        ----------
        local_var : ndarray, dict, int, double, list, str, tuple
            object, that was sent to the specified MPI-rank

        interface : str, int, tuple
            interface-name

        Returns
        -------
        var_neighbor : ndarray, dict, int, double, list, str, tuple
            object, that is received from the other MPI-rank and is of the same type, as the sent object
        """
        if interface in self._interface2rank_map:
            if isinstance(interface, tuple):
                tag = 0
                for entry in interface:
                    tag += self._convert_to_integer(entry)
            else:
                tag = self._convert_to_integer(interface)

            var_neighbor = receive_info(local_var, self._interface2rank_map[interface], tag)
        else:
            var_neighbor = np.array([])

        return var_neighbor

    def send_global_info(self, local_var):
        """
        Method, that sends information to all other MPI-ranks.

        Parameters
        ----------
        local_var : dict
            object, that shall be sent to all other MPI-ranks

        Returns
        -------
        None
        """
        for addressed_rank in self._config_dict['total_ranks_list']:
            if addressed_rank != rank:
                tag = rank + addressed_rank
                send_info(local_var, addressed_rank, tag)

    def receive_global_info(self, local_var):
        """
        Method, that receives information from all other MPI-ranks.

        Parameters
        ----------
        local_var : dict
            object, that was sent to all other MPI-ranks

        Returns
        -------
        var_neighbor : dict
            object, that is received from all other MPI-ranks
        """
        var_neighbor = dict()
        for addressed_rank in self._config_dict['total_ranks_list']:
            if addressed_rank != rank:
                tag = rank + addressed_rank
                var_neighbor.update(receive_info(local_var, addressed_rank, tag))
        return var_neighbor


if __name__ == "__main__":
    """
    Basic Python-script, that is executed by the MPI-script on each MPI-rank. Configuration of this script is either set 
    by the loaded local-solver-manager object or by keyword-arguments in the MPI-script.
    
    DEVELOPER-HINT: 
    Inside this script print-exports won't be passed to terminal. If diagnosis-information is needed, use the 
    logging-framework instead.
    """
    logging.basicConfig(filename='mpi.log', level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info('Running local MPI program...')
    system_argument = sys.argv

    if len(system_argument) > 1:
        mpi_kwargs = {}
        for arg in system_argument:
            try:
                var, value = arg.split('=')
                try:
                    mpi_kwargs[var] = int(value)
                except:
                    mpi_kwargs[var] = value
            except:
                logger.debug('Command line argument not understood, arg = %s cannot be splitted in variable name + '
                             'value' % arg)
                pass

        logger.info('########################################')
        logger.info('MPI rank %i' % rank)
        logger.info('Directory pass to MPI solver = %s' % os.getcwd())
        logger.info('########################################')
        case_path = mpi_kwargs['prefix']
        case_path += str(rank) + mpi_kwargs['ext']
        logger.info('Local object name passed to MPI solver = %s' % case_path)

        local_solver_manager = load_object(case_path)
        logger.debug('Local solver-manager loaded')
        parallel_processor = MPILocalProcessor()
        local_solver_manager.update(parallel_processor)
        local_solver_manager.solve()
        solution = local_solver_manager.solution
        solution_path = mpi_kwargs['solution'] + str(rank) + '_solution' + mpi_kwargs['ext']
        save_object(solution, solution_path)
        save_object(local_solver_manager, case_path)
        logger.info('########################################')
    else:
        logger.warning('\n WARNING. No system argument were passed to the MPIsolver. Nothing to do! n')

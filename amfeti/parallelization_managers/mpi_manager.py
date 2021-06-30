#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
MPI-manager module for parallelization
"""
import os
import logging
import subprocess
import shutil
import numpy as np
from copy import copy, deepcopy
from .parallelization_manager_base import ParallelizationManagerBase
from amfeti.solver_managers.local_solver_manager import LocalSolverManager
from amfeti.tools import invert_dictionary_with_iterables, amfeti_dir
from amfeti.config_base import ConfigBase
from .tools import load_object, run_shell_script


__all__ = ['MPIManager']


class MPIManager(ParallelizationManagerBase):
    """
    MPI-manager, that sets up and configures the MPI-process

    Attributes
    ----------
    _python_file : str
        name of Python-module, that shall be executed by each MPI-rank

    _file_extension : str
        file-type of serialized objects

    _path_prefix : str
        prefix for serialized local files, that are loaded into each MPI-rank

    _rank_path : dict
        absolute paths to each seralized file, that belongs to a MPI-rank
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs : dict
            Additional MPI-arguments handed over as keyword-arguments

        Returns
        -------
        None
        """
        super().__init__()

        mpi_exec = 'mpiexec'
        try:
            mpi_path = os.environ['MPIDIR']
            mpi_exec = os.path.join(mpi_path, mpi_exec).replace('"', '')
        except:
            logger = logging.getLogger(__name__)
            logger.warning("Warning! Using mpiexec in global path")

        try:
            python_path = os.environ['PYTHON_ENV']
            python_exec = os.path.join(python_path, 'python').replace('"', '')
        except:
            logger = logging.getLogger(__name__)
            logger.warning("Warning! Using python in global path")
            python_exec = 'python'

        self.set_config({'write_log': True,
                         'mpi_exec': mpi_exec,
                         'mpi_args': '',
                         'mpi_size': None,
                         'mpi_rank2problems': None,
                         'python_exec': python_exec,
                         'additional_mpi_args': kwargs,
                         'solution_path': None})

        self._python_file = 'mpi_local_processor.py'

        self._file_extension = '.pkl'
        self._path_prefix = 'mpi_rank_'
        self._rank_path = dict()

    def read_solutions(self):
        """
        Loads solutions from serialized solution-files

        Parameters
        ----------
        None

        Returns
        -------
        solution : SolutionBase
            solution-object
        """
        logger = logging.getLogger(__name__)
        solution = dict()
        for rank_id in self._config_dict['mpi_rank2problems'].keys():
            path = os.path.join(self._config_dict['solution_path'], self._path_prefix) + str(rank_id) + '_solution' + \
                   self._file_extension
            solution[rank_id] = load_object(path)
        logger.debug('Solution read')
        return solution

    def load_local_problems(self):
        """
        Loads serialized local problems

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        local_problems_dict = dict()
        for rank_id in self._config_dict['mpi_rank2problems'].keys():
            path = os.path.join(self._config_dict['tmp_folder_absolute'], self._path_prefix) + str(rank_id) + \
                   self._file_extension
            local_solver_manager = load_object(path)
            for problem_id, problem in local_solver_manager._local_problems_dict.items():
                local_problems_dict[problem_id] = problem

        return local_problems_dict

    def set_up_parallel_process(self, local_problems_dict, solver, local_solver_manager_config):
        """
        Sets up the parallel solution-process by serializing local problems and configuring local solver-managers.

        Parameters
        ----------
        local_problems_dict : dict
            local problems

        solver : GlobalSolverBase
            iterative solver for the interface-problem

        local_solver_manager_config : dict
            configuration of the local solver-manager

        Returns
        -------
        None
        """
        self._create_temporary_folder()

        if self._config_dict['mpi_rank2problems'] is None:
            rank2problems = dict()
            rank = 0
            for problem_id in local_problems_dict.keys():
                if rank not in rank2problems:
                    rank2problems[rank] = np.array([problem_id])
                    rank += 1
                else:
                    rank2problems[rank] = np.append(rank2problems[rank], problem_id)
            self.set_config({'mpi_rank2problems': rank2problems})

        self.set_config({'mpi_size': len(list(self._config_dict['mpi_rank2problems'].keys()))})

        problem2rank_map = dict()
        for rank, problems in self._config_dict['mpi_rank2problems'].items():
            for problem_id in problems:
                problem2rank_map[problem_id] = rank

        local_solver_manager_config.update({'parallel_processor': 'MPILocalProcessor',
                                            'local_problems_full': list(local_problems_dict.keys())})

        command = self._create_command_string()
        self._create_launcher_script('run_mpi', command)

        for rank_id, local_problems_in_rank in self._config_dict['mpi_rank2problems'].items():
            local_problems_ranksubset = {problem_id: local_problems_dict[problem_id] for problem_id in local_problems_in_rank}
            curr_local_solver_manager_config = copy(local_solver_manager_config)

            secondary_problems = ()
            interface2rank_map_subset = dict()
            for problem_id, local_problem in local_problems_ranksubset.items():
                local_interfaces = local_problem.interfaces
                for other_id, other_problem in local_problems_dict.items():
                    if other_id not in local_problems_ranksubset.keys():
                        for interface in local_interfaces:
                            if interface in other_problem.interfaces:
                                interface2rank_map_subset[interface] = problem2rank_map[other_id]
                                break
                        if other_id not in secondary_problems:
                            secondary_problems += (other_id,)

            local_problem2rank_map = copy(problem2rank_map)
            for problem_id in local_problems_dict.keys():
                if problem_id in local_problems_in_rank:
                    del (local_problem2rank_map[problem_id])

            curr_local_solver_manager_config['parallel_processor_opt'] = {'interface2rank_map': interface2rank_map_subset,
                                                                          'problem2rank_map': local_problem2rank_map,
                                                                          'total_ranks_list': list(self._config_dict['mpi_rank2problems'].keys())}
            curr_local_solver_manager_config['secondary_problems'] = secondary_problems

            local_solver_manager = LocalSolverManager(local_problems_ranksubset, solver)

            local_solver_manager.set_config(copy(curr_local_solver_manager_config))
            local_name = self._path_prefix + str(rank_id) + self._file_extension
            self._rank_path[rank_id] = self._create_serialized_file(local_solver_manager, local_name)

    def launch_parallel_process(self):
        """
        Run the shell-script

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logger = logging.getLogger(__name__)
        if not os.path.exists(self._launcher_script_path):
            raise FileNotFoundError('File path %s does not exist.' % self._launcher_script_path)

        # executing script
        try:
            os.chdir(self._config_dict['tmp_folder_absolute'])
            os.chmod(self._launcher_script_path, 0o775)

            logger.info('Run directory = %s' % os.getcwd())
            logger.info('######################################################################')

            run_shell_script('run_mpi.sh')
            os.chdir(self._local_folder)
        except:
            raise RuntimeError('Error during the simulation.')

    def _create_command_string(self):
        """
        Creates string, that is written in the shell-script

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        python_file_path = os.path.join(amfeti_dir('parallelization_managers'), self._python_file)

        command_list = ['"' + self._config_dict['mpi_exec'] + '"',
                        self._config_dict['mpi_args'],
                        '-n',
                        str(self._config_dict['mpi_size']),
                        '"' + self._config_dict['python_exec'] + '"',
                        '"' + python_file_path + '"']

        if self._config_dict['solution_path'] is None:
            self._config_dict['solution_path'] = copy(self._config_dict['tmp_folder_absolute'])

        command = ' '.join(command_list)
        command += '  "' + 'prefix' + '=' + os.path.join(self._config_dict['tmp_folder_absolute'], self._path_prefix) + '" '#os.path.join(self._config_dict['tmp_folder'], self._path_prefix)
        command += '  "' + 'ext' + '=' + self._file_extension + '" '
        command += '  "' + 'solution' + '=' + os.path.join(self._config_dict['solution_path'], self._path_prefix) + '" '

        for key, value in self._config_dict['additional_mpi_args'].items():
            command += '  "' + str(key) + '=' + str(value) + '" '

        # export results to a log file called amfeti_solver.log
        if self._config_dict['write_log']:
            command += '>mpi.log'

        logger = logging.getLogger(__name__)
        logger.info('######################################################################')
        logger.info('###################### SOLVER INFO ###################################')
        logger.info('MPI exec path = %s' % self._config_dict['mpi_exec'])
        logger.info('Python exec path = %s' % self._config_dict['python_exec'])

        return command

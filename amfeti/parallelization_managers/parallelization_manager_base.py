#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Basic API of a parallelization manager
"""
from amfeti.config_base import ConfigBase
from .tools import save_object, get_platform
import logging
import os
import sys
import shutil

__all__ = ['ParallelizationManagerBase']


class ParallelizationManagerBase(ConfigBase):
    """
    Basic class for parallelization managers, that set up and run the parallel process and collect data afterwards.

    Attributes
    ----------
    _launcher_script_path : str
        absolute path to shell-script
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
        super().__init__()
        self.set_config({'tmp_folder': 'tmp',
                         'tmp_folder_absolute': None,
                         'remove_tmp': False})
        self._launcher_script_path = None
        self._local_folder = None

    def remove_folder(self):
        """
        Remove temporary folder with all files in it

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logger = logging.getLogger(__name__)
        try:
            if self._config_dict['remove_tmp']:
                shutil.rmtree(self._config_dict['tmp_folder_absolute'], ignore_errors=True)
                self._config_dict['tmp_folder_absolute'] = None
            logger.debug('Tmp-folder: ' + str(self._config_dict['tmp_folder_absolute']))
            logger.debug('Remove-flag: ' + str(self._config_dict['remove_tmp']))
        except:
            logger.warning('Could not remove the folder = %s' % (self._config_dict['tmp_folder_absolute']))

    def _create_temporary_folder(self):
        logger = logging.getLogger(__name__)
        if self._config_dict['tmp_folder_absolute'] is None:
            if self._local_folder is None:
                self._local_folder = os.getcwd()
            else:
                os.chdir(self._local_folder)
            if not os.path.exists(self._config_dict['tmp_folder']):
                os.mkdir(self._config_dict['tmp_folder'])
            self._config_dict['tmp_folder_absolute'] = os.path.join(self._local_folder, self._config_dict['tmp_folder'])
            logger.debug('Tmp-folder: ' + str(self._config_dict['tmp_folder_absolute']))
        else:
            os.chdir(self._local_folder)

    def _create_launcher_script(self, script_name, command_string):
        os_script_name = None
        header_string = None
        platform = get_platform()

        if platform == 'Windows':
            os_script_name = script_name + '.bat'
            header_string = 'rem Windows bat file'
        elif platform == 'Linux':
            os_script_name = script_name + '.sh'
            header_string = '#!/bin/sh'

        command = header_string + '\n'
        command += command_string

        # writing bat file with the command line
        os.chdir(self._config_dict['tmp_folder_absolute'])
        #os.chmod(self._config_dict['tmp_folder_absolute'], 0o775)

        with open(os_script_name, 'w+') as run_file:
            run_file.write(command)
        run_file.close()

        self._launcher_script_path = os.path.join(os.getcwd(), os_script_name)

        #os.chdir(self._local_folder)

    def _create_serialized_file(self, object, name):
        #os.chmod(self._config_dict['tmp_folder_absolute'], 0o775)
        local_path = os.path.join(self._config_dict['tmp_folder_absolute'], name)
        save_object(object, local_path)
        return local_path

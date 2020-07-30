#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Tools for handling parallel processes
"""
from dill import load, dump, detect, HIGHEST_PROTOCOL
import subprocess
import logging
import time
import os
import sys


__all__ = ['save_object',
           'load_object']


def save_object(obj, filename, tries=2, sleep_delay=3):
    """
    Serialize an object

    Parameters
    ----------
    obj : object
        object, that shall be serialized and stored

    filename : str
        absolute path, name and extension of the created file

    tries : int
        number of attempts to store the file

    sleep_delay : int
        time, the method shall wait before the next attempt, when writing the file didn't work

    Returns
    -------
    None
    """
    filename = r"{}".format(filename)
    file_written = False
    for i in range(tries):
        try:
            output = open(filename, "wb")
            dump(obj, output, HIGHEST_PROTOCOL)
            output.close()
            file_written = True
            break
        except:
            time.sleep(sleep_delay)
            continue
    if not file_written:
        logger = logging.getLogger(__name__)
        logger.warning(detect.badobjects(obj))
        raise RuntimeError('Unable to write file %s.' % filename)


def load_object(filename, tries=3, sleep_delay=5):
    """
    Loads a serialized object

    Parameters
    ----------
    filename : str
        absolute path, name and extension of the serialized file

    tries : int
        number of attempts to read the file

    sleep_delay : int
        time, the method shall wait before the next attempt, when reading the file didn't work

    Returns
    -------
    obj : object
        object, that was read from the serialized file
    """
    filename = r"{}".format(filename)
    obj = None
    logger = logging.getLogger(__name__)
    if not os.path.exists(filename):
        error_msg = FileNotFoundError('File path %s does not exist.' % filename)
        logger.error(error_msg)
        raise error_msg
    else:
        for i in range(tries):
            try:
                input = open(filename, 'rb')
                obj = load(input)
                break
            except:
                time.sleep(sleep_delay)
                continue
        if obj is None:
            error_msg = RuntimeError('Could not read file %s ' % filename)
            logger.error(error_msg)
            raise error_msg
    return obj


def run_shell_script(script_name):
    """
    Method that runs an external shell-script

    Parameters
    ----------
    script_name : str
        absolute path, name and extension of the shell-script, that shall be run

    Returns
    -------
    None
    """
    platform = get_platform()
    if platform is 'Linux':
        subprocess.run(['./' + script_name], shell=True, check=True)
    elif platform is 'Windows':
        subprocess.run([script_name], shell=True, check=True)


def get_platform():
    """
    Determines the platform, the program is currently executed on.

    Parameters
    ----------
    None

    Returns
    -------
    platform : str
        platform-name
    """
    if sys.platform[:3] == 'win':
        return 'Windows'
    elif sys.platform[:3] == 'lin':
        return 'Linux'
    else:
        raise ValueError('Platform %s is not supported  ' % sys.platform)

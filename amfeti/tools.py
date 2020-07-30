#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
import numpy as np
from os.path import splitext, isfile, join, dirname


__all__ = ['invert_dictionary',
           'invert_dictionary_with_iterables',
           'amfeti_dir']


def invert_dictionary(dict_map):
    """
    Invert a dictionary-mapping such that values point to keys.

    Parameters
    ----------
    dict_map : dict
        dictionary, that shall be inverted

    Returns
    -------
    dict_map_inv : dict
        inverted dictionary
    """
    def add_new_value_to_key(dictionary, key, value):
        if key in dictionary:
            if not isinstance(dictionary[key], list):
                dictionary[key] = [dictionary[key]]
            dictionary[key].append(value)
        else:
            dictionary[key] = value
        return dictionary

    dict_map_inv = dict()
    for k, v in dict_map.items():
        dict_map_inv = add_new_value_to_key(dict_map_inv, v, k)

    return dict_map_inv


def invert_dictionary_with_iterables(dict_map):
    """
    Invert a dictionary-mapping such that values point to keys. Values may only be iterables and the new keys are the
    iterables' entries.

    Parameters
    ----------
    dict_map : dict
        dictionary, that shall be inverted

    Returns
    -------
    dict_map_inv : dict
        inverted dictionary
    """

    def add_new_value_to_key(dictionary, key, value, value_type=None):
        print(value_type)
        if value_type not in (np.ndarray, tuple, list, str) and value_type is not None:
            raise ValueError('Unknown type of value in dictionary, when inverting dictionary.')

        if key in dictionary:
            if isinstance(dictionary[key], np.ndarray):
                dictionary[key] = np.append(dictionary[key], np.array([value], dtype=object))
            elif isinstance(dictionary[key], tuple) or isinstance(dictionary[key], str):
                dictionary[key] += (value,)
            elif isinstance(dictionary[key], list):
                dictionary[key].append(value)
            else:
                dictionary[key] = value
        else:
            if value_type is np.ndarray:
                dictionary[key] = np.array([value], dtype=object)
            elif value_type is tuple or isinstance(value, str):
                dictionary[key] = (value,)
            elif value_type is list:
                dictionary[key] = [value]
            else:
                dictionary[key] = value
        return dictionary

    dict_map_inv = dict()
    for k, v in dict_map.items():
        for vi in v:
            dict_map_inv = add_new_value_to_key(dict_map_inv, vi, k, type(v))

    return dict_map_inv


def amfeti_dir(filename=''):
    '''
    Return the absolute path of the filename given relative to the amfeti directory.

    Parameters
    ----------
    filename : string, optional
        relative path to something inside the amfeti directory.

    Returns
    -------
    dir : string
        string of the filename inside the amfeti-directory. Default value is '', so the amfeti-directory is returned.
    '''

    amfeti_abs_path = dirname(__file__)
    return join(amfeti_abs_path, filename.lstrip('/'))

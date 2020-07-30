#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

__all__ = ['ConfigBase']


class ConfigBase:
    """
    Base-class for all classes, that have an updateable configuration-dictionary.

    Attributes
    ----------
    _config_dict : dict
        dictionary with all configuration-information
    """
    def __init__(self):
        """
        Parameters
        ----------
        None
        """
        self._config_dict = dict()

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
        Updates properties, that are dependent of the config-dict.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError('Update-method for properties, that depend of the configuration, is not implemented.')

"""
Test for config_base module of AMfeti
"""

from unittest import TestCase
from amfeti.config_base import ConfigBase
from ..tools import CustomDictAssertTest
import numpy as np


class ConfigBaseTest(TestCase):
    def setUp(self):
        self.config_obj = ConfigBase()
        self.custom_asserter = CustomDictAssertTest()

    def test_set_config(self):
        config_desired = {'key1': 5,
                          'key2': np.random.rand(6),
                          'key3': None}

        self.config_obj.set_config(config_desired)

        self.custom_asserter.assert_dict_almost_equal(self.config_obj._config_dict, config_desired)

        self.config_obj._config_dict = {'key1': 8,
                                        'key4': {'subkey1': 3,
                                                 'subkey2': None}}

        config_desired_2 = {'key1': 5,
                          'key2': config_desired['key2'],
                          'key3': None,
                          'key4': {'subkey1': 3,
                                   'subkey2': None}}

        self.config_obj.set_config(config_desired)

        self.custom_asserter.assert_dict_almost_equal(self.config_obj._config_dict, config_desired_2)

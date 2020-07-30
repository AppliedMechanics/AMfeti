#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

from amfeti.config_base import ConfigBase


class SolverManagerBase(ConfigBase):
    def __init__(self, local_problems_dict):
        self._local_problems_dict = local_problems_dict
        super().__init__()

    @property
    def solution(self):
        raise NotImplementedError('Solution property was not implemented for subclass')

    def solve(self):
        pass

    def update_local_problems(self, *args):
        raise NotImplementedError('Update method for local problems was not implemented for subclass')

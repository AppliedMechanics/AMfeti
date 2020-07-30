#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Template-module for the required API of an integrator-class, required by
:mod:`~amfeti.local_problems.dynamic_local_problems`
"""

__all__ = ['IntegratorBase']


class IntegratorBase:
    """
    Base-class for all one-step integration schemes.

    Attributes
    ----------
    dt : float
        time step size
    _t_n : float
        previous time
    _q_n : ndarray
        primary solution of previous time step
    _dq_n : ndarray
        first time-derivative of primary solution of previous time step
    _ddq_n : ndarray
        second time-derivative of primary solution of previous time step
    t_p : float
        next (predicted) time
    q_p : ndarray
        primary solution of next time step (predicted)
    dq_p : ndarray
        first time-derivative of primary solution of next time step (predicted)
    ddq_p : ndarray
        second time-derivative of primary solution of next time step (predicted)
    """

    def __init__(self):
        self.dt = None
        self._t_n = None
        self._q_n = None
        self._dq_n = None
        self._ddq_n = None

        self.t_p = None
        self.q_p = None
        self.dq_p = None
        self.ddq_p = None

    def residual_int(self, dq_p):
        raise NotImplementedError('Internal residual function was not implemented for subclass')

    def residual_ext(self, dq_p):
        raise NotImplementedError('External residual function was not implemented for subclass')

    def jacobian(self, dq_p):
        raise NotImplementedError('Jacobian function was not implemented for subclass')

    def set_prediction(self, q_n, dq_n, ddq_n, t_n):
        raise NotImplementedError('Prediction function was not implemented for subclass')

    def set_correction(self, dq_p):
        raise NotImplementedError('Correction function was not implemented for subclass')
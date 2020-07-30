#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Module for wrappers, that have to be part of the amfeti-folder.

NOTE FOR DEVELOPERS:
Even though these modules are mainly used by tests, they have to be part of the amfeti-folder due to serialization for
parallel tests. Apparently classes cannot be found outside of the amfeti-folder. So even if serialization works,
deserialization might fail.
"""

import numpy as np
from copy import copy


class NonlinearStaticCallbackWrapper:
    def __init__(self, K_0, f_ext_0):
        self.K_0 = K_0
        self.f_ext_0 = f_ext_0
        self.stiffness_param = 8.0e-5

    def K(self, q):
        return self.K_0 + self.stiffness_param * self.K_0 @ q

    def f_int(self, q):
        return self.K_0 @ q + self.stiffness_param * 0.5 * self.K_0 @ np.square(q)

    def f_ext(self, q):
        return self.f_ext_0


class NonlinearDynamicCallbackWrapper:
    def __init__(self, M_0, D_0, K_0, f_ext_0):
        self.K_0 = K_0
        self.M_0 = M_0
        self.D_0 = D_0
        self.f_ext_0 = f_ext_0
        self.stiffness_param = 1e-4
        self.t0 = 0.0
        self.t_max = 0.002
        self.t_end = 0.004

    def K(self, q, dq, t):
        return self.K_0 + self.stiffness_param * self.K_0 @ q

    def f_int(self, q, dq, t):
        return self.K_0 @ q + self.stiffness_param * 0.5 * self.K_0 @ np.square(q)

    def f_ext(self, q, dq, t):
        if self.t0 < t <= self.t_max:
            return self.f_ext_0 * (t - self.t0) / (self.t_max - self.t0)
        elif self.t_max < t < self.t_end:
            return self.f_ext_0 * (1.0 - (t - self.t_max) / (self.t_end - self.t_max))
        else:
            return 0.0

    def M(self, q, dq, t):
        return self.M_0

    def D(self, q, dq, t):
        return self.D_0


class LinearDynamicCallbackWrapper:
    def __init__(self, M_0, D_0, K_0, f_ext_0):
        self.K_0 = K_0
        self.M_0 = M_0
        self.D_0 = D_0
        self.f_ext_0 = f_ext_0
        self.t0 = 0.0
        self.t_max = 0.002
        self.t_end = 0.004

    def K(self, q, dq, t):
        return self.K_0

    def f_int(self, q, dq, t):
        return self.K_0 @ q

    def f_ext(self, q, dq, t):
        if self.t0 < t <= self.t_max:
            return self.f_ext_0 * (t - self.t0) / (self.t_max - self.t0)
        elif self.t_max < t < self.t_end:
            return self.f_ext_0 * (1.0 - (t - self.t_max) / (self.t_end - self.t_max))
        else:
            return 0.0

    def M(self, q, dq, t):
        return self.M_0

    def D(self, q, dq, t):
        return self.D_0


class NewmarkBetaIntegrator:
    def __init__(self, M, f_int, f_ext, K, D, beta=0.25, gamma=0.5):
        self.dt = None
        self._t_n = None
        self._q_n = None
        self._dq_n = None
        self._ddq_n = None

        self.t_p = None
        self.q_p = None
        self.dq_p = None
        self.ddq_p = None

        self.M = M
        self.f_int = f_int
        self.f_ext = f_ext
        self.K = K
        self.D = D

        # Set timeintegration parameters
        self.beta = beta
        self.gamma = gamma

    def residual_int(self, dq_p):
        M = self.M(self.q_p, dq_p, self.t_p)
        f_int_f = self.f_int(self.q_p, dq_p, self.t_p)
        D = self.D(self.q_p, dq_p, self.t_p)

        res = M @ self.ddq_p + f_int_f + D @ dq_p
        return res

    def residual_ext(self, dq_p):
        f_ext_f = self.f_ext(self.q_p, dq_p, self.t_p)
        res = - f_ext_f
        return res

    def jacobian(self, dq_p):
        M = self.M(self.q_p, dq_p, self.t_p)
        D = self.D(self.q_p, dq_p, self.t_p)
        K = self.K(self.q_p, dq_p, self.t_p)
        Jac = 1 / (self.gamma * self.dt) * M + D + self.dt * (self.beta / self.gamma) * K
        return Jac

    def set_prediction(self, q_n, dq_n, ddq_n, t_n):
        self._t_n = t_n
        self._q_n = q_n
        self._dq_n = dq_n
        self._ddq_n = ddq_n

        self.q_p = self._q_n + self.dt * self._dq_n + self.dt ** 2 * (
                1 / 2 - self.beta / self.gamma) * self._ddq_n
        self.dq_p = copy(self._dq_n)
        self.ddq_p = - (1 - self.gamma) / self.gamma * self._ddq_n
        self.t_p = t_n + self.dt
        return

    def set_correction(self, dq_p):
        delta_dq_p = dq_p - self.dq_p

        self.q_p += self.dt * self.beta / self.gamma * delta_dq_p
        self.dq_p = copy(dq_p)
        self.ddq_p += 1 / (self.gamma * self.dt) * delta_dq_p
        return
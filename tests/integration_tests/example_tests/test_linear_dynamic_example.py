"""Integration test for nonlinear dynamic example"""

from amfeti.parallelization_managers.tools import load_object
from amfeti import LinearDynamicFetiSolver
from amfeti.solver_managers import ParallelSolverManager
from amfeti.parallelization_managers import MPIManager
from amfeti.solvers import PCPGsolver
from amfeti.test_tools import LinearDynamicCallbackWrapper, NewmarkBetaIntegrator
from ...tools import CustomDictAssertTest
from .test_example_test_base import ExampleTestBase
from copy import deepcopy, copy
import os
import logging
import numpy as np


class NewmarkBeta:
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


class LinearDynamicExampleTest(ExampleTestBase):
    def setUp(self):
        super().setUp()
        example_dir = os.path.join(os.path.dirname(__file__), 'linear_dynamic_6subs_example')
        self.K_dict_0 = load_object(os.path.join(example_dir, 'K_linear_dynamic_dict_6subs.pkl'))
        self.M_dict_0 = load_object(os.path.join(example_dir, 'M_dict_6subs.pkl'))
        self.D_dict_0 = load_object(os.path.join(example_dir, 'D_dict_6subs.pkl'))
        self.f_ext_dict_0 = load_object(os.path.join(example_dir, 'f_ext_linear_dynamic_dict_6subs.pkl'))
        self.B_dict = load_object(os.path.join(example_dir, 'B_dict_6subs.pkl'))
        self.q_0_dict = load_object(os.path.join(example_dir, 'q_0_static_dict_6subs.pkl'))
        self.q_dict_desired = load_object(os.path.join(example_dir, 'q_solution_dict_6subs.pkl'))
        self.dq_dict_desired = load_object(os.path.join(example_dir, 'dq_solution_dict_6subs.pkl'))
        self.ddq_dict_desired = load_object(os.path.join(example_dir, 'ddq_solution_dict_6subs.pkl'))
        self.custom_asserter = CustomDictAssertTest()

        self.K_dict = dict()
        self.f_int_dict = dict()
        self.f_ext_dict = dict()

        wrapper = dict()
        self.integrator_dict = dict()
        for problem_id, K_0 in self.K_dict_0.items():
            wrapper[problem_id] = LinearDynamicCallbackWrapper(self.M_dict_0[problem_id], self.D_dict_0[problem_id], K_0,
                                                  self.f_ext_dict_0[problem_id])
            self.integrator_dict[problem_id] = NewmarkBetaIntegrator(wrapper[problem_id].M, wrapper[problem_id].f_int,
                                                      wrapper[problem_id].f_ext, wrapper[problem_id].K,
                                                      wrapper[problem_id].D)
            self.integrator_dict[problem_id].dt = 0.0001

    def tearDown(self):
        pass

    def test_serial_solver(self):
        solver = PCPGsolver()
        solver.set_config({'full_reorthogonalization': True})
        fetisolver = LinearDynamicFetiSolver(deepcopy(self.integrator_dict), deepcopy(self.B_dict), 0.0, 0.0003,
                                                deepcopy(self.q_0_dict), deepcopy(self.q_0_dict),
                                                deepcopy(self.q_0_dict), use_parallel=False,
                                                global_solver=solver)
        fetisolver.update()

        solution_obj = fetisolver.solve()

        q_dict_serial = dict()
        dq_dict_serial = dict()
        ddq_dict_serial = dict()
        for problem_id, problem in solution_obj.local_problems.items():
            q_dict_serial[problem_id] = problem.q
            dq_dict_serial[problem_id] = problem.dq
            ddq_dict_serial[problem_id] = problem.ddq

        self.custom_asserter.assert_dict_almost_equal(q_dict_serial, self.q_dict_desired)
        self.custom_asserter.assert_dict_almost_equal(dq_dict_serial, self.dq_dict_desired)
        self.custom_asserter.decimals = 2
        self.custom_asserter.assert_dict_almost_equal(ddq_dict_serial, self.ddq_dict_desired)

        info_dict_keys_desired = dict()
        for tstep in range(0, 3):
            info_dict_keys_desired[tstep] = {'avg_iteration_time': None,
                                               'Total_elaspsed_time': None,
                                               'iterations': None,
                                               'lambda_hist': None,
                                               'residual_hist': None,
                                               'residual': None}

        self.custom_asserter.assert_dict_keys_equal(solution_obj.solver_information, info_dict_keys_desired)

    def test_parallel_solver(self):
        if self.run_parallel_tests:
            solver = PCPGsolver()
            solver.set_config({'full_reorthogonalization': True})
            fetisolver = LinearDynamicFetiSolver(deepcopy(self.integrator_dict), deepcopy(self.B_dict), 0.0, 0.0003,
                                                 deepcopy(self.q_0_dict), deepcopy(self.q_0_dict),
                                                 deepcopy(self.q_0_dict), use_parallel=True,
                                                 global_solver=solver)
            # If 6 processors are available, the following 6 lines can be omitted
            mpi_manager = MPIManager()
            mpi_manager.set_config({'mpi_rank2problems': {0: np.array([1, 3]),
                                                          1: np.array([2, 4]),
                                                          2: np.array([5]),
                                                          3: np.array([6])}})
            solver_manager = ParallelSolverManager(fetisolver._local_problems, fetisolver._config_dict['global_solver'])
            solver_manager.set_config({'parallelization_manager': mpi_manager})

            fetisolver.set_config({'solver_manager': solver_manager})
            fetisolver.update()

            solution_obj = fetisolver.solve()

            q_dict_parallel = dict()
            dq_dict_parallel = dict()
            ddq_dict_parallel = dict()
            for problem_id, problem in solution_obj.local_problems.items():
                q_dict_parallel[problem_id] = problem.q
                dq_dict_parallel[problem_id] = problem.dq
                ddq_dict_parallel[problem_id] = problem.ddq

            self.custom_asserter.assert_dict_almost_equal(q_dict_parallel, self.q_dict_desired)
            self.custom_asserter.assert_dict_almost_equal(dq_dict_parallel, self.dq_dict_desired)
            self.custom_asserter.decimals = 3
            self.custom_asserter.assert_dict_almost_equal(ddq_dict_parallel, self.ddq_dict_desired)

            info_dict_keys_desired = dict()
            for tstep in range(0, 3):
                info_dict_keys_desired[tstep] = {'avg_iteration_time': None,
                                                 'Total_elaspsed_time': None,
                                                 'iterations': None,
                                                 'lambda_hist': None,
                                                 'residual_hist': None,
                                                 'residual': None}

            self.custom_asserter.assert_dict_keys_equal(solution_obj.solver_information, info_dict_keys_desired)
        else:
            logger = logging.getLogger(__name__)
            logger.warning('Parallel test has not been run. If parallel tests shall be run, switch on the '
                           'run_parallel_tests-flag manually.')

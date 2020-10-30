"""Integration test for nonlinear dynamic example"""

from amfeti.parallelization_managers.tools import load_object
from amfeti import NonlinearDynamicFetiSolver
from amfeti.solver_managers import ParallelSolverManager
from amfeti.parallelization_managers import MPIManager
from amfeti.solvers import PCPGsolver
from amfeti.test_tools import NonlinearDynamicCallbackWrapper, NewmarkBetaIntegrator
from ...tools import CustomDictAssertTest
from .test_example_test_base import ExampleTestBase
from copy import deepcopy, copy
import os
import logging
import numpy as np


class NonlinearDynamicExampleTest(ExampleTestBase):
    def setUp(self):
        super().setUp()
        example_dir = os.path.join(os.path.dirname(__file__), 'nonlinear_dynamic_6subs_example')
        self.K_dict_0 = load_object(os.path.join(example_dir, 'K_nonlinear_dynamic_dict_6subs.pkl'))
        self.M_dict_0 = load_object(os.path.join(example_dir, 'M_dict_6subs.pkl'))
        self.D_dict_0 = load_object(os.path.join(example_dir, 'D_dict_6subs.pkl'))
        self.f_ext_dict_0 = load_object(os.path.join(example_dir, 'f_ext_nonlinear_dynamic_dict_6subs.pkl'))
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
            wrapper[problem_id] = NonlinearDynamicCallbackWrapper(self.M_dict_0[problem_id], self.D_dict_0[problem_id], K_0,
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
        fetisolver = NonlinearDynamicFetiSolver(deepcopy(self.integrator_dict), deepcopy(self.B_dict), 0.0, 0.0003,
                                                deepcopy(self.q_0_dict), deepcopy(self.q_0_dict),
                                                deepcopy(self.q_0_dict), use_parallel=False,
                                       loadpath_controller_options = {'nonlinear_solver_options': {'atol': 1.0e-6,
                                                                                                   'rtol': 1.0e-9},
                                                                      'N_steps': 1}, global_solver=solver)
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

    def test_parallel_solver(self):
        if self.run_parallel_tests:
            solver = PCPGsolver()
            solver.set_config({'full_reorthogonalization': True})
            fetisolver = NonlinearDynamicFetiSolver(deepcopy(self.integrator_dict), deepcopy(self.B_dict), 0.0, 0.0003,
                                                    deepcopy(self.q_0_dict), deepcopy(self.q_0_dict),
                                                    deepcopy(self.q_0_dict), use_parallel=True,
                                                    loadpath_controller_options={
                                                        'nonlinear_solver_options': {'atol': 1.0e-6,
                                                                                     'rtol': 1.0e-9},
                                                        'N_steps': 1}, global_solver=solver)
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
            self.custom_asserter.decimals = 2
            self.custom_asserter.assert_dict_almost_equal(ddq_dict_parallel, self.ddq_dict_desired)
        else:
            logger = logging.getLogger(__name__)
            logger.warning('Parallel test has not been run. If parallel tests shall be run, switch on the '
                           'run_parallel_tests-flag manually.')

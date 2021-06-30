"""Integration test for nonlinear static example"""

from amfeti.parallelization_managers.tools import load_object
from amfeti import NonlinearStaticFetiSolver
from amfeti.solver_managers import ParallelSolverManager
from amfeti.parallelization_managers import MPIManager
from amfeti.test_tools.model_wrappers import NonlinearStaticCallbackWrapper
from ...tools import CustomDictAssertTest
from .test_example_test_base import ExampleTestBase
from copy import deepcopy
import os
import logging
import numpy as np


class NonlinearStaticExampleTest(ExampleTestBase):
    def setUp(self):
        super().setUp()
        example_dir = os.path.join(os.path.dirname(__file__), 'nonlinear_static_6subs_example')
        self.K_dict_0 = load_object(os.path.join(example_dir, 'K_nonlinear_static_dict_6subs.pkl'))
        self.f_ext_dict_0 = load_object(os.path.join(example_dir, 'f_ext_nonlinear_static_dict_6subs.pkl'))
        self.B_dict = load_object(os.path.join(example_dir, 'B_dict_6subs.pkl'))
        self.q_0_dict = load_object(os.path.join(example_dir, 'q_0_static_dict_6subs.pkl'))
        self.q_dict_desired = load_object(os.path.join(example_dir, 'q_solution_dict_6subs.pkl'))
        self.custom_asserter = CustomDictAssertTest()

        self.K_dict = dict()
        self.f_int_dict = dict()
        self.f_ext_dict = dict()

        self.wrapper = dict()
        for problem_id, K_0 in self.K_dict_0.items():
            self.wrapper[problem_id] = NonlinearStaticCallbackWrapper(K_0, self.f_ext_dict_0[problem_id])
            self.K_dict[problem_id] = self.wrapper[problem_id].K
            self.f_int_dict[problem_id] = self.wrapper[problem_id].f_int
            self.f_ext_dict[problem_id] = self.wrapper[problem_id].f_ext

    def tearDown(self):
        pass

    def test_serial_solver(self):
        fetisolver = NonlinearStaticFetiSolver(deepcopy(self.K_dict), deepcopy(self.B_dict),
                                               deepcopy(self.f_int_dict), deepcopy(self.f_ext_dict),
                                               deepcopy(self.q_0_dict), use_parallel=False)
        fetisolver.update()

        solution_obj = fetisolver.solve()

        q_dict_serial = dict()
        for problem_id, problem in solution_obj.local_problems.items():
            q_dict_serial[problem_id] = problem.q

        self.custom_asserter.assert_dict_almost_equal(q_dict_serial, self.q_dict_desired)

        info_dict_keys_desired = dict()
        for tstep in range(0, 10):
            linear_solver_dict = dict()
            if tstep is 0:
                linear_solver_dict[0] = {'avg_iteration_time': None,
                                                   'Total_elaspsed_time': None,
                                                   'iterations': None,
                                                   'lambda_hist': None,
                                                   'residual_hist': None,
                                                   'residual': None}
            elif tstep > 0:
                for newton_iter in range(0, 10):
                    linear_solver_dict[newton_iter] = {'avg_iteration_time': None,
                                                         'Total_elaspsed_time': None,
                                                         'iterations': None,
                                                         'lambda_hist': None,
                                                         'residual_hist': None,
                                                         'residual': None}
            info_dict_keys_desired[tstep] = {'newton': {'residual': None,
                                                             'linear_solver': linear_solver_dict,
                                                             'iterations': None
                                                             }}

        self.custom_asserter.assert_dict_keys_equal(solution_obj.solver_information, info_dict_keys_desired)

    def test_parallel_solver(self):
        if self.run_parallel_tests:
            fetisolver = NonlinearStaticFetiSolver(deepcopy(self.K_dict), deepcopy(self.B_dict),
                                                   deepcopy(self.f_int_dict), deepcopy(self.f_ext_dict),
                                                   deepcopy(self.q_0_dict), use_parallel=True)
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
            for problem_id, problem in solution_obj.local_problems.items():
                q_dict_parallel[problem_id] = problem.q

            self.custom_asserter.assert_dict_almost_equal(q_dict_parallel, self.q_dict_desired)

            info_dict_keys_desired = dict()
            for tstep in range(0, 10):
                linear_solver_dict = dict()
                if tstep is 0:
                    linear_solver_dict[0] = {'avg_iteration_time': None,
                                             'Total_elaspsed_time': None,
                                             'iterations': None,
                                             'lambda_hist': None,
                                             'residual_hist': None,
                                             'residual': None}
                elif tstep > 0:
                    for newton_iter in range(0, 10):
                        linear_solver_dict[newton_iter] = {'avg_iteration_time': None,
                                                           'Total_elaspsed_time': None,
                                                           'iterations': None,
                                                           'lambda_hist': None,
                                                           'residual_hist': None,
                                                           'residual': None}
                info_dict_keys_desired[tstep] = {'newton': {'residual': None,
                                                            'linear_solver': linear_solver_dict,
                                                            'iterations': None
                                                            }}

            self.custom_asserter.assert_dict_keys_equal(solution_obj.solver_information, info_dict_keys_desired)
        else:
            logger = logging.getLogger(__name__)
            logger.warning('Parallel test has not been run. If parallel tests shall be run, switch on the '
                           'run_parallel_tests-flag manually.')

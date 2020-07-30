"""Integration test for linear static example"""

from amfeti.parallelization_managers.tools import load_object
from amfeti import LinearStaticFetiSolver
from amfeti.solver_managers import ParallelSolverManager
from amfeti.parallelization_managers import MPIManager
from ...tools import CustomDictAssertTest
from .test_example_test_base import ExampleTestBase
from copy import deepcopy
import os
import logging
import numpy as np


class LinearStaticExampleTest(ExampleTestBase):
    def setUp(self):
        super().setUp()
        example_dir = os.path.join(os.path.dirname(__file__), 'linear_static_6subs_example')
        self.K_dict = load_object(os.path.join(example_dir, 'K_linear_static_dict_6subs.pkl'))
        self.f_dict = load_object(os.path.join(example_dir, 'f_linear_static_dict_6subs.pkl'))
        self.B_dict = load_object(os.path.join(example_dir, 'B_dict_6subs.pkl'))
        self.q_dict_desired = load_object(os.path.join(example_dir, 'q_solution_dict_6subs.pkl'))
        self.custom_asserter = CustomDictAssertTest()

    def tearDown(self):
        pass

    def test_serial_solver(self):
        fetisolver = LinearStaticFetiSolver(deepcopy(self.K_dict), deepcopy(self.B_dict), deepcopy(self.f_dict),
                                            use_parallel=False)
        fetisolver.update()

        solution_obj = fetisolver.solve()

        q_dict_serial = dict()
        for problem_id, problem in solution_obj.local_problems.items():
            q_dict_serial[problem_id] = problem.q

        self.custom_asserter.assert_dict_almost_equal(q_dict_serial, self.q_dict_desired)

    def test_parallel_solver(self):
        if self.run_parallel_tests:
            fetisolver = LinearStaticFetiSolver(deepcopy(self.K_dict), deepcopy(self.B_dict), deepcopy(self.f_dict),
                                                use_parallel=True)
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
        else:
            logger = logging.getLogger(__name__)
            logger.warning('Parallel test has not been run. If parallel tests shall be run, switch on the '
                           'run_parallel_tests-flag manually.')

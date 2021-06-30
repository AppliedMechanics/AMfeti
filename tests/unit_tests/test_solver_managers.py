"""Test routines for solver-manager module"""

from unittest import TestCase
from amfeti.solver_managers import SerialSolverManager, ParallelSolverManager
from amfeti.coarse_grid import NaturalCoarseGrid
from amfeti.solution import StandardSolution
from ..tools import CustomDictAssertTest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, inv, splu
import os
import logging

class DummySolver:
    def __init__(self):
        pass
        self.project = None
        self.precondition = None

    def set_config(self, conf_dict):
        self.project = conf_dict['projection']
        self.precondition = conf_dict['precondition']

    def solve(self, F_action, residual_action, lambda_init):

        lamda = np.array([-0.5, 1, 0.2])
        gap = np.array([0.01, 0.3, -0.04])
        u = F_action(lamda)
        res = residual_action(lamda)
        f = self.precondition(u)
        f_proj = self.project(f)

        return f_proj, {'residual': res, 'gap': gap}


class DummySolution:
    def __init__(self):
        self.local_problems = None
        self.lamda = None
        self.alpha = None
        self.info_dict = None

    def update(self, local_problems_dict, lamda, alpha, solver_info_dict):
        self.local_problems = local_problems_dict
        self.lamda = lamda
        self.alpha = alpha
        self.info_dict = solver_info_dict


class DummyCoarseGrid:
    def __init__(self):
        self.init_solution = np.array([-0.5, 0.0, 0.7, 0.0, 0.4, -0.25])

    def solve(self, v=None):
        if v is None:
            return self.init_solution


class SerialSolverManagerTest(TestCase):
    def setUp(self):
        class DummyLocalProblem:
            def __init__(self, id, B, f, R, K):
                self.B = B
                self.f = f
                self.R = R
                self.K = K
                self.update_system_performed = False
                self.id = id

            @property
            def kernel(self):
                return self.R

            def solve(self, v_dict, external_bool, alpha=None):
                u_dict = dict()
                u, s, vt = np.linalg.svd(self.K.todense(), full_matrices=False)
                K_inv = vt.T @ np.linalg.pinv(np.diag(s)) @ u.T
                K_inv = csr_matrix(K_inv)
                for interface, B in self.B.items():
                    f = B.T @ v_dict[interface]
                    if external_bool:
                        f += self.f
                    if alpha is not None:
                        u = K_inv @ f + self.R @ alpha
                    else:
                        u = K_inv @ f
                    u_dict[interface] = B @ u
                return u_dict

            def precondition(self, v_dict):
                f_dict = dict()
                for interface, B in self.B.items():
                    u = B.T @ v_dict[interface]
                    f_dict[interface] = B @ (self.K @ u)
                return f_dict

            def update_system(self, external_solution_dict, update_input_dict=None):
                self.update_system_performed = True

            def dump_local_information(self):
                info_dict = dict()
                info_dict.update({'residual': self.id * np.array([0.1, 0.2, 0.3, 0.4])})
                return info_dict

        B1 = {'interface1': csr_matrix(np.array([[0, 0, 1, 0, 0],
                                                 [1, 0, 0, 0, 0]])),
              'interface2': csr_matrix(np.array([[-1, 0, 0, 0, 0]]))}
        f1 = np.array([0, 0, 0, 0, 0])
        R1 = csr_matrix(np.array([[0.1], [0], [0.3], [-0.25], [0]]))
        K1 = csr_matrix(np.array([[2, -2, 0, 0, 0], [-2, 4, -2, 0, 0], [0, -2, 4, -2, 0], [0, 0, -2, 4, -2], [0, 0, 0, -2, 2]], dtype=float))
        B2 = {'interface1': csr_matrix(np.array([[-1, 0, 0, 0],
                                                 [0, -1, 0, 0]])),
              'interface2': csr_matrix(np.array([[0, 0, 0, 1]]))}
        f2 = np.array([0, 3, 2.3, 0])
        R2 = csr_matrix(np.array([[-0.25], [0.1], [0], [-0.15]]))
        K2 = csr_matrix(np.array([[2, -2, 0, 0], [-2, 4, -2, 0], [0, -2, 4, -2], [0, 0, -2, 4]], dtype=float))
        local_problems = {'sub1': DummyLocalProblem(1, B1, f1, R1, K1),
                          'sub2': DummyLocalProblem(2, B2, f2, R2, K2)}
        solver = DummySolver()
        self.solution = DummySolution()
        self.solver_manager = SerialSolverManager(local_problems, solver)
        self.customasserter = CustomDictAssertTest()

    def tearDown(self):
        pass

    def test_solution(self):
        self.solver_manager._solution = self.solution
        lambda_desired = np.random.rand(12)
        alpha_desired = np.random.rand(4)
        info_dict_desired = {'gap': np.random.rand(12),
                             'xyz': None}
        self.solver_manager._lambda_sol = lambda_desired
        self.solver_manager._alpha_sol = alpha_desired
        self.solver_manager._info_dict = info_dict_desired

        solution_actual = self.solver_manager.solution

        self.assertTrue(isinstance(solution_actual, DummySolution))
        assert_array_almost_equal(solution_actual.lamda, lambda_desired)
        assert_array_almost_equal(solution_actual.alpha, alpha_desired)

    def test_no_lagrange_multiplier(self):
        self.solver_manager._global_dof_dimension = 12
        self.assertEqual(self.solver_manager.no_lagrange_multiplier, 12)

    def test_set_config(self):
        config_dict = {'tolerance': 1e-7}

        self.solver_manager.set_config(config_dict)

        config_dict_desired = {'coarse_grid': self.solver_manager._config_dict['coarse_grid'],
                               'solution': self.solver_manager._config_dict['solution'],
                               'tolerance': 1e-7}

        self.assertTrue(isinstance(self.solver_manager._config_dict['coarse_grid'], NaturalCoarseGrid))
        self.assertTrue(isinstance(self.solver_manager._config_dict['solution'], StandardSolution))
        self.customasserter.assert_dict_almost_equal(self.solver_manager._config_dict, config_dict_desired)

        config_dict = {'coarse_grid': None,
                       'solution': self.solution}

        self.solver_manager.set_config(config_dict)

        config_dict_desired = {'coarse_grid': None,
                               'solution': self.solution,
                               'tolerance': 1e-7}

        self.customasserter.assert_dict_almost_equal(self.solver_manager._config_dict, config_dict_desired)

    def test_update(self):
        config_dict = {'solution': self.solution}
        self.solver_manager.set_config(config_dict)

        self.solver_manager.update()

        self.assertTrue(isinstance(self.solver_manager._coarse_grid, NaturalCoarseGrid))
        self.assertTrue(isinstance(self.solver_manager._solution, DummySolution))

        interface2dofmap_desired = {'interface1': np.array([0, 1]),
                                    'interface2': np.array([2])}

        self.customasserter.assert_dict_almost_equal(self.solver_manager._interface2dof_map, interface2dofmap_desired)
        self.assertEqual(self.solver_manager._global_dof_dimension, 3)
        self.assertEqual(self.solver_manager._coarse_grid._dimension_interface, 3)

    def test_update_local_problems(self):
        lambda_sol = np.random.rand(12)
        update_input_dict = {'load_factor': 0.3}
        local_info_dict = dict()
        local_info_dict_actual = self.solver_manager.update_local_problems(lambda_sol, update_input_dict, local_info_dict)
        for problem_id, local_problem in self.solver_manager._local_problems_dict.items():
            self.assertTrue(local_problem.update_system_performed)
        local_info_dict_desired = {'sub1': {'residual': np.array([0.1, 0.2, 0.3, 0.4])},
                                   'sub2': {'residual': np.array([0.2, 0.4, 0.6, 0.8])}}
        self.customasserter.assert_dict_almost_equal(local_info_dict_actual, local_info_dict_desired)

    def test_interfacedict2vector(self):
        config_dict = {'solution': self.solution}
        self.solver_manager.set_config(config_dict)
        self.solver_manager.update()

        interfacedict = {'interface1': np.array([1.5, 3.2]),
                         'interface2': np.array([2.5])}

        vector_actual = self.solver_manager._interfacedict2vector(interfacedict)
        vector_desired = np.array([1.5, 3.2, 2.5])

        assert_array_almost_equal(vector_actual, vector_desired)

        vector_actual = self.solver_manager._interfacedict2vector(self.solver_manager._local_problems_dict['sub1'].B)
        vector_desired = np.array([[0, 0, 1, 0, 0],
                                   [1, 0, 0, 0, 0],
                                   [-1, 0, 0, 0, 0]])

        assert_array_almost_equal(vector_actual.todense(), vector_desired)

    def test_vector2interfacedict(self):
        config_dict = {'solution': self.solution}
        self.solver_manager.set_config(config_dict)
        self.solver_manager.update()

        vector = np.array([1.5, 3.2, 2.5])

        interfacedict_actual = self.solver_manager._vector2interfacedict(vector)
        interfacedict_desired = {'interface1': np.array([1.5, 3.2]),
                                 'interface2': np.array([2.5])}

        self.customasserter.assert_dict_almost_equal(interfacedict_actual, interfacedict_desired)

        vector = csr_matrix(np.array([[0, 0, 1, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [-1, 0, 0, 0, 0]]))

        interfacedict_actual = self.solver_manager._vector2interfacedict(vector)
        self.customasserter.assert_dict_almost_equal(interfacedict_actual, self.solver_manager._local_problems_dict['sub1'].B)

    def test_F_action(self):
        config_dict = {'solution': self.solution}
        self.solver_manager.set_config(config_dict)
        self.solver_manager.update()

        lamda = np.array([0.5, 1, 1.5])

        gap_actual = self.solver_manager._F_action(lamda)
        gap_desired = np.array([-2.5, -2.8, -1.65])

        assert_array_almost_equal(gap_actual, gap_desired)

    def test_residual(self):
        config_dict = {'solution': self.solution}
        self.solver_manager.set_config(config_dict)
        self.solver_manager.update()

        lamda = np.array([0.5, 1, 1.5])

        residual_actual = self.solver_manager._residual(lamda)
        residual_desired = np.array([-4.3, -4, 4.3])

        assert_array_almost_equal(residual_actual, residual_desired)

    def test_apply_preconditioner(self):
        config_dict = {'solution': self.solution}
        self.solver_manager.set_config(config_dict)
        self.solver_manager.update()

        gap = np.array([0.5, 1, 1.5])

        lamda_actual = self.solver_manager._apply_preconditioner(gap)
        lamda_desired = np.array([1, 5, 9])

        assert_array_almost_equal(lamda_actual, lamda_desired)

    def test_solve(self):
        config_dict = {'solution': self.solution}
        self.solver_manager.set_config(config_dict)
        self.solver_manager.update()

        self.solver_manager.solve()

        lamda_desired = np.array([0.669136, -0.535309, 1.472099])
        alpha_desired = np.array([40.311809, -18.567904])
        gap_desired = np.array([0.01,  0.3, -0.04])
        residual_desired = np.array([-6.5, -5.4, 2.87])

        assert_array_almost_equal(self.solver_manager._lambda_sol, lamda_desired)
        assert_array_almost_equal(self.solver_manager._alpha_sol, alpha_desired)
        assert_array_almost_equal(self.solver_manager._info_dict['gap'], gap_desired)
        assert_array_almost_equal(self.solver_manager._info_dict['residual'], residual_desired)

    def test_initialize_lambda(self):
        coarse_grid = DummyCoarseGrid()
        self.solver_manager._coarse_grid = coarse_grid
        lamda_sol_desired = np.array([-0.5, 0.0, 0.7, 0.0, 0.4, -0.25])
        lamda_sol_actual = self.solver_manager.initialize_lambda()
        assert_array_almost_equal(lamda_sol_actual, lamda_sol_desired)


class ParallelSolverManagerTest(TestCase):

    def setUp(self):
        class DummyLocalProblem:
            def __init__(self, id, B, f, R, K):
                self.B = B
                self.f = f
                self.R = R
                self.K = K
                self.update_system_performed = False
                self.id = id

            @property
            def kernel(self):
                return self.R

            def solve(self, v_dict, external_bool, alpha=None):
                u_dict = dict()
                u, s, vt = np.linalg.svd(self.K.todense(), full_matrices=False)
                K_inv = vt.T @ np.linalg.pinv(np.diag(s)) @ u.T
                K_inv = csr_matrix(K_inv)
                for interface, B in self.B.items():
                    f = B.T @ v_dict[interface]
                    if external_bool:
                        f += self.f
                    if alpha is not None:
                        u = K_inv @ f + self.R @ alpha
                    else:
                        u = K_inv @ f
                    u_dict[interface] = B @ u
                return u_dict

            def precondition(self, v_dict):
                f_dict = dict()
                for interface, B in self.B.items():
                    u = B.T @ v_dict[interface]
                    f_dict[interface] = B @ (self.K @ u)
                return f_dict

            def update_system(self, external_solution_dict, update_input_dict=None):
                self.update_system_performed = True

            def dump_local_information(self):
                info_dict = dict()
                info_dict.update({'residual': self.id * np.array([0.1, 0.2, 0.3, 0.4])})
                return info_dict

        class DummySolutionParallel:
            def __init__(self):
                B1 = {'interface1': csr_matrix(np.array([[0, 0, 1, 0, 0],
                                                         [1, 0, 0, 0, 0]])),
                      'interface2': csr_matrix(np.array([[-1, 0, 0, 0, 0]]))}
                f1 = np.array([0, 0, 0, 0, 0])
                R1 = csr_matrix(np.array([[0.1], [0], [0.3], [-0.25], [0]]))
                K1 = csr_matrix(
                    np.array(
                        [[2, -2, 0, 0, 0], [-2, 4, -2, 0, 0], [0, -2, 4, -2, 0], [0, 0, -2, 4, -2], [0, 0, 0, -2, 2]],
                        dtype=float))
                B2 = {'interface1': csr_matrix(np.array([[-1, 0, 0, 0],
                                                         [0, -1, 0, 0]])),
                      'interface2': csr_matrix(np.array([[0, 0, 0, 1]]))}
                f2 = np.array([0, 3, 2.3, 0])
                R2 = csr_matrix(np.array([[-0.25], [0.1], [0], [-0.15]]))
                K2 = csr_matrix(np.array([[2, -2, 0, 0], [-2, 4, -2, 0], [0, -2, 4, -2], [0, 0, -2, 4]], dtype=float))
                local_problems = {'sub1': DummyLocalProblem(1, B1, f1, R1, K1),
                                  'sub2': DummyLocalProblem(2, B2, f2, R2, K2)}
                self.local_problems = local_problems
                self.dual_solution = None
                self.kernel_modes = None
                self.solver_information = "solver info"

            def update(self, local_problems_dict, lamda, alpha, solver_info_dict):
                self.local_problems = local_problems_dict
                self.lamda = lamda
                self.alpha = alpha
                self.info_dict = solver_info_dict

        class DummyMPIManager:
            def __init__(self, **kwargs):
                self.is_called = False
                self._config_dict = dict()
                self.set_config({'tmp_folder': 'tmp',
                                 'tmp_folder_absolute': None,
                                 'remove_tmp': False})
                self._launcher_script_path = None
                self._local_folder = None
                mpi_exec = 'mpiexec'
                try:
                    mpi_path = os.environ['MPIDIR']
                    mpi_exec = os.path.join(mpi_path, mpi_exec).replace('"', '')
                except:
                    logger = logging.getLogger(__name__)
                    logger.warning("Warning! Using mpiexec in global path")

                try:
                    python_path = os.environ['PYTHON_ENV']
                    python_exec = os.path.join(python_path, 'python').replace('"', '')
                except:
                    logger = logging.getLogger(__name__)
                    logger.warning("Warning! Using python in global path")
                    python_exec = 'python'

                self.set_config({'write_log': True,
                                 'mpi_exec': mpi_exec,
                                 'mpi_args': '',
                                 'mpi_size': None,
                                 'mpi_rank2problems': {1: local_problems},
                                 'python_exec': python_exec,
                                 'additional_mpi_args': kwargs,
                                 'solution_path': None})

                self._python_file = 'mpi_local_processor.py'

                self._file_extension = '.pkl'
                self._path_prefix = 'mpi_rank_'
                self._rank_path = dict()

            def set_config(self, new_config_dict):
                if isinstance(new_config_dict, dict):
                    self._config_dict.update(new_config_dict)

            def read_solutions(self):
                return {1: DummySolutionParallel(), 2: DummySolutionParallel()}

            def load_local_problems(self):
                B1 = {'interface1': csr_matrix(np.array([[0, 0, 1, 0, 0],
                                                         [1, 0, 0, 0, 0]])),
                      'interface2': csr_matrix(np.array([[-1, 0, 0, 0, 0]]))}
                f1 = np.array([0, 0, 0, 0, 0])
                R1 = csr_matrix(np.array([[0.1], [0], [0.3], [-0.25], [0]]))
                K1 = csr_matrix(
                    np.array(
                        [[2, -2, 0, 0, 0], [-2, 4, -2, 0, 0], [0, -2, 4, -2, 0], [0, 0, -2, 4, -2], [0, 0, 0, -2, 2]],
                        dtype=float))
                B2 = {'interface1': csr_matrix(np.array([[-1, 0, 0, 0],
                                                         [0, -1, 0, 0]])),
                      'interface2': csr_matrix(np.array([[0, 0, 0, 1]]))}
                f2 = np.array([0, 3, 2.3, 0])
                R2 = csr_matrix(np.array([[-0.25], [0.1], [0], [-0.15]]))
                K2 = csr_matrix(np.array([[2, -2, 0, 0], [-2, 4, -2, 0], [0, -2, 4, -2], [0, 0, -2, 4]], dtype=float))
                local_problems = {'sub1': DummyLocalProblem(1, B1, f1, R1, K1),
                                  'sub2': DummyLocalProblem(2, B2, f2, R2, K2)}
                return local_problems

            def set_up_parallel_process(self, local_problems_dict, solver, local_solver_manager_config):
                self.is_called = True

            def launch_parallel_process(self):
                pass

        B1 = {'interface1': csr_matrix(np.array([[0, 0, 1, 0, 0],
                                                 [1, 0, 0, 0, 0]])),
              'interface2': csr_matrix(np.array([[-1, 0, 0, 0, 0]]))}
        f1 = np.array([0, 0, 0, 0, 0])
        R1 = csr_matrix(np.array([[0.1], [0], [0.3], [-0.25], [0]]))
        K1 = csr_matrix(
            np.array([[2, -2, 0, 0, 0], [-2, 4, -2, 0, 0], [0, -2, 4, -2, 0], [0, 0, -2, 4, -2], [0, 0, 0, -2, 2]],
                     dtype=float))
        B2 = {'interface1': csr_matrix(np.array([[-1, 0, 0, 0],
                                                 [0, -1, 0, 0]])),
              'interface2': csr_matrix(np.array([[0, 0, 0, 1]]))}
        f2 = np.array([0, 3, 2.3, 0])
        R2 = csr_matrix(np.array([[-0.25], [0.1], [0], [-0.15]]))
        K2 = csr_matrix(np.array([[2, -2, 0, 0], [-2, 4, -2, 0], [0, -2, 4, -2], [0, 0, -2, 4]], dtype=float))
        local_problems = {'sub1': DummyLocalProblem(1, B1, f1, R1, K1),
                          'sub2': DummyLocalProblem(2, B2, f2, R2, K2)}
        solver = DummySolver()
        self.solution = DummySolutionParallel()
        self.solver_manager = ParallelSolverManager(local_problems, solver)
        self.solver_manager._parallelization_manager = DummyMPIManager()
        self.customasserter = CustomDictAssertTest()
        self.local_problems = local_problems

    def tearDown(self):
        pass

    def test_solution(self):
        self.solver_manager._solution = self.solution
        solution_actual = self.solver_manager.solution
        self.assertTrue(isinstance(solution_actual, type(self.solution)))

        custom_test = CustomDictAssertTest()
        custom_test.assert_dict_keys_equal(solution_actual.local_problems, self.local_problems)
        for key in self.local_problems.keys():
            self.assertEqual(self.local_problems[key].id,
                             solution_actual.local_problems[key].id)  # comparing DummyLocalProblems by their ID

    def test_no_lagrange_multiplier(self):
        self.solver_manager.update()
        self.assertEqual(self.solver_manager.no_lagrange_multiplier, 3)

    def test_update(self):
        config_dict = {'solution': self.solution, 'coarse_grid': NaturalCoarseGrid(),
                       'parallelization_manager': self.solver_manager._parallelization_manager}
        self.solver_manager.set_config(config_dict)

        self.solver_manager.update()

        self.assertTrue(isinstance(self.solver_manager._coarse_grid, NaturalCoarseGrid))
        self.assertTrue(isinstance(self.solver_manager._solution, type(self.solution)))
        self.assertEqual(self.solver_manager._parallelization_manager.is_called, True)

        self.solver_manager._parallelization_manager = None
        self.solver_manager.update()

        with self.assertRaises(ValueError):
            self.solver_manager._local_problems_distributed = True
            self.solver_manager.update()

    def test_solve(self):
        self.solver_manager.solve()
        self.assertEqual(self.solver_manager._local_problems_dict.keys(), self.local_problems.keys())

    def test_update_local_problems(self):
        lambda_sol = np.random.rand(12)
        update_input_dict = {'load_factor': 0.3}
        local_info_dict = dict()
        local_info_dict_actual = self.solver_manager.update_local_problems(lambda_sol, update_input_dict,
                                                                           local_info_dict)
        for problem_id, local_problem in self.solver_manager._local_problems_dict.items():
            self.assertTrue(local_problem.update_system_performed)
        local_info_dict_desired = {'sub1': {'residual': np.array([0.1, 0.2, 0.3, 0.4])},
                                   'sub2': {'residual': np.array([0.2, 0.4, 0.6, 0.8])}}
        self.customasserter.assert_dict_almost_equal(local_info_dict_actual, local_info_dict_desired)
        self.assertEqual(self.solver_manager._parallelization_manager.is_called, True)

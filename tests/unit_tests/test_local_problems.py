"""Test routines for local_problems module"""

from unittest import TestCase
from amfeti.local_problems import LinearStaticLocalProblem, LinearDynamicLocalProblem, IntegratorBase
from amfeti.local_problems.local_problem_base import LocalProblemBase
from amfeti.scaling import MultiplicityScaling
from amfeti.preconditioners import DirichletPreconditioner
from amfeti.linalg.datatypes import Matrix
from ..tools import CustomDictAssertTest
from scipy.sparse import csc_matrix, csr_matrix
import numpy as np
from copy import copy
from numpy.testing import assert_array_equal, assert_array_almost_equal


class DummyPreconditioner:
    def __init__(self):
        self.Q=None

    def update(self, K, interface_dofs):
        self.Q = K.data[np.ix_(interface_dofs, interface_dofs)]


class LocalProblemBaseTest(TestCase):
    def setUp(self):
        class DummyPreconditioner:
            def __init__(self):
                pass

            def update(self, K, B):
                pass
        self.local_problem = LocalProblemBase(1)

    def tearDown(self):
        pass

    def test_create_local_problem(self):
        local_problem = LocalProblemBase(1)
        self.assertEqual(local_problem.id, 1)
        local_problem = LocalProblemBase('my_new_local_problem')
        self.assertEqual(local_problem.id, 'my_new_local_problem')

    def test_solve(self):
        f_test = {'interface': np.array([0, 1, 4, 2.1])}
        self.assertEqual(self.local_problem.solve(f_test), f_test)

    def test_precondition(self):
        q_test = {'interface': np.array([0.1, 0.2, 0.15, 0.05])}
        self.assertEqual(self.local_problem.precondition(q_test), q_test)


class LinearStaticLocalProblemTest(TestCase):
    def setUp(self):
        self.K = csc_matrix(np.array([[2, -2, 0, 0], [-2, 4, -2, 0], [0, -2, 4, -2], [0, 0, -2, 2]]))
        self.f = np.array([1, 2, 0, 0])
        self.B = {'interface1': csr_matrix(np.array([[1, 0, 0, 0], [0, 0, 0, -1]])),
                  'interface2': csr_matrix(np.array([[-1, 0, 0, 0]])),
                  'interface3': csr_matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))}

        self.local_problem = LinearStaticLocalProblem(1, self.K, self.B, self.f)
        self.custom_asserter = CustomDictAssertTest()

    def tearDown(self):
        pass

    def test_create_local_problem(self):
        B_input = {'interface1': csc_matrix(np.array([[1, 0, 0, 0], [0, 0, 0, -1]])),
                  'interface2': csc_matrix(np.array([[-1, 0, 0, 0]])),
                  'interface3': csc_matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))}
        test_problem = LinearStaticLocalProblem(1, self.K, B_input, self.f)

        self.assertEqual(test_problem.id, 1)
        self.assertEqual(test_problem.dimension, 4)
        self.assertTrue(isinstance(test_problem.K, Matrix))
        assert_array_equal(test_problem.K.data.todense(), self.K.todense())
        self.custom_asserter.assert_dict_equal(test_problem.B, self.B)
        assert_array_equal(test_problem.f, self.f)

        desired_config = {'pseudoinverse_config': {'method': 'svd',
                                                   'tolerance': 1e-8},
                          'preconditioner': None,
                          'scaling': None}
        self.custom_asserter.assert_dict_almost_equal(test_problem._config_dict, desired_config)

    def test_set_config(self):
        test_config = {'pseudoinverse_config': {'method': 'cholsps',
                                                'tolerance': 1e-8},
                       'tolerance': 1e-7}
        self.local_problem.set_config(test_config)
        config_desired = {'pseudoinverse_config': {'method': 'cholsps',
                                                   'tolerance': 1e-8},
                          'tolerance': 1e-7,
                          'preconditioner': None,
                          'scaling': None}
        self.custom_asserter.assert_dict_almost_equal(self.local_problem._config_dict, config_desired)

        self.setUp()
        test_config = {'pseudoinverse_config': {'method': 'splusps',
                                                'tolerance': 1e-10},
                       'preconditioner': DummyPreconditioner()}
        self.local_problem.set_config(test_config)
        config_desired = {'pseudoinverse_config': {'method': 'splusps',
                                                   'tolerance': 1e-10},
                          'preconditioner': DummyPreconditioner(),
                          'scaling': MultiplicityScaling()}

        self.custom_asserter.assert_dict_almost_equal(self.local_problem._config_dict['pseudoinverse_config'], config_desired['pseudoinverse_config'])
        self.assertTrue(isinstance(self.local_problem._config_dict['preconditioner'], DummyPreconditioner))
        self.assertTrue(isinstance(self.local_problem._config_dict['scaling'], MultiplicityScaling))

        self.setUp()
        test_config = {'tolerance': 1e-9}
        self.local_problem.set_config(test_config)
        config_desired = {'pseudoinverse_config': {'method': 'svd',
                                                   'tolerance': 1e-8},
                          'tolerance': 1e-9,
                          'preconditioner': None,
                          'scaling': None}
        self.custom_asserter.assert_dict_almost_equal(self.local_problem._config_dict, config_desired)

    def test_solve(self):
        q_b_actual = self.local_problem.solve({'interface1': np.array([-0.1, 0.3]),
                                               'interface2': np.array([0.2]),
                                               'interface3': np.array([-0.2, 0.6])})
        q_b_desired = {'interface1': np.array([0.0875, -0.0875]),
                       'interface2': np.array([-0.0875]),
                       'interface3': np.array([0.0875, -0.1375])}

        self.custom_asserter.assert_dict_almost_equal(q_b_actual, q_b_desired)

        #test additional interface not affecting local problem
        q_b_actual = self.local_problem.solve({'interface1': np.array([-0.1, 0.3]),
                                               'interface2': np.array([0.2]),
                                               'interface3': np.array([-0.2, 0.6]),
                                               'not_in_lp_interface': np.array([5, 12])})
        q_b_desired = {'interface1': np.array([0.0875, -0.0875]),
                       'interface2': np.array([-0.0875]),
                       'interface3': np.array([0.0875, -0.1375])}
        self.custom_asserter.assert_dict_almost_equal(q_b_actual, q_b_desired)

        #test external loads
        q_b_actual = self.local_problem.solve({'interface1': np.array([-0.1, 0.3]),
                                               'interface2': np.array([0.2]),
                                               'interface3': np.array([-0.2, 0.6])}, True)
        q_b_desired = {'interface1': np.array([0.65, 0.6]),
                       'interface2': np.array([-0.65]),
                       'interface3': np.array([0.65, 0.3])}
        self.custom_asserter.assert_dict_almost_equal(q_b_actual, q_b_desired)

        # test nullspace
        q_b_actual = self.local_problem.solve({'interface1': np.array([-0.1, 0.3]),
                                               'interface2': np.array([0.2]),
                                               'interface3': np.array([-0.2, 0.6])}, True, np.array([0.1]))
        q_b_desired = {'interface1': np.array([0.7 , 0.55]),
                       'interface2': np.array([-0.7]),
                       'interface3': np.array([0.7 , 0.35])}
        self.custom_asserter.assert_dict_almost_equal(q_b_actual, q_b_desired)

    def test_expand_external_solution(self):
        external_solution_dict = {'interface1': np.array([-0.1, 0.3]),
                                  'interface2': np.array([0.2]),
                                  'interface3': np.array([-0.2, 0.6]),
                                  'not_in_lp_interface': np.array([5, 12])}

        lamda_actual = self.local_problem._expand_external_solution(external_solution_dict)
        lamda_desired = np.array([-0.5, 0.6, 0, -0.3])
        assert_array_almost_equal(lamda_actual, lamda_desired)

    def test_distribute_to_interfaces(self):
        q = np.array([0.1, 0.05, 0.02, 0.12])
        q_b_dict_actual = self.local_problem._distribute_to_interfaces(q)
        q_b_dict_desired = {'interface1': np.array([0.1, -0.12]),
                            'interface2': np.array([-0.1]),
                            'interface3': np.array([0.1, 0.05])}
        self.custom_asserter.assert_dict_almost_equal(q_b_dict_actual, q_b_dict_desired)

    def test_update_preconditioner_and_scaling(self):
        self.local_problem.update_preconditioner_and_scaling()

        self.assertEqual(self.local_problem.preconditioner, None)
        self.assertEqual(self.local_problem.scaling, None)

        self.local_problem.preconditioner = DummyPreconditioner()
        self.local_problem.scaling = MultiplicityScaling()

        self.assertEqual(self.local_problem.preconditioner.Q, None)

        self.local_problem.update_preconditioner_and_scaling()

        Q_desired = np.array([[2, -2, 0], [-2, 4, 0], [0, 0, 2]])
        assert_array_equal(self.local_problem.preconditioner.Q.todense(), Q_desired)

        scaling_dict_desired = {'interface1': csr_matrix(np.array([[1/4, 0], [0, 1/2]])),
                                'interface2': csr_matrix(np.array([[1/4]])),
                                'interface3': csr_matrix(np.array([[1/4, 0], [0, 1/2]]))}

        self.custom_asserter.assert_dict_almost_equal(self.local_problem.scaling.scaling_dict, scaling_dict_desired)

    def test_precondition(self):
        #No preconditioner
        external_solution_dict = {'interface1': np.array([-0.1, 0.3]),
                                  'interface2': np.array([0.2]),
                                  'interface3': np.array([-0.2, 0.6]),
                                  'not_in_lp_interface': np.array([5, 12])}

        preconditioned_solution = self.local_problem.precondition(external_solution_dict)

        preconditioned_solution_desired = {'interface1': np.array([-0.1, 0.3]),
                                           'interface2': np.array([0.2]),
                                           'interface3': np.array([-0.2, 0.6]),
                                           'not_in_lp_interface': np.array([5, 12])}

        self.custom_asserter.assert_dict_almost_equal(preconditioned_solution, preconditioned_solution_desired)

        #DirichletPreconditioner + MultiplicityScaling
        self.local_problem.preconditioner = DirichletPreconditioner()
        self.local_problem.scaling = MultiplicityScaling()
        self.local_problem.update_preconditioner_and_scaling()

        preconditioned_solution = self.local_problem.precondition(external_solution_dict)

        preconditioned_solution_desired = {'interface1': np.array([-0.2125, 0.225]),
                                           'interface2': np.array([0.2125]),
                                           'interface3': np.array([-0.2125, 0.65])}

        self.custom_asserter.assert_dict_almost_equal(preconditioned_solution, preconditioned_solution_desired)


class LinearDynamicLocalProblemTest(TestCase):
    def setUp(self):
        class DummyIntegrator(IntegratorBase):
            def __init__(self):
                super().__init__()
                self.J = np.array([[3, -2, 0, 0], [-2, 5, -2, 0], [0, -2, 5, -2], [0, 0, -2, 3]])
                self.dt = 0.01
                self._f_ext = np.array([0, 0, 0, 2])
                self.beta = 0.25
                self.gamma = 0.5

            def jacobian(self, dq_p):
                return self.J

            def residual_int(self, dq_p):
                return self.J @ dq_p

            def residual_ext(self, dq_p):
                return self._f_ext * self.t_p

            def set_prediction(self, q_n, dq_n, ddq_n, t_n):
                self._t_n = t_n
                self._q_n = q_n
                self._dq_n = dq_n
                self._ddq_n = ddq_n

                self.q_p = self._q_n + self.dt * self._dq_n + self.dt ** 2 * (1 / 2 - self.beta / self.gamma) * \
                           self._ddq_n
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

        integrator = DummyIntegrator()
        self.B = {'interface1': csr_matrix(np.array([[1, 0, 0, 0], [0, 0, 0, -1]])),
                  'interface2': csr_matrix(np.array([[-1, 0, 0, 0]])),
                  'interface3': csr_matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))}

        self.local_problem = LinearDynamicLocalProblem(1, integrator, self.B, {'t0': 0,
                                                                               'q0': np.zeros(4),
                                                                               'dq0': np.zeros(4),
                                                                               'ddq0': np.zeros(4)})
        self.custom_asserter = CustomDictAssertTest()

    def tearDown(self):
        pass

    def test_store_solutions(self):
        self.assertEqual(len(self.local_problem.q.keys()), 1)
        self.assertAlmostEqual(self.local_problem.t[0], 0)
        assert_array_almost_equal(self.local_problem.q[0], np.zeros(4))
        assert_array_almost_equal(self.local_problem.dq[0], np.zeros(4))
        assert_array_almost_equal(self.local_problem.ddq[0], np.zeros(4))

        self.local_problem._store_solutions(0.1, np.array([0.3, -0.2, 0.187, 0.0]), np.array([0.1, 0.2, 0.3, 0.4]),
                                            np.array([-11.65, 4.321, -0.6871, 6.43]))

        assert_array_almost_equal(self.local_problem.t, np.array([0, 0.1]))
        assert_array_almost_equal(self.local_problem.q[1], np.array([0.3, -0.2, 0.187, 0.0]))
        assert_array_almost_equal(self.local_problem.dq[1], np.array([0.1, 0.2, 0.3, 0.4]))
        assert_array_almost_equal(self.local_problem.ddq[1], np.array([-11.65, 4.321, -0.6871, 6.43]))

    def test_update_system(self):
        self.local_problem._dq_p = np.array([0.7, -0.3, 0.95, -1.12])
        self.local_problem._delta_dq = np.array([0.0, 0.1, -0.43, 2.5])

        self.local_problem.update_system({'interface1': np.array([0.1, 0.42]),
                                          'interface2': np.array([0.64]),
                                          'interface3': np.array([-0.67, 1.32])}, {'test_dict': True})

        assert_array_almost_equal(self.local_problem._dq_p, np.array([0.7, -0.2, 0.52, 1.38]))
        assert_array_almost_equal(self.local_problem.t, np.array([0, 0.01]))

        assert_array_almost_equal(self.local_problem.q[1], np.array([0.0035, -0.001, 0.0026, 0.0069]))
        assert_array_almost_equal(self.local_problem.dq[1], np.array([0.7, -0.2, 0.52, 1.38]))
        assert_array_almost_equal(self.local_problem.ddq[1], np.array([140.0, -40.0, 104.0, 276.0]))

        assert_array_almost_equal(self.local_problem._integrator._q_n, np.array([0.0035, -0.001, 0.0026, 0.0069]))
        assert_array_almost_equal(self.local_problem._integrator._dq_n, np.array([0.7, -0.2, 0.52, 1.38]))
        assert_array_almost_equal(self.local_problem._integrator._ddq_n, np.array([140.0, -40.0, 104.0, 276.0]))

        assert_array_almost_equal(self.local_problem.f, np.array([-2.5, 3.44, -0.24, -3.14]))

    def test_dump_local_information(self):
        self.local_problem._integrator.t_p = 0.02

        info_dict = self.local_problem.dump_local_information()

        assert_array_almost_equal(info_dict['t'], 0.02)

    def test_solve(self):
        # Test pure interface-forces
        u_b_dict_actual = self.local_problem.solve({'interface1': np.array([0.1, 0.3]),
                                                    'interface2': np.array([0.64]),
                                                    'interface3': np.array([-0.67, 1.4])})

        u_b_dict_desired = {'interface1': np.array([0.278, -0.068]),
                            'interface2': np.array([-0.278]),
                            'interface3': np.array([0.278, -0.188])}

        self.custom_asserter.assert_dict_almost_equal(u_b_dict_actual, u_b_dict_desired)

        # Test with external forces (here: residuals)
        u_b_dict_actual = self.local_problem.solve({'interface1': np.array([0.1, 0.3]),
                                                    'interface2': np.array([0.64]),
                                                    'interface3': np.array([-0.67, 1.4])}, True)

        u_b_dict_desired = {'interface1': np.array([0.27988235, -0.07811765]),
                            'interface2': np.array([-0.27988235]),
                            'interface3': np.array([0.27988235, -0.18517647])}

        self.custom_asserter.assert_dict_almost_equal(u_b_dict_actual, u_b_dict_desired)

        # Test, that given rigid-body-modes have no influence on the solution
        u_b_dict_actual = self.local_problem.solve({'interface1': np.array([0.1, 0.3]),
                                                    'interface2': np.array([0.64]),
                                                    'interface3': np.array([-0.67, 1.4])}, True, np.array([0.1]))

        u_b_dict_desired = {'interface1': np.array([0.27988235, -0.07811765]),
                            'interface2': np.array([-0.27988235]),
                            'interface3': np.array([0.27988235, -0.18517647])}

        self.custom_asserter.assert_dict_almost_equal(u_b_dict_actual, u_b_dict_desired)



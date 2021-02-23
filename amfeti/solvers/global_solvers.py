#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Solver module, that provides customized solvers for the global problem
"""
from amfeti.config_base import ConfigBase
import logging
import numpy as np
import time
from copy import copy
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve

__all__ = ['PCPGsolver',
           'GMRESsolver',
           'ORTHOMINsolver']


class GlobalSolverBase(ConfigBase):
    def __init__(self):
        super().__init__()

    def solve(self, *args):
        return None


class PCPGsolver(GlobalSolverBase):
    """
    Preconditioned Conjugate Projected Gradient-iterative solver, that is usually used to solve the linear global
    problem. This solver is an extension of the well-known iterative Conjugate Gradient methods by a preconditioner and
    a nullspace-projection for singular problems. Moreover, this solver supports full reorthogonalization, which is able
    to improve convergence, if F-orthogonality degrades during the iterations.

    References
    ----------
    [1]  D.J. Rixen and C. Farhat (1999): A simple and efficient extension of a class of substructure based
         preconditioners to heterogeneous structural mechanics problems. International Journal for Numerical Methods in
         Engineering 44 489--516.
    [2]  C. Farhat and F.X. Roux (1994): Implicit parallel processing in structural mechanics. Computational Mechanics
         Advances 2 1--124.
    [3]  M.C. Leistner, P. Gosselet, D.J. Rixen (2018): Recycling of solution spaces in multipreconditioned FETI methods
         applied to structural dynamics. International Journal of Numerical Methods in Engineering 116 141--160
         doi:10.1002/nme.5918.

    Attributes
    ----------
    _config_dict : dict
        configuration dictionary
    """
    def __init__(self):
        """
        Parameters
        ----------
        None
        """
        super().__init__()
        self._config_dict = {'tolerance': 1e-7,
                             'max_iter': None,
                             'projection': None,
                             'precondition': None,
                             'energy_norm': False,
                             'save_history': False,
                             'full_reorthogonalization': False}

    def solve(self, F_callback, residual_callback, lambda_init):
        """
        Solve-method of the PCPG-method

        Parameters
        ----------
        F_callback : callable
            method, that applies the solution-vector on the system-matrix F and returns the result
        residual_callback : callable
            method, that calculates and return the system's residual from the solution-vector
        lambda_init : ndarray
            initial guess for the solution

        Returns
        -------
        lambda_sol : ndarray
            calculated solution
        info_dict : dict
            general information on the solution-process
        """
        logger = logging.getLogger(__name__)

        interface_size = len(lambda_init)

        if self._config_dict['max_iter'] is None:
            self._config_dict['max_iter'] = int(1 * interface_size)

        logger.info('Setting PCPG tolerance = %4.2e' % self._config_dict['tolerance'])
        logger.info('Setting PCPG max number of iterations = %i' % self._config_dict['max_iter'])

        # initialize variables
        info_dict = {}
        global_start_time = time.time()
        residual_hist = np.array([])
        lambda_hist = np.array([])
        if self._config_dict['full_reorthogonalization']:
            Y = dict()
            Q = dict()
        lambda_sol = np.zeros_like(lambda_init)
        rk = residual_callback(lambda_init)
        k = 0

        for k in range(self._config_dict['max_iter']):
            info_dict[k] = {}
            wk = self._project(rk)
            zk = self._precondition(wk)
            yk = self._project(zk)

            if self._config_dict['full_reorthogonalization']:
                yk1 = yk
                for i in range(k):
                    yki = Y[i]
                    qki = Q[i]
                    yk -= np.dot(qki, yk1) / np.dot(qki, yki) * yki
            elif k > 0:
                yk -= np.dot(qk_1, yk) / np.dot(qk_1, yk_1) * yk_1

            vn1 = np.dot(wk, zk)

            if self._config_dict['save_history']:
                lambda_hist = np.append(lambda_hist, lambda_sol)
                residual_hist = np.append(residual_hist, np.linalg.norm(rk))

            if self._config_dict['energy_norm']:
                norm_wk = np.sqrt(vn1)
                logging.info(
                    'Iteration = %i, Norm of project preconditioned residual  sqrt(<yk,wk>) = %2.5e!' % (k, norm_wk))
                if norm_wk <= self._config_dict['tolerance']:
                    # evaluate the exact norm
                    _norm_wk = np.linalg.norm(wk)
                    if _norm_wk <= self._config_dict['tolerance']:
                        logger.info('PCPG has converged after %i' % (k + 1))
                        logger.info('Iteration = %i, Norm of project residual wk = %2.5e!' % (k, _norm_wk))
                        break
            else:
                norm_wk = np.linalg.norm(wk)
                if norm_wk <= self._config_dict['tolerance']:
                    logger.info('PCPG has converged after %i' % (k + 1))
                    break

            Fyk = F_callback(yk)
            if self._config_dict['full_reorthogonalization']:
                Y[k] = copy(yk)
                Q[k] = copy(Fyk)
            else:
                yk_1 = copy(yk)
                qk_1 = copy(Fyk)

            aux2 = np.linalg.norm(np.dot(yk, Fyk))
            alpha_k = float(vn1 / aux2)

            lambda_sol += alpha_k * yk

            rk -= alpha_k * Fyk

        lambda_sol = self._project(lambda_sol) + lambda_init

        if (k > 0) and k == (self._config_dict['max_iter'] - 1) and norm_wk > self._config_dict['tolerance']:
            logger.warning('Maximum iteration was reached, MAX_INT = %i, without converging!' % (k + 1))
            logger.warning('Projected norm = %2.5e , where the PCPG tolerance is set to %2.5e' % (norm_wk, self._config_dict['tolerance']))

        elapsed_time = time.time() - global_start_time
        logger.info('#' * 60)
        logger.info('{"Total_elaspsed_time_PCPG" : %2.2f} # Elapsed time [s]' % (elapsed_time))
        logger.info('Number of PCPG Iterations = %i !' % (k + 1))
        avg_iteration_time = elapsed_time / (k + 1)
        logger.info('{"avg_iteration_time_PCPG" : %2.4f} # Elapsed time [s]' % (avg_iteration_time))
        logger.info('#' * 60)

        info_dict['avg_iteration_time'] = elapsed_time / (k + 1)
        info_dict['Total_elaspsed_time_PCPG'] = elapsed_time
        info_dict['PCPG_iterations'] = k + 1
        info_dict['lambda_hist'] = lambda_hist
        info_dict['residual_hist'] = residual_hist
        info_dict['residual'] = norm_wk
        return lambda_sol, info_dict

    def _precondition(self, v):
        if self._config_dict['precondition'] is not None:
            precondition = self._config_dict['precondition']
            v = precondition(v)
            return v
        else:
            return copy(v)

    def _project(self, v):
        if self._config_dict['projection'] is not None:
            project = self._config_dict['projection']
            v = project(v)
        return v


class GMRESsolver(PCPGsolver):
    """
    Generalized Minimal RESidual method as another option for a linear iterative solver, that is also able to solve
    non-symmetric problems, but is more expensive in terms of memory than the CG-method.

    Attributes
    ----------
    _config_dict : dict
        configuration dictionary
    """
    def __init__(self):
        """
        Parameters
        ----------
        None
        """
        super().__init__()
        self._config_dict = {'tolerance': 1e-9,
                             'max_iter': None,
                             'projection': None,
                             'precondition': None,
                             'energy_norm': False,
                             'save_history': False}

    def solve(self, F_callback, residual_callback, lambda_init):
        """
        Solve-method of the GMRES-method

        Parameters
        ----------
        F_callback : callable
            method, that applies the solution-vector on the system-matrix F and returns the result
        residual_callback : callable
            method, that calculates and return the system's residual from the solution-vector
        lambda_init : ndarray
            initial guess for the solution

        Returns
        -------
        lambda_sol : ndarray
            calculated solution
        info_dict : dict
            general information on the solution-process
        """
        logger = logging.getLogger(__name__)

        interface_size = len(lambda_init)

        if self._config_dict['max_iter'] is None:
            self._config_dict['max_iter'] = int(1 * interface_size)

        logger.info('Setting GMRES tolerance = %4.2e' % self._config_dict['tolerance'])
        logger.info('Setting GMRES max number of iterations = %i' % self._config_dict['max_iter'])

        # initialize variables
        info_dict = {}
        global_start_time = time.time()
        residual_hist = np.array([])
        lambda_hist = np.array([])

        V = dict()
        H = csr_matrix(np.zeros((2, 1)),dtype=complex)

        lambda_sol = np.zeros_like(lambda_init,dtype=complex)
        rk = residual_callback(lambda_init)
        residual_0 = copy(rk)
        lambda_0 = copy(lambda_init)
        k = 0

        wk = self._project(rk)
        zk = self._precondition(wk)
        yk = self._project(zk)
        beta = np.linalg.norm(zk)
        norm_vk = np.linalg.norm(yk)
        V[0] = yk / norm_vk
        vk_stack = V[0][np.newaxis].T
        V_stack = csr_matrix(vk_stack)

        for k in range(self._config_dict['max_iter']):
            info_dict[k] = {}
            Fvk = F_callback(V[k])

            wk = self._project(Fvk)
            zk = self._precondition(wk)
            yk = self._project(zk)

            if self._config_dict['save_history']:
                lambda_hist = np.append(lambda_hist, lambda_sol)
                residual_hist = np.append(residual_hist, np.linalg.norm(rk))

            if self._config_dict['energy_norm']:
                norm_wk = np.sqrt(np.vdot(wk, zk))
                logging.info(
                    'Iteration = %i, Norm of project preconditioned residual  sqrt(<yk,wk>) = %2.5e!' % (k, norm_wk))
                if norm_wk <= self._config_dict['tolerance']:
                    # evaluate the exact norm
                    _norm_wk = np.linalg.norm(wk)
                    if _norm_wk <= self._config_dict['tolerance']:
                        logger.info('GMRES has converged after %i' % (k + 1))
                        logger.info('Iteration = %i, Norm of project residual wk = %2.5e!' % (k, _norm_wk))
                        break
            else:
                norm_wk = np.linalg.norm(rk)
                if norm_wk <= self._config_dict['tolerance']:
                    logger.info('GMRES has converged after %i' % (k + 1))
                    break

            vk = yk

            H = csr_matrix(H)
            for i in range(k+1):
                hik = np.vdot(vk.T, V[i])
                H[i, k] = hik
                vk -= hik * V[i]

            norm_vk = np.linalg.norm(vk)
            H[k + 1, k] = norm_vk

            e1 = csr_matrix(np.zeros(k + 2))
            e1[0, 0] = 1

            uj = spsolve(H.T @ H, H.T @ (beta * e1).T)

            Vuj = V_stack @ uj

            lambda_sol = lambda_0 + Vuj.T

            rk = residual_0 - F_callback(Vuj)

            if k is not (self._config_dict['max_iter'] - 1):
                V[k + 1] = vk / H[k + 1, k]
                vk_stack = V[k + 1][np.newaxis].T
                V_stack = hstack((V_stack, vk_stack))

                H = hstack((H, csr_matrix(np.zeros((H.shape[0], 1)))))
                H = vstack((H, csr_matrix(np.zeros((1, H.shape[1])))))

        lambda_sol = self._project(lambda_sol) + lambda_init

        if (k > 0) and k == (self._config_dict['max_iter'] - 1) and norm_wk > self._config_dict['tolerance']:
            logger.warning('Maximum iteration was reached, MAX_INT = %i, without converging!' % (k + 1))
            logger.warning('Projected norm = %2.5e , where the GMRES tolerance is set to %2.5e' % (
            norm_wk, self._config_dict['tolerance']))

        elapsed_time = time.time() - global_start_time
        logger.info('#' * 60)
        logger.info('{"Total_elaspsed_time_PCPG" : %2.2f} # Elapsed time [s]' % (elapsed_time))
        logger.info('Number of GMRES Iterations = %i !' % (k + 1))
        avg_iteration_time = elapsed_time / (k + 1)
        logger.info('{"avg_iteration_time_GMRES" : %2.4f} # Elapsed time [s]' % (avg_iteration_time))
        logger.info('#' * 60)

        info_dict['avg_iteration_time'] = elapsed_time / (k + 1)
        info_dict['Total_elaspsed_time_GMRES'] = elapsed_time
        info_dict['GMRES_iterations'] = k + 1
        info_dict['lambda_hist'] = lambda_hist
        info_dict['residual_hist'] = residual_hist
        info_dict['residual'] = norm_wk

        return lambda_sol, info_dict

class ORTHOMINsolver(PCPGsolver):


    """
    Orthomin method as another option for a linear iterative solver, that is also able to solve
    non-symmetric problems, minimizes the true residual in the iterative step. Expensive than GMRES
    in terms of memory

    Attributes
    ----------
    _config_dict : dict
        configuration dictionary
    """
    def __init__(self):
        """
        Parameters
        ----------
        None
        """
        super().__init__()
        self._config_dict = {'tolerance': 1e-8,
                             'max_iter': None,
                             'projection': None,
                             'precondition': None,
                             'energy_norm': False,
                             'save_history': False}

    def solve(self, F_callback, residual_callback, lambda_init):
        """
        Solve-method of the Orthomin-method

        Parameters
        ----------
        F_callback : callable
            method, that applies the solution-vector on the system-matrix F and returns the result
        residual_callback : callable
            method, that calculates and return the system's residual from the solution-vector
        lambda_init : ndarray
            initial guess for the solution

        Returns
        -------
        lambda_sol : ndarray
            calculated solution
        info_dict : dict
            general information on the solution-process
        """
        logger = logging.getLogger(__name__)

        interface_size = len(lambda_init)

        if self._config_dict['max_iter'] is None:
            self._config_dict['max_iter'] = int(1 * interface_size)

        logger.info('Setting ORTHOMIN tolerance = %4.2e' % self._config_dict['tolerance'])
        logger.info('Setting ORTHOMIN max number of iterations = %i' % self._config_dict['max_iter'])

        # initialize variables
        info_dict = {}
        global_start_time = time.time()
        residual_hist = np.array([])
        lambda_hist = np.array([])
        projected_subspace = []

        V = dict()
        Proj_V = dict()


        lambda_sol = np.zeros_like(lambda_init)
        rk = residual_callback(lambda_init)
        wk = self._project(rk)
        zk = self._precondition(wk)
        yk = self._project(zk)


        V[0] = yk
        Proj_V[0] = F_callback(V[0])


        for k in range(self._config_dict['max_iter']):
            info_dict[k] = {}
            alpha = np.dot(np.conjugate(Proj_V[k]), rk) / np.dot(np.conjugate(Proj_V[k]), Proj_V[k])
            lambda_sol = lambda_sol + V[k] * alpha

            rk = rk - alpha * Proj_V[k]
            wk = self._project(rk)
            zk = self._precondition(wk)
            yk = self._project(zk)

            if self._config_dict['save_history']:
                lambda_hist = np.append(lambda_hist, lambda_sol)
                residual_hist = np.append(residual_hist, np.linalg.norm(rk))
                projected_subspace = np.append(projected_subspace, Proj_V[k])

                norm_wk = np.linalg.norm(rk)
                if norm_wk <= self._config_dict['tolerance']:
                    logger.info('Orthomin has converged after %i' % (k + 1))
                    break

            else:
                norm_wk = np.linalg.norm(rk)
                if norm_wk <= self._config_dict['tolerance']:
                    logger.info('Orthomin has converged after %i' % (k + 1))
                    break

            AZ1 = F_callback(yk)
            V[k+1] = yk
            Proj_V[k+1] = AZ1

            for i in range(k+1):
                beta  = np.dot(np.conjugate(Proj_V[i]),Proj_V[k+1] ) / np.dot(np.conjugate(Proj_V[i]), Proj_V[i])
                V[k+1] = V[k+1] -  beta*V[i]
                Proj_V[k+1] = Proj_V[k+1] -  beta*Proj_V[i]



        lambda_sol = self._project(lambda_sol) + lambda_init

        if (k > 0) and k == (self._config_dict['max_iter'] - 1) and norm_wk > self._config_dict['tolerance']:
            logger.warning('Maximum iteration was reached, MAX_INT = %i, without converging!' % (k + 1))
            logger.warning('Projected norm = %2.5e , where the Orthomin tolerance is set to %2.5e' % (
            norm_wk, self._config_dict['tolerance']))

        elapsed_time = time.time() - global_start_time
        logger.info('#' * 60)
        logger.info('{"Total_elaspsed_time_ORTHOMIN" : %2.2f} # Elapsed time [s]' % (elapsed_time))
        logger.info('Number of ORTHOMIN Iterations = %i !' % (k + 1))
        avg_iteration_time = elapsed_time / (k + 1)
        logger.info('{"avg_iteration_time_ORTHOMIN" : %2.4f} # Elapsed time [s]' % (avg_iteration_time))
        logger.info('#' * 60)

        info_dict['avg_iteration_time'] = elapsed_time / (k + 1)
        info_dict['Total_elaspsed_time_ORTHOMIN'] = elapsed_time
        info_dict['Iterations'] = k + 1
        info_dict['lambda_hist'] = lambda_hist
        info_dict['residual_hist'] = residual_hist
        info_dict['residual'] = norm_wk
        info_dict['Projected_search_direction'] = Proj_V

        return lambda_sol, info_dict


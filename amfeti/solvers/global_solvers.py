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
from amfeti.linalg.datatypes import Matrix,Pseudoinverse
from scipy import linalg as spp
from copy import copy
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


__all__ = ['PCPGsolver',
           'GMRESsolver',
           'ORTHOMINsolver',
           'M_ORTHOMIN']


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
                             'projection': False,
                             'precondition': False,
                             'energy_norm': False,
                             'save_history': True,
                             'full_reorthogonalization': True}

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
        AuxilaryVariable1 = np.array([])
        AuxilaryVariable2 = np.array([])

        OrthogonalProduct = np.array([])
        for k in range(self._config_dict['max_iter']+20):


            info_dict[k] = {}

            # Project the residual to get a better guess
            wk = self._project(rk)

            # Precondition the projected residual to compute appropriate forcing corrections
            zk = self._precondition(wk)

            # Re-projection
            yk = self._project(zk)



            if self._config_dict['full_reorthogonalization']:

                # yk1 is the projected search direction for the current iteration
                yk1 = yk

                for i in range(k):
                    yki = Y[i]
                    qki = Q[i]
                    # print('Basis error is', np.linalg.norm(qki - F_callback(yki)))
                    beta_coeff = np.dot(Q[i], yk1) / np.dot(Q[i], Y[i])
                    yk = yk -  beta_coeff*Y[i]

                # yk = yk/ np.linalg.norm(yk)
            elif k > 0:
                yk -= np.dot(qk_1, yk) / np.dot(qk_1, yk_1) * yk_1


            print('The norm of yk is :', np.linalg.norm(yk))


            # Alpha numerator depends on residual and the preconditioned residual
            Alpha_numerator = np.dot(rk, zk)

            if self._config_dict['save_history']:
                lambda_hist = np.append(lambda_hist, lambda_sol)
                residual_hist = np.append(residual_hist, np.linalg.norm(rk))

            if self._config_dict['energy_norm']:
                norm_wk = np.sqrt(Alpha_numerator)
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

            qik = F_callback(yk)
            print('The norm of Fyk is:', np.linalg.norm(qik))
            if self._config_dict['full_reorthogonalization']:
                Y[k] = copy(yk)
                Q[k] = copy(qik)

                if k> 0:
                    OrthogonalProduct_k = np.dot(np.conjugate(Y[k]), F_callback(Y[k - 1]))
                    OrthogonalProduct = np.append(OrthogonalProduct, OrthogonalProduct_k)

            else:
                yk_1 = copy(yk)
                qk_1 = copy(qik)


            Alpha_denominator = np.linalg.norm(np.dot(yk, qik))
            # aux2 = ScalingParameter*aux2
            alpha_k = float((Alpha_numerator) /Alpha_denominator)



            AuxilaryVariable1= np.append(AuxilaryVariable1,Alpha_denominator)
            AuxilaryVariable2 = np.append(AuxilaryVariable2,Alpha_numerator)

            lambda_sol += alpha_k * yk

            print('Actual residual is ', np.linalg.norm(residual_callback(lambda_sol)))


            rk -= alpha_k * qik


            print('Normalized PCPG residual is ', np.linalg.norm(rk))
        # lambda_sol = self._project(lambda_sol) + lambda_init
        a=4


        plt.figure()
        plt.plot(range(k),OrthogonalProduct[range(k)])
        plt.xlabel('Iteration count')
        plt.ylabel('Norm of the orthogonal SD check')
        plt.yscale('linear')
        # plt.ylim(1e-6, 1)
        plt.title('The orthogonality in CG for projected search directions')
        plt.show()

        plt.figure()
        plt.plot(range(k),AuxilaryVariable1[range(k)], label="Norm of y'*Fy")
        plt.plot(range(k), AuxilaryVariable2[range(k)],  label="minimization parameter - alpha")
        plt.xlabel('Iteration count')
        plt.ylabel('Norm of the measured quantities')
        plt.yscale('log')
        plt.ylim(1e-15, 1e15)

        plt.legend('Norm of yT,Fy', 'minimization parameter - alpha')
        plt.title('CG convergence issues')
        plt.show()


        plt.figure()
        plt.plot(range(k+1),residual_hist , label = 'Convergence of the CG')
        plt.yscale('log')
        plt.ylim(1e-9, 1e4)
        plt.show()
        # plt.legend()
        plt.title('CG convergence issues')


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
        info_dict['Iterations'] = k + 1
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
        self._config_dict = {'tolerance': 1e-7,
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
        info_dict['Iterations'] = k + 1
        info_dict['lambda_hist'] = lambda_hist
        info_dict['residual_hist'] = residual_hist
        info_dict['residual'] = norm_wk

        return lambda_sol, info_dict

class M_ORTHOMIN(PCPGsolver):
    """

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
                             'energy_norm': False,
                             'save_history': True,
                             'full_reorthogonalization': True}

    def solve(self, F_callback, F_callback_single_precond, residual_callback, lambda_init):
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

        """ Initialize the storage vectors"""
        W_dict = {}
        Q_dict = {}
        delta_dict ={}

        """ Initialize the solution and residual vectors"""
        lambda_sol = np.zeros_like(lambda_init)
        rk = residual_callback(lambda_init)
        InitialResidual = rk
        Intialnorm = np.linalg.norm(rk)
        zk = self._precondition(rk)

        SearchDirections = zk
        ProjectedSearchDirections=self.F_callback_loop(SearchDirections, F_callback_single_precond)

        Qorthoginalized, SearchDirectionsorthoginalized = self.OrthogonalizeVectors(ProjectedSearchDirections, SearchDirections)
        Qupdated, SearchDirectionsUpdated = self.DeleteZeroSearchdirection(Qorthoginalized, SearchDirectionsorthoginalized)
        Q_dict[0], W_dict[0] = self.OrthonormalStep(Qupdated, SearchDirectionsUpdated)

        print('starting loop')

        NumberOfVectors = Q_dict[0].shape[1]
        for k in range(self._config_dict['max_iter']):
            info_dict[k] = {}
            Minimizationstep = np.dot(np.conjugate(Q_dict[k].T),rk)
            AlphaParameter =Minimizationstep

            lambda_sol = lambda_sol + np.dot(W_dict[k],AlphaParameter)
            rk = rk - np.dot(Q_dict[k], (AlphaParameter))
            ActualResidual = np.linalg.norm(residual_callback(lambda_sol))

            print('Residue        = ', np.linalg.norm(rk))
            print('Actual Residue = ', ActualResidual)
            
            wk = self._project(rk)
            zk = self._precondition(wk)
            SearchDirections = self._project(zk)
            ProjectedSearchDirections = self.F_callback_loop(SearchDirections, F_callback_single_precond)

            W_dict[k+1] = SearchDirections
            Q_dict[k+1] = ProjectedSearchDirections
            if self._config_dict['full_reorthogonalization']:
                for i in range(k+1):
                    Numerator = np.dot( np.conjugate(Q_dict[i].T), Q_dict[k + 1])
                    beta_ik   = Numerator
                    W_dict[k+1] = W_dict[k+1] - np.dot(W_dict[i],(beta_ik))
                    Q_dict[k+1] = Q_dict[k+1] - np.dot(Q_dict[i],(beta_ik))

            if (np.linalg.norm(Q_dict[k + 1] - self.F_callback_loop(W_dict[k + 1], F_callback_single_precond))) >= 1e-6:
                print("-----the basis is inconsistent after orthogonalization wrt vector blocks-----")

            Qorthoginalized, SearchDirectionsorthoginalized = self.OrthogonalizeVectors(Q_dict[k+1], W_dict[k+1])
            Qupdated, SearchDirectionsUpdated = self.DeleteZeroSearchdirection(Qorthoginalized, SearchDirectionsorthoginalized)
            Q_dict[k + 1], W_dict[k + 1] = self.OrthonormalStep(Qupdated, SearchDirectionsUpdated)
            NumberOfVectors = NumberOfVectors + Q_dict[k+1].shape[1]
            print('The number of vectors stored is                                      :', Qupdated.shape[1])
            print('the rank of the matrix of projected search direction after filtering :', np.linalg.matrix_rank(Qupdated))
            if (np.linalg.norm(Q_dict[k + 1] - self.F_callback_loop(W_dict[k + 1], F_callback_single_precond))) >= 1e-6:
                print("-----the basis is inconsistent after orthogonalization wrt each other-----")

            if self._config_dict['save_history']:
                lambda_hist = np.append(lambda_hist, lambda_sol)
                residual_hist = np.append(residual_hist, np.linalg.norm(rk))

            if self._config_dict['energy_norm']:
                norm_wk = np.sqrt(vn1)
                logging.info(
                    'Iteration = %i, Norm of project preconditioned residual  sqrt(<yk,wk>) = %2.5e!' % (k, norm_wk))
                if norm_wk <= self._config_dict['tolerance']:
                    _norm_wk = np.linalg.norm(wk)
                    if _norm_wk <= self._config_dict['tolerance']:
                        logger.info('PCPG has converged after %i' % (k + 1))
                        logger.info('Iteration = %i, Norm of project residual wk = %2.5e!' % (k, _norm_wk))
                        break
            else:
                norm_wk = np.linalg.norm(rk)
                if norm_wk <= self._config_dict['tolerance']:
                    logger.info('PCPG has converged after %i' % (k + 1))
                    break

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
        info_dict['Iterations'] = NumberOfVectors
        info_dict['lambda_hist'] = lambda_hist
        info_dict['residual_hist'] = residual_hist
        info_dict['residual'] = norm_wk
        return lambda_sol, info_dict

    def F_callback_loop(self, SD, F_callback_single_precond):
        ProjecctedSD = np.zeros((SD.shape[0], SD.shape[1]), dtype=complex)
        for iCounter in range(SD.shape[1]):
            ProjecctedSD[:, iCounter] = F_callback_single_precond(SD[:, iCounter])
        return ProjecctedSD

    def OrthonormalStep(self, NDArrayProjVectors, NDArraySearchVectors):
        for iCounter in range(NDArrayProjVectors.shape[1]):
            NDArrayProjVectors[:, iCounter]   = NDArrayProjVectors[:, iCounter]/ np.linalg.norm(NDArrayProjVectors[:, iCounter])
            NDArraySearchVectors[:, iCounter] = NDArraySearchVectors[:, iCounter] /np.linalg.norm(NDArrayProjVectors[:, iCounter])
        return NDArrayProjVectors, NDArraySearchVectors

    def OrthogonalizeVectors(self, NDArrayProjVectors, NDArraySearchVectors):
        QtQ = np.dot(np.conjugate(NDArrayProjVectors.T), (NDArrayProjVectors))
        GetRank = np.linalg.matrix_rank(NDArrayProjVectors)
        print('the rank of the matrix of projected search direction is :', GetRank)
        Ldecomp, Ddecomp, perm = spp.ldl(QtQ, hermitian=True)

        Linverse = np.linalg.pinv(np.conjugate(Ldecomp.T))
        Dinverse = np.linalg.pinv(np.sqrt(Ddecomp))

        Qupdated = np.dot(NDArrayProjVectors, np.dot(Linverse, Dinverse))
        SearchDirectionsUpdated = np.dot(NDArraySearchVectors, np.dot(Linverse, Dinverse))
        return Qupdated, SearchDirectionsUpdated

    def DeleteZeroSearchdirection(self, NDArrayProjVectors, NDArraySearchVectors, tol = 1e-6):
        StoreColumnId = np.empty((0), dtype=int)
        for iCounter in range(NDArrayProjVectors.shape[1]):
            if np.linalg.norm(NDArrayProjVectors[:, iCounter]) < tol:
                StoreColumnId = np.append(StoreColumnId, iCounter)

        NDArrayProjVectors   = np.delete(NDArrayProjVectors, StoreColumnId, axis=1)
        NDArraySearchVectors = np.delete(NDArraySearchVectors, StoreColumnId, axis=1)
        return NDArrayProjVectors, NDArraySearchVectors

class ORTHOMINsolver(PCPGsolver):


    """
    Orthomin iterative scheme solves generalized linear systems (complex, non-symmetric, badly conditioned) similar to GMRES.
    Unlike, GMRES, it solves the minimization problem at each step and thus the residual is available at each step. On the other hand,
    one has to store additional varaibles compared to GMRES between iterations.
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

        V = dict()
        Q = dict()


        lambda_sol = np.zeros_like(lambda_init)
        rk = residual_callback(lambda_init)

        k = 0

        wk = self._project(rk)
        zk = self._precondition(wk)
        yk = self._project(zk)


        V[0] = yk


        q0= F_callback(yk)
        norm_q0= np.linalg.norm(q0)
        Q[0] = q0

        for k in range(self._config_dict['max_iter']):
            info_dict[k] = {}
            alpha = np.vdot(rk, Q[k]) / np.vdot(Q[k],Q[k])
            lambda_sol = lambda_sol + V[k] * alpha

            rk =rk - alpha * Q[k]

            wk = self._project(rk)
            zk = self._precondition(wk)
            yk = self._project(zk)

            if self._config_dict['save_history']:
                lambda_hist = np.append(lambda_hist, lambda_sol)
                residual_hist = np.append(residual_hist, np.linalg.norm(rk))
                eta_subspace = np.append(eta_subspace, Proj_V[k])

                norm_wk = np.linalg.norm(rk)
                if norm_wk <= self._config_dict['tolerance']:
                    logger.info('Orthomin has converged after %i' % (k + 1))
                    break

            else:
                norm_wk = np.linalg.norm(rk)
                if norm_wk <= self._config_dict['tolerance']:
                    logger.info('Orthomin has converged after %i' % (k + 1))
                    break

            eta = F_callback(yk)
            v_k = yk


            for i in range(k+1):
                beta  = np.vdot(eta, Q[i])/ np.vdot(Q[i],Q[i])
                v_k = v_k -  beta*V[i]
                eta = eta -  beta*Q[i]


            V[k+1] = v_k
            Q[k+1] = eta


        if (k > 0) and k == (self._config_dict['max_iter'] - 1) and norm_wk > self._config_dict['tolerance']:
            logger.warning('Maximum iteration was reached, MAX_INT = %i, without converging!' % (k + 1))
            logger.warning('Projected norm = %2.5e , where the Orthomin tolerance is set to %2.5e' % (
            norm_wk, self._config_dict['tolerance']))

        elapsed_time = time.time() - global_start_time
        logger.info('#' * 60)
        logger.info('{"Total_elaspsed_time_PCPG" : %2.2f} # Elapsed time [s]' % (elapsed_time))
        logger.info('Number of ORTHOMIN  Iterations = %i !' % (k + 1))
        avg_iteration_time = elapsed_time / (k + 1)
        logger.info('{"avg_iteration_time_GMRES" : %2.4f} # Elapsed time [s]' % (avg_iteration_time))
        logger.info('#' * 60)

        info_dict['avg_iteration_time'] = elapsed_time / (k + 1)
        info_dict['Total_elaspsed_time_GMRES'] = elapsed_time
        info_dict['Iterations'] = k + 1
        info_dict['lambda_hist'] = lambda_hist
        info_dict['residual_hist'] = residual_hist
        info_dict['residual'] = norm_wk
        info_dict['Projected_search_direction'] = Q

        return lambda_sol, info_dict






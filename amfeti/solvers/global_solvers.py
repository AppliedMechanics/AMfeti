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
from matplotlib import pyplot as plt

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
        self._config_dict = {'tolerance': 1e-8,
                             'max_iter': None,
                             'projection': None,
                             'precondition': None,
                             'multiprecondition': None,
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
        InitialResidual = rk
        k = 0
        Store =[]
        Alpha_denominator = []
        Alpha_numerator = []
        Store_residual=[]
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
                    Orthoparameter= np.vdot(qki, yk1) / np.vdot(qki, yki)
                    yk -=  Orthoparameter* yki
            elif k > 0:
                yk -= np.vdot(qk_1, yk) / np.vdot(qk_1, yk_1) * yk_1

            vn1 = np.vdot(wk, zk)

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

                if k > 0:
                    Store.append(np.dot(Y[0].T, F_callback(Y[k])))

            else:
                yk_1 = copy(yk)
                qk_1 = copy(Fyk)


            aux2 = np.linalg.norm(np.vdot(yk, Fyk))
            alpha_k = (vn1 / aux2)

            if k>0:
                Alpha_denominator.append(aux2)
                Alpha_numerator.append(vn1)

            lambda_sol = lambda_sol +  Y[k]*alpha_k

            rk = rk - alpha_k * Q[k]


            if k==0:
                r0 = rk - alpha_k * Q[k]

            if k>0:
                Store_residual.append( np.dot(zk.T,r0))
            ComputeError = np.linalg.norm(rk - (InitialResidual- F_callback(lambda_sol)))


            print('The difference between the residuals is ',ComputeError)

            BasisError = np.linalg.norm(Fyk- F_callback(yk))

            if BasisError >1e-6:
                print('The basis  is inconsistent ', BasisError)

            print('The CG residual is ',np.linalg.norm(rk))

            solution = residual_callback(lambda_sol)
            print('The actual residual is ',np.linalg.norm(solution))

        lambda_sol = self._project(lambda_sol) + lambda_init

        if (k > 0) and k == (self._config_dict['max_iter'] - 1) and norm_wk > self._config_dict['tolerance']:
            logger.warning('Maximum iteration was reached, MAX_INT = %i, without converging!' % (k + 1))
            logger.warning('Projected norm = %2.5e , where the PCPG tolerance is set to %2.5e' % (norm_wk, self._config_dict['tolerance']))

        plt.figure()
        # plt.plot(np.arange(k), np.abs(Store))
        plt.plot(np.arange(k), np.abs(Store_residual))
        # plt.plot(np.arange(k), np.abs(Alpha_numerator))
        # plt.plot(np.arange(k), np.abs(Alpha_denominator))
        plt.yscale('log')
        # plt.legend(['Orthogonal_p'], ['Alpha_n'], ['Alpha_d'])
        plt.show()
        plt.figure()
        plt.plot(np.arange(k), residual_hist[0:k])
        plt.yscale('log')
        # plt.legend(['Orthogonal_p'], ['Alpha_n'], ['Alpha_d'])
        plt.show()
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

    def _multiprecondition(self, v):
        if self._config_dict['precondition'] is not None:
            precondition = self._config_dict['multiprecondition']
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
        info_dict['GMRES_iterations'] = k + 1
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
                             'full_reorthogonalization': True,
                             'Recycling': True,
                             'relax_tolerance': 0.001,
        }

    def solve(self, F_callback, F_callback_single_precond, residual_callback, lambda_init,SD):
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
        rk = residual_callback(lambda_sol)


        if self._config_dict['Recycling'] is True:

            """ Computes  the updated orthonormalized SD, residual if recycling flag is on"""
            ProjectedSearchDirections, SearchDirections, rk, lambda_sol  = self.RecycleVectors(SD,rk, lambda_sol, F_callback_single_precond,residual_callback)




        NewSearchDirections = self._multiprecondition(rk)
        NewProjectedSearchDirections = self.F_callback_loop(NewSearchDirections, F_callback_single_precond)


        if self._config_dict['Recycling'] is True and SD.size != 0:
            "Append the new search directions to the recycled ones, update the solution and "
            Q_dict[0], W_dict[0],rk, lambda_sol =self.BlockGramSchmidt(ProjectedSearchDirections, SearchDirections, NewProjectedSearchDirections,
                                       NewSearchDirections, rk, lambda_sol, F_callback_single_precond)

        else:
            "Compute an orthonormal basis"
            NewProjectedSearchDirections, NewSearchDirections = self.OrthogonalizeVectors(NewProjectedSearchDirections,NewSearchDirections)

            NewProjectedSearchDirections, NewSearchDirections = self.DeleteZeroSearchdirection(NewProjectedSearchDirections, NewSearchDirections)

            Q_dict[0], W_dict[0] = self.OrthonormalStep(NewProjectedSearchDirections, NewSearchDirections)

        self._Basis_Check(Q_dict[0], W_dict[0], F_callback_single_precond)


        print('starting loop')
        Counter = 0
        for k in range(self._config_dict['max_iter']):
            info_dict[k] = {}

            Minimizationstep = np.dot(np.conjugate(Q_dict[k].T),rk)
            AlphaParameter =Minimizationstep

            FilteringParameter = np.dot(W_dict[k],AlphaParameter)
            lambda_sol = lambda_sol + FilteringParameter
            rk = rk - np.dot(Q_dict[k], (AlphaParameter))


            # if self._config_dict['Recycling'] is True and  k==0 :
            #
            #     "This simply appends the current search directions with the old ones to ensure an orthonormal basis"
            #     Q_dict[0], W_dict[0] = self.BlockGramSchmidt(ProjectedSearchDirections, SearchDirections,
            #                                                                               NewProjectedSearchDirections,NewSearchDirections,
            #                                                                               F_callback_single_precond)
            #

            ActualResidual = np.linalg.norm(residual_callback(lambda_sol))

            print('Residue        = ', np.linalg.norm(rk))
            print('Actual Residue = ', ActualResidual)
            
            wk = self._project(rk)
            zk = self._multiprecondition(wk)



            if np.linalg.norm(rk) > self._config_dict['tolerance']*self._config_dict['relax_tolerance'] :

                zk = self._multiprecondition(wk)
                SearchDirections = self._project(zk)
                ProjectedSearchDirections = self.F_callback_loop(SearchDirections, F_callback_single_precond)
            else:
                zk = self._precondition(wk)
                SearchDirections = self._project(zk)
                SearchDirections.shape = (lambda_sol.size,1)
                ProjectedSearchDirections = F_callback(SearchDirections)


            W_dict[k+1] = SearchDirections
            Q_dict[k+1] = ProjectedSearchDirections
            if self._config_dict['full_reorthogonalization']:
                for i in range(k+1):
                    Numerator = np.dot( np.conjugate(Q_dict[i].T), Q_dict[k + 1])
                    beta_ik   = Numerator
                    W_dict[k+1] = W_dict[k+1] - np.dot(W_dict[i],(beta_ik))
                    Q_dict[k+1] = Q_dict[k+1] - np.dot(Q_dict[i],(beta_ik))



            if  Q_dict[k+1].shape[1] == 1:

                print('Only single preconditioning now')


            else:
                Qorthoginalized, SearchDirectionsorthoginalized = self.OrthogonalizeVectors(Q_dict[k+1], W_dict[k+1])

                Qupdated, SearchDirectionsUpdated = self.DeleteZeroSearchdirection(Qorthoginalized, SearchDirectionsorthoginalized)

                print('The number of vectors stored is:', Qupdated.shape[1])
                print('the rank of the matrix of projected search direction after filtering :', np.linalg.matrix_rank(Qupdated))

                self._Basis_Check(Q_dict[k + 1], W_dict[k + 1], F_callback_single_precond)
                Counter += W_dict[k + 1].shape[1]


            Q_dict[k + 1], W_dict[k + 1] = self.OrthonormalStep(Qupdated, SearchDirectionsUpdated)



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
        info_dict['Iterations'] = Counter
        print('The total SD stored are :',Counter )
        info_dict['lambda_hist'] = lambda_hist
        info_dict['residual_hist'] = residual_hist
        info_dict['residual'] = norm_wk

        if self._config_dict['Recycling'] is True:

            MaxVectors = 20
            RecyclableSearchDirections = np.array([])
            iCounter = 0
            for iNrVectors in range(k):

                if len(RecyclableSearchDirections) == 0 :
                    RecyclableSearchDirections = W_dict[iNrVectors]

                else:
                    RecyclableSearchDirections = np.append(RecyclableSearchDirections,W_dict[iNrVectors],axis=1)




            info_dict['P_SearchDirections'] = RecyclableSearchDirections

        else:
            info_dict['P_SearchDirections'] = []
            info_dict['P_ProjSearchDirections'] = []

        return lambda_sol, info_dict

    def _Basis_Check(self,ProjVectors, SearchVectors,F_callback_single_precond):

        if (np.linalg.norm(ProjVectors - self.F_callback_loop(SearchVectors, F_callback_single_precond))) >= 1e-6:
            print("-----the basis is inconsistent, fix the bug in the code")

        else:
            print('Good to proceed')

    def F_callback_loop(self, SD, F_callback_single_precond):
        ProjecctedSD = np.zeros((SD.shape[0], SD.shape[1]), dtype=complex)
        for iCounter in range(SD.shape[1]):
            ProjecctedSD[:, iCounter] = F_callback_single_precond(SD[:, iCounter])
        return ProjecctedSD

    def OrthonormalStep(self, NDArrayProjVectors, NDArraySearchVectors):
        SizeOfInterfaceProblem= self._config_dict['max_iter']
        if NDArraySearchVectors.size == SizeOfInterfaceProblem :
            print('Only 1 vector to orthonormalize')
            Norm = np.linalg.norm(NDArrayProjVectors)
            NDArraySearchVectors= NDArraySearchVectors/Norm
            NDArrayProjVectors = NDArrayProjVectors/Norm
            NDArrayProjVectors.shape = (SizeOfInterfaceProblem,1)
            NDArraySearchVectors.shape = (SizeOfInterfaceProblem,1)
        else:
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

    def ComputeInitialResidual(self, NDArrayProjVectors, NDArraySearchVectors,lambda_init,initial_residual,residual_callback):

        Difference_ini = np.linalg.norm(initial_residual - residual_callback(lambda_init))

        if Difference_ini >= 1e-6:
            print('The intial residual in recycling is wrong')


        InitialResidual = initial_residual
        for iCounter in range(NDArrayProjVectors.shape[1]):
            Minimizationstep = np.dot(np.conjugate(NDArrayProjVectors[:,iCounter].T), InitialResidual)
            lambda_init = lambda_init + np.dot(NDArraySearchVectors[:,iCounter], Minimizationstep)
            InitialResidual = InitialResidual - np.dot(NDArrayProjVectors[:,iCounter], (Minimizationstep))


        Difference =     np.linalg.norm(InitialResidual - residual_callback(lambda_init))

        if Difference >= 1e-6:
            print('The intial residual in recycling is wrong')

        return  InitialResidual, lambda_init

    def RecycleVectors(self,SD,rk,lambda_sol, F_callback_single_precond, residual_callback):

        if SD.size == 0:
            print('First sweep no recycling')
            # SearchDirections = rk
            # Qupdated = F_callback(rk)
            # ProjSearchDirections, SearchDirections  = self.OrthonormalStep(Qupdated, SearchDirections)
            ProjSearchDirections=[]
            SearchDirections = []

        else:

            """     Project the SD on the new interface operator"""
            ProjSearchDirections = self.F_callback_loop(SD, F_callback_single_precond)

            """ Orthonormalize them using L-D-L*"""
            ProjSearchDirections, SearchDirections = self.OrthogonalizeVectors(ProjSearchDirections,
                                                                                 SD)
            """ Remove linearly dependent search directions"""
            ProjSearchDirections, SearchDirections = self.DeleteZeroSearchdirection(ProjSearchDirections, SearchDirections)

            """ Orthonormalize the basis"""
            ProjSearchDirections, SearchDirections = self.OrthonormalStep(ProjSearchDirections,
                                                                               SearchDirections)

            """ Compute the recycled residual and updated solution from these search directions"""
            rk, lambda_sol = self.ComputeInitialResidual(ProjSearchDirections, SearchDirections, lambda_sol, rk,
                                                    residual_callback)

        return ProjSearchDirections, SearchDirections, rk, lambda_sol

    def BlockGramSchmidt(self, NDArrayProjVectors, NDArraySearchVectors, CurrentProjVectors, CurrentSearchVectors,rk, lambda_sol, F_callback_single_precond):

        Numerator = np.dot(np.conjugate(NDArrayProjVectors.T), CurrentProjVectors)

        CurrentSearchVectors =CurrentSearchVectors - np.dot(NDArraySearchVectors, Numerator)
        CurrentProjVectors = CurrentProjVectors - np.dot(NDArrayProjVectors, Numerator)

        ProjectedSearchDirections, SearchDirections = self.OrthogonalizeVectors(CurrentProjVectors,CurrentSearchVectors)
        ProjectedSearchDirections, SearchDirections = self.DeleteZeroSearchdirection(ProjectedSearchDirections, SearchDirections)

        ProjectedSearchDirections, SearchDirections = self.OrthonormalStep(ProjectedSearchDirections,
                                                                                     SearchDirections)

        Minimizationstep = np.dot(np.conjugate(ProjectedSearchDirections.T), rk)

        FilteringParameter = np.dot(SearchDirections, Minimizationstep)
        lambda_sol = lambda_sol + FilteringParameter
        rk = rk - np.dot(ProjectedSearchDirections, Minimizationstep)


        # vk_stack = SearchDirections[np.newaxis].T
        # SearchDirections = np.append(SearchDirections,vk_stack,axis=1)
        # vk_stack = ProjectedSearchDirections[np.newaxis].T
        # ProjectedSearchDirections = np.append(ProjectedSearchDirections,vk_stack,axis=1)

        if (np.linalg.norm(NDArrayProjVectors - self.F_callback_loop(NDArraySearchVectors, F_callback_single_precond))) >= 1e-6:
            print("-----the basis is inconsistent during the orthogonalization of recycling process-----")

        return  ProjectedSearchDirections, SearchDirections, rk, lambda_sol
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

        logger.info('Setting GMRES tolerance = %4.2e' % self._config_dict['tolerance'])
        logger.info('Setting GMRES max number of iterations = %i' % self._config_dict['max_iter'])

        # initialize variables
        info_dict = {}
        global_start_time = time.time()
        residual_hist = np.array([])
        lambda_hist = np.array([])

        V = dict()
        Proj_V = dict()


        lambda_sol = np.zeros_like(lambda_init)
        rk = residual_callback(lambda_init)
        residual_0 = copy(rk)
        """ copy with numpy unlinks the two arrays 
        """
        lambda_0 = copy(lambda_init)
        k = 0
        norm_vk = np.linalg.norm(rk)

        wk = self._project(rk)
        zk = self._precondition(wk)
        yk = self._project(zk)


        V[0] = yk

        # vk_stack = V[0][np.newaxis].T
        # V_stack = csr_matrix(vk_stack)

        proj_v= F_callback(yk)
        norm_proj_vk= np.linalg.norm(proj_v)
        Proj_V[0] = proj_v
        projected_subspace = []

        for k in range(self._config_dict['max_iter']):
            info_dict[k] = {}
            alpha = np.vdot(rk, Proj_V[k]) / np.vdot(Proj_V[k],Proj_V[k])
            lambda_sol = lambda_sol + V[k] * alpha

            rk =rk - alpha * Proj_V[k]

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

            eta = F_callback(yk)
            v_k = yk


            for i in range(k+1):
                beta  = np.vdot(eta, Proj_V[i])/ np.vdot(Proj_V[i],Proj_V[i])
                v_k = v_k -  beta*V[i]
                eta = eta -  beta*Proj_V[i]


            V[k+1] = v_k  #/np.linalg.norm(v_k)
            Proj_V[k+1] = eta #/ np.linalg.norm(eta)


        if (k > 0) and k == (self._config_dict['max_iter'] - 1) and norm_wk > self._config_dict['tolerance']:
            logger.warning('Maximum iteration was reached, MAX_INT = %i, without converging!' % (k + 1))
            logger.warning('Projected norm = %2.5e , where the Orthomin tolerance is set to %2.5e' % (
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
        info_dict['Projected_search_direction'] = Proj_V

        return lambda_sol, info_dict



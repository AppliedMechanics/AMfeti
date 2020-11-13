#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
import numpy as np
from scipy.sparse import csc_matrix, issparse, lil_matrix, linalg as sla
from scipy.linalg import cho_solve
from .tools import splusps, cholsps

__all__ = ['Pseudoinverse',
           'Matrix',
           'Vector']


class Pseudoinverse:
    ''' This class intend to solve singular systems
    build the null space of matrix operator and also
    build the inverse matrix operator

    Ku = f

    where K is singular, then the general solution is

    u = K_pinvf + alpha*R

    argument
        K : np.array
            matrix to be inverted
        tol : float
            float tolerance for building the null space

    return:
        K_pinv : object
        object containg the null space and the inverse operator
    '''

    def __init__(self, method='splusps', tolerance=1.0E-8):

        self.list_of_solvers = ['cholsps', 'splusps', 'svd']
        if method not in self.list_of_solvers:
            raise ('Selection method not avalible, please selection one in the following list :' % (
                self.list_of_solvers))

        self.solver_opt = method
        self.pinv = None
        self.null_space = np.array([])
        self.free_index = []
        self.tolerance = 1.0E-8
        self.matrix = None

    def set_tolerance(self, tol):
        ''' setting P_inverse tolerance

        arguments
            tol : tol
                new pseudo-inverse tolerance
        return
            None
        '''
        self.tolerance = tol

        return

    def set_solver_opt(self, solver_opt):
        ''' This methods set the P_inverse method

            argument
                solver_opt : str
                    string with solver opt

            returns
        '''
        if solver_opt in self.list_of_solvers:
            self.solver_opt = solver_opt
        else:
            raise ('Error! Select solver is not implemented. ' + \
                   '\n Please check list_of_solvers variable.')

    def compute(self, K, tol=None, solver_opt=None):
        ''' This method computes the kernel and inverse operator
        '''

        # store matrix to future use
        self.matrix = K

        if solver_opt is None:
            solver_opt = self.solver_opt

        if tol is None:
            tol = self.tolerance

        if solver_opt == 'splusps':
            lu, idf, R = splusps(K, tol=tol)

            # add constraint in K matrix and applu SuperLu again
            if len(idf):
                Pr = lambda x: x - R.dot(R.T.dot(x))
            else:
                Pr = lambda x: x

            K_pinv = lambda x: Pr(lu.solve(x))

        elif solver_opt == 'cholsps':
            U, idf, R = cholsps(K, tol=tol)
            U[idf, :] = 0.0
            U[:, idf] = 0.0
            U[idf, idf] = 1.0
            K_pinv = lambda f: cho_solve((U, False), f)

        elif solver_opt == 'svd':
            K_inv, R = pinv_and_null_space_svd(K, tol=tol)
            K_pinv = np.array(K_inv).dot
            idf = []

        else:
            raise ('Solver %s not implement. Check list_of_solvers.')

        self.pinv = K_pinv
        self.free_index = idf
        if R is not None:
            self.null_space = R
        else:
            self.null_space = np.array([])

        return self

    def apply(self, f, alpha=np.array([]), check=False):
        ''' function to apply K_pinv
        and calculate a solution based on alpha
        by the default alpha is set to the zero vector

        argument
            f : np.array
                right hand side of the equation
            alpha : np.array
                combination of the kernel of K alpha*R
            check : boolean
                check if f is orthogonal to the null space
        '''
        K_pinv = self.pinv
        idf = self.free_index

        # f must be orthogonal to the null space R.T*f = 0
        # P = sparse.eye(f.shape[0]).tolil()
        # if idf:
        #    P[idf,idf] = 0.0
        #     f = P.dot(f)
        # if self.solver_opt == 'cholsps':
        #    f[idf] = 0.0

        if check:
            if not self.has_solution(f):
                raise ('System has no solution because right hand side is \
                       \n not orthogonal to the null space of the matrix operator.')

        u_hat = K_pinv(f)

        if alpha.size > 0:
            u_hat += self.calc_kernel_correction(alpha)

        return u_hat

    def calc_kernel_correction(self, alpha):
        ''' apply kernel correction to
        calculate another particular solution
        '''
        R = self.null_space
        u_corr = R.dot(alpha)
        return u_corr

    def check_null_space(self, tolerance=1.0E-3):
        ''' check null calculated null space is a null space
        of self.matrix considering two aspects,
        1. K*v = 0    where v is a vector in the R = [v1, v2  ...,vm]
            check ||K*v||/||v|| us < tolerance
        2. R is a full row rank matrix

        arguments:
            tolerance : float
                tolerance for the norm of the vector v in R
               by the the K1 matrix, which represents a tolerance for
               checking if v in R is really a kernel vector of K
        return
            bool : boolean

            True if all vector in null space are in the tolerance
        '''
        bool = False
        K = self.matrix
        R = self.null_space
        n, m = R.shape
        null_space_size = 0
        for v in R.T:
            if is_null_space(K, v, tol=tolerance):
                null_space_size += 1

        R_rank = np.linalg.matrix_rank(R.T)
        if m == null_space_size and R_rank == m:
            bool = True

        return bool

    def has_solution(self, f):
        ''' check if f is orthogonal to the null space

        arguments
            f : np.array
                right hand side of Ku=f
        return
            boolean

        '''
        R = self.null_space
        v = R.T.dot(f)
        ratio = np.linalg.norm(v) / np.linalg.norm(f)

        bool = False
        if ratio < self.tolerance:
            bool = True
        return bool


class Matrix:
    '''  Basic matrix class
    '''
    counter = 0

    def __init__(self, K, key_dict={}, name=None, pseudoinverse_kargs={'method': 'svd', 'tolerance': 1.0E-8}):
        '''
        pseudoinverse_key_args=(method='splusps',tolerance=1.0E-8)
        '''
        Matrix.counter += 1
        self.id = Matrix.counter
        if isinstance(K, np.matrix):
            K = np.array(K)
        self.data = K
        self.key_dict = key_dict
        self.type = None
        self.issingular = None
        self.prefix = 'K'
        self.eliminated_id = set()
        self.psudeoinverve = Pseudoinverse(**pseudoinverse_kargs)
        self.inverse_computed = False
        if name is None:
            self.update_name()
        else:
            self.name = name

    def set_psudeoinverve_alg(self, name):
        ''' Parameters
                name : str
                    name of the pseudoinverse method
        '''
        pseudoinverse_key_args = {'method': name}
        self.psudeoinverve = Pseudoinverse(**pseudoinverse_key_args)
        self.issingular = None

    def update_name(self):
        self.name = self.prefix + str(self.id)

    def compute_psudeoinverse(self):
        self.psudeoinverve.compute(self.data)
        self.inverse_computed = True

    @property
    def shape(self):
        return self.data.shape

    @property
    def trace(self):
        return self.data.diagonal().sum()

    @property
    def det(self):
        return np.linalg.det(self.data)

    @property
    def eigenvalues(self):
        w, v = np.linalg.eig(self.data)
        return np.sort(w)[::-1]

    def dot(self, x):
        return self.data.dot(x)

    def inverse(self):
        pass

    @property
    def kernel(self):
        ''' compute the kernel of the matrix
        based on the pseudoinverse algorithm
        '''
        if not self.inverse_computed:
            self.psudeoinverve.compute(self.data)
            self.inverse_computed = True

        return self.psudeoinverve.null_space

    def apply_inverse(self, b):

        if not self.inverse_computed:
            self.psudeoinverve.compute(self.data)
            self.inverse_computed = True

        return self.psudeoinverve.pinv(b)

    def get_block(self, row_key, column_key):
        pass

    def eliminate_by_identity(self, dof_ids, multiplier=1.0):
        ''' This function eliminates matrix rows and columns
        by replacing rows and columns by identity matrix

        [[k11, k12, k13                 [[k11, 0, k13
          k21, k22, k23]    ->            0, 1,    0]
          k21, k22, k23]                  k21, 0, k23]]

        Parameters:
            dof_ids : OrderedSet or a Str
                if OrderedSet a set of dofs to be eliminated by identity
                if string a key of self.key_dict which maps to the set of dof
                to be eliminated

        return eliminated K matrix

        '''

        if isinstance(dof_ids, str):
            dofs = list(self.key_dict[dof_ids])
        elif isinstance(dof_ids, int):
            dofs = list(self.key_dict[dof_ids])
        else:
            dofs = list(dof_ids)

        if list(dofs)[0] is None:
            return

        if issparse(self.data):
            self.data = self.data.tolil()
        dirichlet_stiffness = multiplier * self.trace / self.shape[0]
        self.data[dofs, :] = 0.0
        self.data[:, dofs] = 0.0
        self.data[dofs, dofs] = dirichlet_stiffness
        self.eliminated_id.update(dofs)

        if issparse(self.data):
            return self.data.tocsr()
        else:
            return self.data

    def save_to_file(self, filename=None):
        if filename is None:
            filename = self.name + '.pkl'
            print('Filename is = %s' % filename)

        save_object(self, filename)


class SparseMatrix(Matrix):
    '''  Basic matrix class
    '''

    def __init__(self, K, key_dict={}):
        super().__init__(K, key_dict={})


class Vector:
    counter = 0

    def __init__(self, v, key_dict={}, name=None):
        self.id = Vector.counter
        self.data = np.array(v)
        self.key_dict = key_dict
        self.prefix = 'v'

        if name is None:
            self.update_name()
        else:
            self.name = name

    def update_name(self):
        self.name = self.prefix + str(self.id)

    def replace_elements(self, dof_ids, value):
        '''
         Parameters:
            dof_ids : OrderedSet or a Str
                if OrderedSet a set of dofs will be replace by the value
                if string a key of self.key_dict which maps to the set of dof
                will be replace by the value
            value : float
                float to replace the values in the initial array

        return a new vnumpy.array
        '''
        if isinstance(dof_ids, str):
            dofs = list(self.key_dict[dof_ids])
        else:
            dofs = list(dof_ids)

        self.data[dofs] = value
        return self.data


def pinv_and_null_space_svd(K, tol=1.0E-8):
    ''' calc pseudo inverve and
    null space using SVD technique
    '''

    if issparse(K):
        K = K.todense()

    n, n = K.shape
    V, val, U = np.linalg.svd(K)

    total_var = np.sum(val)

    norm_eigval = val / val[0]
    idx = [i for i, val in enumerate(norm_eigval) if val > tol]
    val = val[idx]

    invval = 1.0 / val[idx]

    subV = V[:, idx]

    Kinv = np.matmul(subV, np.matmul(np.diag(invval), subV.T))

    last_idx = idx[-1]
    if n > len(idx):
        R = np.array(V[:, last_idx + 1:])
    else:
        R = np.array([])

    return Kinv, R
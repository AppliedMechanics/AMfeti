from amfeti.config_base import ConfigBase
import logging
import numpy as np
import time
from amfeti.linalg.datatypes import Matrix,Pseudoinverse
from scipy import linalg as spp
from copy import copy
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gcrotmk
from scipy.sparse import linalg as linearoperator
from amfeti.solvers import M_ORTHOMIN, ORTHOMINsolver, GMRESsolver

__all__ = [ 'Total_Recycling'

]

class GlobalRecyclingBase(ConfigBase):
    def __init__(self):
        super().__init__()

    def solve(self, *args):
        return None


class Total_Recycling(GlobalSolverBase):
    def __init__(self):
        """
        Parameters
        ----------
        None
        """
        super().__init__()
        self._config_dict = { 'Recycling':False,
            'PreviouslyStoredVectors':True}

    def solve(self, F_callback, residual_callback):


        if self._config_dict['Recycling'] is False:



        if self._config_dict['PreviouslyStoredVectors'] is None:
            InitializeVectors = []
        else:
            InitializeVectors = self._config_dict['PreviouslyStoredVectors']

        F_projection = linearoperator.aslinearoperator(F_callback)
        rk = residual_callback(lambda_init)


        gcrotmk(F_projection,x0=lambda_init,tol='tolerance', maxiter=len(lambda_init), m= len(lambda_init), k = np.rint(0.2*len(lambda_init)),CU= InitializeVectors,truncate='oldest')

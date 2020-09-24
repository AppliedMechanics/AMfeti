
import numpy as np
import collections
import os
import matplotlib.pyplot as plt
import shutil
import time, logging, json, subprocess
import sys
from datetime import datetime
curdir = os.getcwd()


if __name__ == '__main__':
    help_doc = ''' 
            Python script for scalability test in AMfeti (https://github.com/AppliedMechanics/AMfeti)
            Domain:
                                                     W
                                           __ __ __ __ __ __ __
                                          |                    |
                                          |                    |
                                        H |                    |
                                          |                    |
                                          |__ __ __ __ __ __ __|
            Options:
            W        : Width in meters for the 2D plane-stress body. Default = 60
            H        : Height in meters for the 2D plane-stress body. Default = 60
            
            divX     : Number of global divisions in the X direction. Default = 24
            divY     : Number of global divisions in the Y direction. Default = 24
            
            domainX  : list of domains in the X direciton. Default = [1,2,3]
            domainY  : list of domains in the Y direciton. Default = [1,2,3]
            
            method   : pseudoinverse of subdomain matrices. Default = splusps (others: cholsps, svd)
            algorithm: serial or parallel exec.             Default = ParallelFETIsolver (others: SerialFETIsolver)
            tol      : tolerance of solver error norm.      Default = 1.0E-5
            precond  : Preconditioner type.                 Default - Identity (options: Lumped, Dirichlet, LumpedDirichlet, SuperLumped)
            
            strong        : scalability test, Boolear True for strong and False for weak. Default = True
            loglevel      : INFO, DEBUG, ERROR, WARNING, CRITICAL. Default = WARNING
            launcher_only : Boolean True to create scripts without launching solver. Default = False
            delete_files  : Boolean True to delete *.pkl files after mpirun. Default = True
            salomon       : {} dict with salomon parameters e.g. {'queue':'qexp','ncpus' : 24, 'default_time':30, 'effectivity': 0.7}.  Default = {}
                            'default_time' is given in minutes, an estimation of required HPC time will be computed based on it.
            
            example of command call:
            >> python  create_test_case.py W=60 H=60 domainX=[2,3] domainX=[1,1] 
            '''

    default_dict = {'W'             : 60.0,
                    'H'             : 60.0,
                    'divY'          : 24,
                    'divX'          : 24,
                    'domainX'       : [1, 2, 3],
                    'domainY'       : [1, 2, 3],
                    'method'        : 'splusps',
                    'FETI_algorithm': 'ParallelFETIsolver',
                    'tol'           : 1.0E-5,
                    'precond'       : None,
                    'strong'        : True,
                    'loglevel'      : 'WARNING',
                    'launcher_only' : False,
                    'delete_files'  : True,
                    'salomon'       : {},
                    'BC_type'       : 'RX'}
    # add default dict to local variables
    locals().update(default_dict)

    for domain_x, domain_y in zip(domainX, domainY):
        mpi_size = domain_x * domain_y
        # subdomain dimensions
        subdomain_w = W / domain_x
        subdomain_h = H / domain_y
        # subdomain mesh
        subdomain_divx = divX / domain_x
        subdomain_divy = divY / domain_y

        K_dict, M_dict, B_dict, f_dict, K_effective_dict, f_effective_dict  = create_subdomain_matrices(n_domains_x=domain_x, n_domains_y=domain_y,
                                                                                                        n_ele_x=subdomain_divx, n_ele_y=subdomain_divy,
                                                                                                        length=W, height=H, split_system=False)

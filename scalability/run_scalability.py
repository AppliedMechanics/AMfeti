
import numpy as np
import collections
import os
import matplotlib.pyplot as plt
import shutil
import time, logging, json, subprocess
import sys
from datetime import datetime
from amfeti.scalabilty.case_generator import *
curdir = os.getcwd()
header = '#' * 50

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
            solver   : dual interface algorithm             Default = GNRES (others: PCPG, future: MPPCPG, MPGNRES)
            tol      : tolerance of solver error norm.      Default = 1.0E-5
            precond  : Preconditioner type.                 Default - Identity (options: Lumped, Dirichlet, LumpedDirichlet, SuperLumped)
            
            strong        : scalability test, Boolean True for strong and False for weak. Default = True
            loglevel      : logging.(INFO/ DEBUG/ ERROR/ WARNING/ CRITICAL). Default = logging.WARNING
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
                    'solver'        : 'GMRES',
                    'tol'           : 1.0E-5,
                    'precond'       : None,
                    'strong'        : True,
                    'loglevel'      : logging.INFO,
                    'launcher_only' : False,
                    'delete_files'  : True,
                    'salomon'       : {},
                    'BC_type'       : 'RX'}
    # add default dict to local variables
    locals().update(default_dict)

    ###############################   create folder to store information   ############################################
    date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    scalability_folder = os.path.join(curdir, date_str)
    os.mkdir(scalability_folder)
    # change to scalability local folder
    os.chdir(scalability_folder)
    ##############################################  mpi size ##########################################################
    if len(domainY) != len(domainX):
        logging.warning('DomainY list with different length of DomainX. Setting new DomainY')
        domainY = [domainY[0]] * len(domainX)
        logging.warning(('new DomainY = ', domainY))

    max_mpi_size = (max(domainX) * max(domainY))
    min_mpi_size = (min(domainX) * min(domainY))
    ##############################################  logging  ##########################################################
    logging.basicConfig(level=loglevel, filename='scalability_master.log', filemode='w', format='%(levelname)s - %(message)s')
    logging.info(header)
    logging.info('#############    SCALABILITY TEST  ###############')
    logging.info(header)
    logging.info(datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
    if strong:
        logging.info('Performing STRONG parallel scalability.')
    else:
        logging.info('Performing WEAK parallel scalability.')
    if precond is None:
        logging.info('Preconditioner type  = %s' % 'Identity')
    else:
        logging.info('Preconditioner type  = %s' % precond)
    logging.info('Solver method  = %s' %solver )
    logging.info('Solver tolerance  = %2.2e' % tol)
    logging.info('Pseudoinverse method  = %s' % method)
    logging.info('Number of global divisions in X directions = %i' % divX)
    logging.info('Number of global divisions in Y directions = %i' % divY)
    logging.info('Maximum MPI size  = %i' % max_mpi_size)
    logging.info('Minimum MPI size  = %i' % min_mpi_size)
    ###################################################################################################################
    for domain_x, domain_y in zip(domainX, domainY):
        mpi_size = domain_x * domain_y
        # subdomain dimensions
        subdomain_w = W / domain_x
        subdomain_h = H / domain_y
        # subdomain mesh
        subdomain_divx = divX / domain_x
        subdomain_divy = divY / domain_y

        K_dict, M_dict, B_dict, f_dict, K_effective_dict, f_effective_dict  = create_subdomain_matrices(n_domains_x=domain_x, n_domains_y=domain_y,
                                                                                                        n_ele_X_global=divX, n_ele_Y_global=divY,
                                                                                                        length=W, height=H, split_system=False)
        logging.info(header)

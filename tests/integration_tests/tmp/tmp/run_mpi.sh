#!/bin/sh
"mpiexec"  -n 4 "python" "/home/aseibold/Coding/AMfeti/amfeti/parallelization_managers/mpi_local_processor.py"  "prefix=/home/aseibold/Coding/AMfeti/tests/integration_tests/tmp/tmp/mpi_rank_"   "ext=.pkl"   "solution=/home/aseibold/Coding/AMfeti/tests/integration_tests/tmp/tmp/mpi_rank_" >mpi.log
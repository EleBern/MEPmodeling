#!/bin/bash

#PBS -l walltime=5:00:00

# reserve 1 CPU, 1 GB disc space, 1 GB RAM, and 1 GB scratch space
#PBS -l select=1:ncpus=1:mem=5gb:scratch_local=1gb  

#PBS -N pygpc_test

# stop the script immediately if any command fails,
# so that errors are visible in the error file
set -e

# Set directories
export INPUTDIR=/storage/brno2/home/eb22aal/MEPmodeling
export OUTPUTDIR=/storage/brno2/home/eb22aal/MEPmodeling/Results/${PBS_JOBID}
mkdir $OUTPUTDIR

# load the python module
module add python/3.9.12-gcc-10.2.1-rg2lpmk
export PYTHONUSERBASE=/storage/brno2/home/eb22aal/pygpc
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH

# copy the files to the scratch directory
cp -r ${INPUTDIR} $SCRATCHDIR

# go to the scratchdir 
cd ${SCRATCHDIR}

python main.py

# copy the outputs back to where qsub was called
cp -r pygpc ${OUTPUTDIR}/

# apply a scratch automatic cleanup utility
rm -r $SCRATCHDIR
#!/bin/bash

#PBS -l walltime=5:00:00

#PBS -l select=1:ncpus=1:mem=15gb:scratch_local=2gb  

#PBS -N pygpc_test

# stop the script immediately if any command fails,
# so that errors are visible in the error file
set -e

# Set directories
export INPUTDIR=/storage/brno2/home/eb22aal/MEP_pygpc/MEPmodeling
export OUTPUTDIR=/storage/brno2/home/eb22aal/MEP_pygpc/Results/${PBS_JOBID%%.*}
mkdir -p $OUTPUTDIR

# load the python module
module add python/3.9.12-gcc-10.2.1-rg2lpmk
export PYTHONUSERBASE=/storage/brno2/home/eb22aal/pygpc_env
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH

# copy the files to the scratch directory
cp -r ${INPUTDIR} $SCRATCHDIR

# go to the scratchdir 
cd ${SCRATCHDIR}/MEPmodeling

python main.py

# copy the outputs back to where qsub was called
cp -r pygpc ${OUTPUTDIR}/

rm -r $SCRATCHDIR/*
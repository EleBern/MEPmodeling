#!/bin/bash

#PBS -l walltime=24:00:00

# reserve 1 CPU, 1 GB disc space, 1 GB RAM, and 2 GB scratch space
#PBS -l select=1:ncpus=1:mem=5gb:scratch_local=2gb  

#PBS -N S1

# stop the script immediately if any command fails,
# so that errors are visible in the error file
set -e

# Set directories
export INPUTDIR=/storage/brno2/home/eb22aal/MEP_pygpc/MEPmodeling
export OUTPUTDIR=/storage/brno2/home/eb22aal/Reduce_MUAPs/S1/${PBS_JOBID%%.*}
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

function round10 {
    echo $(( ((${1%.*}+9)/10)*10 ))
}

export n_MUAPs=$(round10 $PBS_ARRAY_INDEX)

python gen_muaps.py $n_MUAPs
python ga_MEPmodel_bio.py 1 1 $n_MUAPs

# copy the outputs back to where qsub was called
cp -r fitted_results/bio/* ${OUTPUTDIR}/

clean_scratch
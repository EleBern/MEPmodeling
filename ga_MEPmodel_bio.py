"""
Biological MEP model entry point.

Example usage:
    from ga_MEPmodel_bio import ga_MEPmodel_bio
    ga_MEPmodel_bio(subj=1, withRC=1, AMPAweight=None, reRun=0)

    subj        : subject number
    withRC      : include recurrent connections (default 1)
    AMPAweight  : fixed AMPA weight value, or None to fit freely
    reRun       : 0 – load fitted result and plot simulated MEP
                  1 – re-run model fitting (backs up previous fitted result)
"""

import os
import sys
import h5py
import numpy as np

from MEPmodel_bio import MEPmodel_bio
from config_model_bio import config_model_bio
from h5_helpers import load_h5_to_dict

# ==========================================================================
def ga_MEPmodel_bio(subj, withRC=1, AMPAweight=None, reRun=0):
    """
    Main entry point for biological MEP model fitting using a Genetic Algorithm.
    """
    root = os.getcwd()

    # ----- model setting -----
    ref = config_model_bio(subj, withRC, AMPAweight)

    # ----- derive h5 result path (replace any existing extension) -----
    resultname_h5 = os.path.splitext(ref['resultname'])[0] + '.h5'
    result_path   = os.path.join(root, resultname_h5)

    # ----- run GA or load fitted result -----
    if os.path.isfile(result_path) and not reRun:
        print(f'Use fitted result: \n{resultname_h5}')
        with h5py.File(result_path, 'r') as f:
            tmp = load_h5_to_dict(f)
        p_post = tmp['p_post'].flatten()

    # ----- show result -----
    plotOn = 0
    ref = MEPmodel_bio(p_post, ref, plotOn)
    
    return ref


# ==========================================================================
if __name__ == '__main__':
    subj       = int(sys.argv[1])   if len(sys.argv) > 1 else 1
    withRC     = int(sys.argv[2])   if len(sys.argv) > 2 else 1
    AMPAweight = float(sys.argv[3]) if len(sys.argv) > 3 else None
    reRun      = int(sys.argv[4])   if len(sys.argv) > 4 else 0
    ga_MEPmodel_bio(subj, withRC, AMPAweight, reRun)
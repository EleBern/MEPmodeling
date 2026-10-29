"""
pygpc sensitivity analysis for the biological MEP model.

The spinal network is NOT re-simulated here.  The MN firing times fitted
earlier are loaded once from disk; every pygpc sample only regenerates the
MUAPs (a, b, lam, delay) and superposes them at those firing times.

Only output: R2 between simulated and measured MEPs.
"""

import os
import h5py
import numpy as np

from gen_muaps import gen_muaps
from ga_MEPmodel_bio import ga_MEPmodel_bio

# ==========================================================================
def run_model(b, lam, subj, WITHRC, AMPAWEIGHT, p):
    """
    Call to the biological spinal periferal model.

    a, b : floats
        MUAP amplitude distribution parameters [a, b] A = a*b^x. a is hard coded to 6.28
    lam  : float
        MUAP shape parameter (axonal delay = 2.5 * lam)
    subj : int
        the subject to fit
    withRC: int
        0 for the model without Renshaw cells, 1 for the model with Renshaw cells
    AMPAweight: list of floats
        [] to select the best value
    spike_file: str
        the name of the file containing the MN spike times

    Returns
    -------
    R2 : float
        Goodness of fit of the simulated vs. measured MEPs
    """

    # ----- generate MUAPs for this sample -----
    delay = np.ones(100) * (2.35 * lam - 0.18)                                   # fixed parameter
    _, _ = gen_muaps(n_neurons=100,
                        amplitude=[6.28, b],                                # fix a to 6.28
                        axonalDelay=delay,
                        lam=lam)

    ref = ga_MEPmodel_bio(subj, withRC=WITHRC, AMPAweight=AMPAWEIGHT, reRun=0, p_post=p)

    return ref['R2']


# ==========================================================================
if __name__ == '__main__':
    # fixed settings 
    SUBJ       = 1
    WITHRC     = 1
    AMPAWEIGHT = None

    # pygpc parameters
    a   = 5      # pygpc parameter [2, 14] uniform
    b   = 100    # pygpc parameter [45, 425] uniform
    lam = 3      # pygpc parameter [1.111 - 5.706], normal distribution mu=3.619, std=0.774
    p_post = [0.86032462, 1.05927459, 1.57477574, 0.76463778, 9.80825998, 19.96654087, 11.40240176, 0.14947802, 0.04018722, 3.42343178, 5., 0.14387608]

    # Output metric
    R2 = run_model(b, lam, SUBJ, WITHRC, AMPAWEIGHT, p_post)
    print(R2)
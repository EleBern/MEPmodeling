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
def run_model(b, lam, subj, withRC, AMPAweight,
              spike_file):
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

    ref = ga_MEPmodel_bio(subj, withRC=1, AMPAweight=None, reRun=0)

    return ref['R2']


# ==========================================================================
if __name__ == '__main__':
    # fixed settings 
    SUBJ       = 1
    WITHRC     = 1
    AMPAWEIGHT = None

    ROOT       = os.path.dirname(os.path.realpath(__file__))
    SPIKE_FILE = os.path.join(ROOT, 'fitted_results', 'bio', f'mu_spiketimes_S{SUBJ}.h5')

    # pygpc parameters
    a   = 5      # pygpc parameter [2, 14] uniform
    b   = 100    # pygpc parameter [45, 425] uniform
    lam = 3      # pygpc parameter [1.111 - 5.706], normal distribution mu=3.619, std=0.774

    # Output metric
    R2 = run_model(b, lam, SUBJ, WITHRC, AMPAWEIGHT, SPIKE_FILE)
    print(R2)
"""
pygpc forward model for the biological MEP model.

The spinal network is NOT re-simulated here.  The MN firing times fitted
earlier are loaded once from disk; every pygpc sample only regenerates the
MUAPs (a, b, lam) and superposes them at those firing times.

Only output: R2 between simulated and measured MEPs.
"""

import os
import h5py
import numpy as np

from gen_muaps import gen_muaps
from h5_helpers import load_h5_to_dict
from config_model_bio import config_model_bio
from MEPmodel_bio import cal_error
from MEPmodel_bio_core import MEPmodel_bio_core

# ----- fixed settings -----
SUBJ       = 1
WITHRC     = 1
AMPAWEIGHT = None

ROOT       = os.path.dirname(os.path.realpath(__file__))
SPIKE_FILE = os.path.join(ROOT, 'fitted_results', 'bio', f'mu_spiketimes_S{SUBJ}.h5')

# setup is expensive, so build it once and reuse it for every pygpc sample
_CACHE = {}


# ==========================================================================
def get_setup(subj=SUBJ, withRC=WITHRC, AMPAweight=AMPAWEIGHT, spike_file=None):
    """
    Target MEP (ref) + previously saved MN firing times.  Built on the first
    call, cached afterwards.
    """
    spike_file = spike_file or SPIKE_FILE
    key = (subj, withRC, spike_file)

    if key not in _CACHE:
        ref = config_model_bio(subj, withRC, AMPAweight)

        if not os.path.isfile(spike_file):
            raise FileNotFoundError(
                f'Saved spike times not found: {spike_file}\n'
                'Run the fitted model once (MEPmodel_bio_core) to generate it.'
            )
        with h5py.File(spike_file, 'r') as f:
            spike_times = np.asarray(load_h5_to_dict(f)['spike_times'], dtype=float)

        nIntensities = ref['y0'].shape[1]
        if spike_times.shape[2] != nIntensities:
            raise ValueError(
                f'{spike_file} holds {spike_times.shape[2]} TMS intensities, '
                f'but subject {subj} has {nIntensities}.'
            )

        _CACHE[key] = (ref, spike_times)

    return _CACHE[key]


# ==========================================================================
def run_model(a, b, lam, subj=SUBJ, withRC=WITHRC, AMPAweight=AMPAWEIGHT,
              spike_file=None):
    """
    pygpc forward model.

    a, b : MUAP amplitude range [a, b]
    lam  : MUAP shape parameter (axonal delay = 2.5 * lam)

    Returns R2 of the simulated vs. measured MEPs.
    """
    ref, spike_times = get_setup(subj, withRC, AMPAweight, spike_file)

    # ----- generate MUAPs for this sample -----
    delay = 2.5 * lam                                   # fixed parameter
    muaps, tmuap = gen_muaps(n_neurons=spike_times.shape[0],
                             amplitude=[a, b],
                             axonalDelay=delay,
                             lam=lam)
    ref['model']['muaps'] = muaps
    ref['model']['tmuap'] = tmuap

    # ----- rebuild MEP from the saved spike times (no network simulation) -----
    sim = MEPmodel_bio_core(ref['model'], spike_times=spike_times)

    # ----- output metric -----
    ref = cal_error(ref, sim)

    return ref['R2']


# ==========================================================================
if __name__ == '__main__':
    a   = 5      # pygpc parameter
    b   = 100    # pygpc parameter
    lam = 3      # pygpc parameter

    R2 = run_model(a, b, lam)
    print(R2)
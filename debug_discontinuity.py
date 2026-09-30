import os
import numpy as np
import matplotlib.pyplot as plt
from gen_muaps import gen_muaps
from pygpc_main import get_setup
from MEPmodel_bio import cal_error
from MEPmodel_bio_core import MEPmodel_bio_core

def run_model(b, lam, subj, withRC, AMPAweight, spike_file):
    _CACHE = {}
    ref = {}
    ref, spike_times = get_setup(subj, withRC, AMPAweight, spike_file)
    ref["spike_times"] = spike_times

    # ----- generate MUAPs for this sample -----
    delay = np.ones(100) * 8.325#(2.35 * lam - 0.18)                                   # fixed parameter
    muaps, tmuap = gen_muaps(n_neurons=100,
                             amplitude=[6.28, b],                                # fix a to 6.28
                             axonalDelay=delay,
                             lam=lam)
    # muaps, tmuap = load_muap() # Test to see if it reproduces the same R2 as with fitted parameters
    ref['model']['muaps'] = muaps
    ref['model']['tmuap'] = tmuap

    # ----- rebuild MEP from the saved spike times (no network simulation) -----
    sim = MEPmodel_bio_core(ref['model'], spike_times=spike_times)

    # ----- output metric -----
    ref = cal_error(ref, sim)

    return ref


if __name__ == '__main__':
    # fixed settings 
    SUBJ       = 1
    WITHRC     = 1
    AMPAWEIGHT = None

    ROOT       = os.path.dirname(os.path.realpath(__file__))
    SPIKE_FILE = os.path.join(ROOT, 'fitted_results', 'bio', f'mu_spiketimes_S{SUBJ}.h5')

    # pygpc parameters
    b   = 100    # pygpc parameter [45, 425] uniform
    lam = 3
    b_array = np.arange(100,150,1)
    lam_array = np.arange(3,4,0.1)      # pygpc parameter [1.111 - 5.706], normal distribution mu=3.619, std=0.774

    ref = run_model(b, lam, SUBJ, WITHRC, AMPAWEIGHT, SPIKE_FILE)
    tmuap = ref['model']['tmuap']
    spike_times = ref["spike_times"]
    y0 = ref["y0"]

    plt.figure()
    # for lam in lam_array:
    for b in b_array:
        # Output metric
        ref = run_model(b, lam, SUBJ, WITHRC, AMPAWEIGHT, SPIKE_FILE)
        print(ref["R2"])
        #plt.plot(lam, ref["R2"], "*k")
        plt.plot(b, ref["R2"], "*k")
        if not np.allclose(tmuap, ref['model']['tmuap']):
            print("Error in tmuap, con b= ", b)
        if not np.allclose(spike_times, ref['spike_times'], equal_nan=True):
                    print("Error in spike_times, con b= ", b)
        if not np.allclose(y0, ref['y0'], equal_nan=True):
                            print("Error in y0, con b= ", b)
    plt.show()
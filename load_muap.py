import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from h5_helpers import load_h5_to_dict


def load_muap(plotOn=0, amplitudeDist=0):
    """
    Load motor unit action potentials (MUAPs) 

    Parameters
    ----------
    plotOn     : int or bool
                 If truthy, plot all MUAPs (default 0).

    Returns
    -------
    muaps : np.ndarray  [n_samples x n_muaps]   MUAPs 
    t     : np.ndarray  [n_samples x 1]         time vector (ms)
    """
    root    = os.path.dirname(os.path.realpath(__file__))
    h5_path = os.path.join(root, "data_MUAP", "pheno_muaps.h5")

    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            tmp = load_h5_to_dict(f)

        muaps = tmp["muaps"]   # [n_samples x n_muaps]
        t     = tmp["t"].T     # [n_samples x 1]

    if amplitudeDist:
        amplitude_distribution(muaps)

    if plotOn:
        fig = plt.figure()
        for i in range(1, 21):   # 1:20
            ax = fig.add_subplot(4, 5, i)
            start_col = (i - 1) * 5
            end_col   = i * 5

            ax.plot(t, 1e6 * muaps[:, start_col:end_col], linewidth=1.5)
            ax.grid(True)

            labels = [str(val) for val in range(start_col + 1, end_col + 1)]
            ax.legend(labels, loc="upper right")

        fig.supxlabel('ms')
        fig.supylabel('μV')
        plt.show()

    return muaps, t

def amplitude_distribution(muaps):
    max_peak = np.max(muaps, axis=0)
    print(np.max(max_peak))
    min_peak = np.min(muaps, axis=0)
    amplitude = 1e3 * np.sort(max_peak - min_peak)
    popt = fit_amplitude(amplitude)
    fig = plt.figure()
    print(np.shape(max_peak))
    plt.plot(np.arange(len(amplitude)), amplitude, "*", label="MUAP amplitude")
    plt.plot(np.arange(len(amplitude)), exponential(np.arange(len(amplitude)) / (len(amplitude)-1), *popt), "r", label="Best fit")
    plt.title("MUAPs amplitude distribution")
    plt.ylabel("Amplitude [mV]")
    plt.xlabel("Motor unit sorted index")
    plt.xlim([0, 100])
    plt.legend()
    plt.show()

def exponential(x, a, b):
    return a*b**(x)

def fit_amplitude(amplitude):
    x = np.arange(len(amplitude)) / (len(amplitude)-1)
    #y = exponential(x, 2.5)
    popt, pcov = curve_fit(exponential, x, amplitude, p0=[1e-5, 100])
    print(popt)

    return popt
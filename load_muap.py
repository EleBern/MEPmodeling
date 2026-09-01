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

    # Calculate and fit MUAPs amplitude distribution
    if amplitudeDist:
        _ = amplitude_distribution(muaps)

    if plotOn:
        fig = plt.figure()
        for i in range(1, 21):  
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
    """
        Plot the MUAPs amplitude distribution and its best fit
    
        Parameters
        ----------
        muaps   : np.array [n_muaps,] MUAPs
    """
    max_peak = np.max(muaps, axis=0)
    print("Peak amplitude (from 0 V to positive peak) of largest MUAP: ", np.max(max_peak), " V")
    min_peak = np.min(muaps, axis=0)
    amplitude = 1e3 * (max_peak - min_peak)
    popt = fit_amplitude(amplitude)
    fig = plt.figure()
    plt.plot(np.arange(len(amplitude)), amplitude, "*", label="MUAP amplitude")
    plt.plot(np.arange(len(amplitude)), exponential(np.arange(len(amplitude)) / (len(amplitude)-1), *popt), "r", label="Best fit")
    plt.title("MUAPs amplitude distribution")
    plt.ylabel("Amplitude [mV]")
    plt.xlabel("Motor unit sorted index")
    plt.xlim([0, 100])
    plt.legend()
    plt.show()
    return popt


def exponential(x, a, b):
    return a*b**(x)

def fit_amplitude(amplitude):
    """
        Find the best fit for the MUAPs amplitude distribution, prints the best fit parameters
    
        Parameters
        ----------
        amplitude   : np.array [n_muaps,] amplitude of the MUAPs
                    
    
        Returns
        -------
        popt : np.array  [2,]   best exponential fit parameters: minimum amplitude, base of the exponential
    """
    x = np.arange(len(amplitude)) / (len(amplitude)-1)
    popt, _ = curve_fit(exponential, x, amplitude, p0=[1e-5, 100])
    print("Optimal exponential parameters (scaling factor, base):", popt)

    return popt
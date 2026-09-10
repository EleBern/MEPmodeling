"""
Helper functions to generate synthetic MUAPS
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def time_vectors():
    """
    Generates and returns the time vectors of the MUAPs (tmuap) and of the
    motor neurons (t_mn)
    """
    tmuap = np.arange(0, 20, 0.1)
    t_mn = np.linspace(0, 20, 200)
    return tmuap, t_mn


def gof(anatomical_muaps, muaps):
    """
    Goodness of fit of each synthetic MUAP to the corresponding anatomically
    derived MUAP, in terms of coefficient of determination and normalised root
    mean square error.

    Parameters
    ----------
    anatomical_muaps : ndarray, shape (200, N)
        Anatomically derived MUAPs [V], one per column.
    muaps : ndarray, shape (200, N)
        Synthetic MUAPs [V], one per column.

    Returns
    -------
    R2 : ndarray, shape (N,)
        Coefficient of determination R^2 = 1 - SS_res / SS_tot of each MUAP.
        NaN for a flat anatomical MUAP (SS_tot = 0, R^2 undefined).
    NRMSE : ndarray, shape (N,)
        Root mean square error of each MUAP divided by the amplitude of the
        anatomical MUAP, i.e. by (max - min) / 2, the same definition of
        amplitude used to generate the synthetic MUAPs. Dimensionless, so it
        is comparable between small and large motor units. NaN for a flat
        anatomical MUAP (zero amplitude).
    """
    anatomical_muaps = np.asarray(anatomical_muaps, dtype=float)
    muaps = np.asarray(muaps, dtype=float)

    residual = muaps - anatomical_muaps
    RMSE = np.sqrt(np.mean(residual ** 2, axis=0))

    # RMSE normalised by the amplitude of the anatomical MUAP
    amplitude = (np.max(anatomical_muaps, axis=0) - np.min(anatomical_muaps, axis=0)) / 2
    NRMSE = np.where(amplitude > 0, RMSE / np.where(amplitude > 0, amplitude, 1.0), np.nan)

    # R^2 against the mean of the anatomical MUAP as reference model
    SS_res = np.sum(residual ** 2, axis=0)
    SS_tot = np.sum((anatomical_muaps - np.mean(anatomical_muaps, axis=0)) ** 2, axis=0)
    R2 = np.where(SS_tot > 0, 1.0 - SS_res / np.where(SS_tot > 0, SS_tot, 1.0), np.nan)

    return R2, NRMSE


def fit_lam(anatomical_muaps, amplitude, axonalDelay, p0=(0.5, 1.0, 2.0, 4.0, 8.0),
            lam_bounds=(0.05, 20.0)):
    """
    Fit the shape parameter lambda of the first-order Hermite-Rodriguez
    function to each anatomically derived MUAP with scipy.optimize.curve_fit.

    Parameters
    ----------
    anatomical_muaps : ndarray, shape (200, N)
        Anatomically derived MUAPs [V], one per column.
    amplitude : ndarray, shape (N,)
        Half peak-to-peak amplitude A_i of each MUAP [V].
    axonalDelay : ndarray, shape (N,)
        Axonal delay tau_i of each motor unit [ms].
    p0 : sequence of float, optional
        Initial guesses for lambda [ms].
    lam_bounds : (float, float), optional
        Lower and upper bound for lambda [ms].

    Returns
    -------
    lam : ndarray, shape (N,)
        Best-fitting lambda of each motor unit [ms].
    """
    # Same time vectors as in gen_muaps
    tmuap, t_mn = time_vectors()

    anatomical_muaps = np.asarray(anatomical_muaps, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    axonalDelay = np.asarray(axonalDelay, dtype=float)

    n_mu = anatomical_muaps.shape[1]
    lam = np.full(n_mu, np.nan)

    for i in range(n_mu):
        z = axonalDelay[i] - t_mn - tmuap        # (200,)
        y = anatomical_muaps[:, i]               # (200,)
        A = amplitude[i]

        if not np.isfinite(A) or A <= 0:
            continue                              # flat MUAP, nothing to fit

        def hermite_rodriguez(z, lam_i):
            shape = z * np.exp(-(z / lam_i) ** 2)
            return shape / np.max(shape)          # unit-peak waveform

        # Both the model and the data are divided by A_i: this leaves the
        # minimum unchanged but keeps the residuals of the order of 1, which
        # is needed because curve_fit stops on an absolute gradient tolerance
        # and would otherwise return p0 unchanged for data of ~1e-6 V.
        y_n = y / A

        best_sse = np.inf
        for guess in p0:
            try:
                popt, _ = curve_fit(hermite_rodriguez, z, y_n, p0=[guess],
                                    bounds=([lam_bounds[0]], [lam_bounds[1]]))
            except (RuntimeError, ValueError):
                continue                          # this start did not converge
            sse = np.sum((hermite_rodriguez(z, popt[0]) - y_n) ** 2)
            if sse < best_sse:
                best_sse = sse
                lam[i] = popt[0]

    bad = ~np.isfinite(lam)
    if bad.any():
        lam[bad] = np.nanmedian(lam)
        print(f"Warning: lambda could not be fitted for {bad.sum()} MUAP(s); "
              f"the median lambda was used instead.")

    return lam


def amplitude_distribution(muaps, dt=0.01, plotOn=False):
    """
        Find the best fit for the MUAPs amplitude distribution.
        Prints the best fit parameters. Plots the MUAPs amplitude distribution and its best fit
    
        Parameters
        ----------
        muaps   : np.array [samples, n_muaps] MUAPs
        dt      : float the times step of the recorded MUAPs

        Returns
        -------
        popt : ndarray, shape (2,)
            Best-fitting a and b for the amplitude distribution
        amplitude : ndarray, shape (n_muaps,)
            The amplitude of each MUAP.
    """

    amplitude = np.zeros(np.shape(muaps)[1])
    for i in range(np.shape(muaps)[1]):
        peak_times = find_peak_times(muaps[:, i], dt)
        # Calculate the amplitude of the MUAPs from the peaks
        if len(peak_times) > 1:
            temp = np.diff(muaps[peak_times, i])
            amplitude[i] = np.max(np.abs(temp)) / 2
            # Allow for negative amplitudes
            index = np.argmax(np.abs(np.diff(muaps[peak_times, i])))
            if temp[index] > 0:
                amplitude[i] *= -1
        else:
            amplitude[i] = 0

    #popt = fit_amplitude(np.abs(amplitude))
    x = np.arange(len(amplitude)) / (len(amplitude)-1)
    popt, _ = curve_fit(exponential, x, np.abs(amplitude), p0=[1e-5, 100])
    print("Optimal exponential parameters (scaling factor, base):", popt)

    if plotOn:
        fig = plt.figure()
        plt.plot(np.arange(len(amplitude)), amplitude, "*", label="MUAP amplitude")
        plt.plot(np.arange(len(amplitude)), exponential(np.arange(len(amplitude)) / (len(amplitude)-1), *popt), "r", label="Best fit")
        plt.title("MUAPs amplitude distribution")
        plt.ylabel("Amplitude [V]")
        plt.xlabel("Motor unit sorted index")
        plt.xlim([0, 100])
        plt.legend()
        plt.show()
    return popt, amplitude


def exponential(x, a, b):
    return a*b**(x)


def find_peak_times(muap, dt):
    """
        Find the peak times of the MUAP
        Find peaks with a minimum height obest fit for the MUAPs amplitude distribution.f 1/5 of the maximum voltage
        Peaks must be at least 1.5 ms apart, if not they are considered from the same oscillation
    
        Parameters
        ----------
        muap    : np.array [samples,] MUAP
        dt      : float the times step of the recorded MUAP

        Returns
        -------
        peak_times : ndarray, 
            Indices of the timing of the MUAP peaks
    """

    height = np.max(1e6 * muap) / 5
    pos_peaks, _ = find_peaks(1e6 * muap, height=height, distance=1.5/dt) 
    neg_peaks, _ = find_peaks(- 1e6 * muap, height=height, distance=1.5/dt)
    peak_times = np.sort(np.concatenate([pos_peaks, neg_peaks]))
    return peak_times
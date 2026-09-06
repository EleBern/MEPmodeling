"""
gen_muaps.py
============

Generate Motor Unit Action Potential (MUAP) waveforms using the first-order
Hermite-Rodriguez function, as described in:

    https://journals.physiology.org/doi/full/10.1152/jn.00626.2017

(Eqs. 4-5)
-----------------------

Differences from the paper:
    - The whole MUAP waveform is generated and saved, it is then sampled according to the motor neurons' spike times 
      in MEPmodel_bio_core.py and MEPmodel_pheno_core.py. For this purpose:
        - The heaviside function is not used
        - Motor neuron spike times are replaced by an array of times between 0 and 20 ms with 0.1 ms steps
The parameters of the synthetic MUAPs are not taken from the paper, but fit to anatomically derived
MUAPs (https://pubmed.ncbi.nlm.nih.gov/31465437/). For this purpose:
    - The axonal delay is set to align the zero-crossing of the synthetic MUAPs with the zero-crossing of
      reference anatomical MUAPs
    - Either the MUAP amplitude distribution, or the amplitude of each MUAP
      reproduces that of the anatomically derived 
    - The duration of the synthetic MUAPs (lambda) is fit to the anatomical MUAPs.
      Either each MUAP duration is fit, or an average is used.
"""

import os
import h5py
import numpy as np
from scipy.optimize import curve_fit
from h5_helpers import load_h5_to_dict
from load_muap import amplitude_distribution
from zero_crossing import crossing_times


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
    tmuap = np.arange(0, 20, 0.1)
    t_mn = np.linspace(0, 20, 200)

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


def gen_muaps(n_neurons, amplitude, axonalDelay, lam):
    """
    Generate MUAP waveforms using the first-order Hermite-Rodriguez
    function described in the paper (Eqs. 4-5). See module docstring for
    full details.

    Parameters
    ----------
    n_neurons : int, optional
        Number of motor neurons (default 100)
    a : float, optional
        Amplitude scale constant [V] for the first motor unit (default
        3.75457942e-06/2).
    b : float, optional
        The base of the exponential of the amplitude of the MUAPs
        (default 5.28518724e+02).
    lam : float or ndarray, shape (N,)
        Shape parameter of the Hermite-Rodriguez function [ms], one value per
        motor unit as returned by fit_lam.

    Returns
    -------
    muaps : ndarray, shape (200, N)
    tmuap : ndarray, shape (200,)
    """
    # Paper parameters
    if isinstance(lam, np.ndarray):
        lam = np.atleast_1d(np.asarray(lam, dtype=float))[None, :]

    # Time vector
    tmuap = np.arange(0, 20, 0.1)
    t_mn = np.linspace(0, 20, 200)

    # Amplitude A_i: exponential (log-spaced) distribution over a n_neurons-fold
    # range, from A_1 = a (first-recruited MU) to A_M = b*a
    # (last-recruited MU). Motor units are assumed ordered by recruitment
    # order (column index 0 = first recruited).
    if len(amplitude) == 2:
        a = amplitude[0]
        b = amplitude[1]
        if n_neurons > 1:
            frac = np.arange(n_neurons) / (n_neurons - 1)
        else:
            frac = np.zeros(1)
        A = a * b ** frac  
    else:
        A = amplitude
    print(A)
    # Eq. 4: H_i(t) = A_i * (tau_i - t) * exp(-((tau_i - t)/lambda)^2) * u(tau_i - t)
    t_M = t_mn[:, None]      # (200, 1)
    t_D = axonalDelay        # shape (N,)
    t_D = t_D[None, :]       # (1, N)
    t_MUAP = tmuap[:, None]       # (1, N)
    A = A[None, :]           # (1, N)
  
    print(np.shape(t_M), np.shape(t_D), np.shape(t_MUAP))
    #z = axonalDelay - t_mn - tmuap  # (200, N)
    z = t_D - t_M - t_MUAP # (200, N)

    #z = np.vstack((z,) * 100).T
    # Peak of each MUAP taken separately (lambda now differs between motor
    # units, and the peak of z*exp(-(z/lam)^2) scales with lambda)
    normalization_factor = np.max(z * np.exp(-(z / lam) ** 2), axis=0, keepdims=True) # To ensure that Am = b * A1
    muaps = A * (z * np.exp(-(z / lam) ** 2)) / normalization_factor 

    return muaps, tmuap


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Import anatomical MUAPs and calculate the amplitude distribution
    root    = os.path.dirname(os.path.realpath(__file__))
    h5_path = os.path.join(root, "data_MUAP", "muap.h5")

    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            tmp = load_h5_to_dict(f)

    anatomical_muaps = tmp["muaps"]   # [n_samples x n_muaps]
    # Calculate and fit MUAPs amplitude distribution
    popt = amplitude_distribution(anatomical_muaps)
    max_peak = np.max(anatomical_muaps, axis=0)
    print("Peak amplitude (from 0 V to positive peak) of largest MUAP: ", np.max(max_peak), " V")
    min_peak = np.min(anatomical_muaps, axis=0)
    amplitude = (max_peak - min_peak) / 2

    h5_path = os.path.join(root, "data_MUAP", "Dist1_Monopolar_Rest_NormalCV_New.hdf5")

    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            tmp = load_h5_to_dict(f)
    anatomical_muaps2 = tmp["MUAPShapes"]
    cond = np.sum(np.abs(anatomical_muaps2), axis=1) != 0
    first_nonzero = np.argmax(cond)      # 0-based index of first True (assumes at least one nonzero row)
    idx = first_nonzero - 1              # keep one zero row before the signal starts, like MATLAB's idx = find(...)-1

    t2 = np.linspace(0, 20, 20001)
    anatomical_muaps2 = -anatomical_muaps2[idx:, :]              # flipped (sign)
    t2 = t2[idx:]
    t2 = t2 - t2.min()
    axonalDelay = 2 * crossing_times(t2, anatomical_muaps2)

    # Fit one lambda per anatomical MUAP (100 MUAPs -> 100 lambdas)
    lam = fit_lam(anatomical_muaps, amplitude, axonalDelay)
    print("lam shape:", lam.shape)
    print("lam range (ms):", lam.min(), "to", lam.max())

    # Generate the synthetic MUAPs with that amplitude distribution
    N = 100

    muaps, tmuap = gen_muaps(n_neurons=N, amplitude=popt, axonalDelay=axonalDelay, lam=lam)

    print("muaps shape:", muaps.shape)  # (200, N)
    print("tmuap shape:", tmuap.shape)  # (200,)
    print("tmuap range:", tmuap[0], "to", tmuap[-1])
    print("Amplitude range (V):", muaps.max(axis=0).min(), "to", muaps.max(axis=0).max())
    print("max |muaps[0]| (V):", np.abs(muaps[0]).max())
    print("min |muaps[0]| (V):", np.abs(muaps[0]).min())
    print(np.argwhere(np.isnan(muaps[0])))

    plt.figure(figsize=(8, 5))
    for n in range(N):
        plt.figure()
        plt.plot(tmuap, 1e3 * anatomical_muaps[:, n], "k")
        plt.plot(tmuap, 1e3 * muaps[:, n], "r")#label=f"MU {n + 1}")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Amplitude (mV)")
    # plt.title("Simulated MUAP shapes (first-order Hermite-Rodriguez function)")
        plt.xlim([0, 20])
        # plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.show()

    # Save muaps and the corresponding time vector (t_muaps) to an HDF5
    out_dir = os.path.join(os.getcwd(), "data_MUAP")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pheno_muaps.h5")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("muaps", data=muaps)
        f.create_dataset("t", data=tmuap)

    print(f"Saved MUAPs to {out_path}")
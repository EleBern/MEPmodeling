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
    - Either the MUAP amplitude distribution (real_A = False), or the amplitude of each MUAP (real_A = True)
      is set to reproduce that of the anatomically derived  MUAPs
    - The duration of the synthetic MUAPs (lambda) is fit to the anatomical MUAPs.
      Either each MUAP duration is fit (real_lam), or an average is used (real_lam).
"""

import os
import h5py
import numpy as np
from zero_crossing import crossing_times
from helper_f_genmuaps import amplitude_distribution, fit_lam, gof, time_vectors

def gen_muaps(n_neurons, amplitude, axonalDelay, lam, zero_muaps=None):
    """
    Generate MUAP waveforms using the first-order Hermite-Rodriguez
    function described in the paper (Eqs. 4-5). See module docstring for
    full details.

    Parameters
    ----------
    n_neurons    : int
        Number of motor neurons
    amplitude    : ndarray
        Either amplitude of each MUAP (n_neurons,)
        Or scaling and base of exponent of the amplitude distribution (2,)
    axonal delay : ndarray, shape (n_neurons,)
        Zero-crossing time of each anatomical MUAP
    lam          : float or ndarray, shape (n_neurons,)
        Shape parameter of the Hermite-Rodriguez function [ms]. One value per
        motor unit as returned by fit_lam, or a mean value.
    zero_muaps   : ndarray of bool, shape (n_neurons,), optional
        Motor units whose anatomical MUAP is all zero. The corresponding
        synthetic MUAPs are set to all zero as well. If None, every MUAP is
        generated from the Hermite-Rodriguez function.

    Returns
    -------
    muaps : ndarray, shape (200, n_neurons)
    tmuap : ndarray, shape (200,)
    """
    # Paper parameters
    if isinstance(lam, np.ndarray):
        lam = np.atleast_1d(np.asarray(lam, dtype=float))[None, :]

    # Time vectors
    tmuap, t_mn = time_vectors()

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

    t_M = t_mn[:, None]      # (200, 1)
    t_D = axonalDelay        # shape (N,)
    t_D = t_D[None, :]       # (1, N)
    t_MUAP = tmuap[:, None]       # (1, N)
    A = A[None, :]           # (1, N)
  
    z = t_D - t_M - t_MUAP # (200, N)

    # Eq. 4: H_i(t) = A_i * (tau_i - t) * exp(-((tau_i - t)/lambda)^2) * u(tau_i - t)
    normalization_factor = np.max(z * np.exp(-(z / lam) ** 2), axis=0, keepdims=True) # To ensure that Am = b * A1
    muaps = A * (z * np.exp(-(z / lam) ** 2)) / normalization_factor 

    # An all-zero anatomical MUAP gives an all-zero synthetic MUAP. Its
    # amplitude and zero-crossing time are undefined (0 and NaN), so the
    # column is overwritten instead of being computed
    if zero_muaps is not None:
        muaps[:, np.asarray(zero_muaps, dtype=bool)] = 0.0

    return muaps, tmuap


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from downsample_muaps import load_unprocessed_muaps

    verbose = True   # Print info on MUAP parameters
    plotOn = False   # Plot generated MUAPs
    real_A = True   # If False, use amplitude distribution of anatomical MUAPs
    real_lam = True  # If False, use mean lambda

    # Import anatomical MUAPs
    root    = os.path.dirname(os.path.realpath(__file__))
    h5_path = os.path.join(root, "data_MUAP", "Dist1_Monopolar_Rest_NormalCV_New.hdf5")

    t, anatomical_muaps, downsampled_t, downsampled_muaps = load_unprocessed_muaps(h5_path)

    # Calculate the amplitude and amplitude distribution of the anatomical MUAPs
    popt, amplitude = amplitude_distribution(anatomical_muaps, t[1]-t[0])
    max_peak = np.max(anatomical_muaps, axis=0)
    min_peak = np.min(anatomical_muaps, axis=0)
    #amplitude = (max_peak - min_peak) / 2

    # Find the zero-crossing of the anatomical MUAPs
    axonalDelay = 2 * crossing_times(t, anatomical_muaps)

    # Motor units without an anatomical MUAP (all-zero waveform). They have no
    # amplitude, duration or zero-crossing time, so their synthetic MUAP is set
    # to all zero. The exception is a fully synthetic MUAP (real_A = False and
    # real_lam = False): its amplitude comes from the fitted distribution and
    # its duration from the mean lambda, so it is generated like any other MUAP
    # and only its missing zero-crossing time is replaced by the mean one
    zero_muaps = np.all(anatomical_muaps == 0, axis=0)
    if not (real_A or real_lam):
        axonalDelay[zero_muaps] = np.nanmean(axonalDelay[~zero_muaps])
        zero_muaps = None

    # Fit one lambda per anatomical MUAP (100 MUAPs -> 100 lambdas)
    lam = fit_lam(downsampled_muaps, amplitude, axonalDelay)
    if not real_lam:
        lam = np.mean(lam)
        axonalDelay = np.ones(len(axonalDelay)) * np.mean(axonalDelay)
    

    # Generate the synthetic MUAPs with that amplitude distribution
    N = 100

    if real_A:
        muaps, tmuap = gen_muaps(n_neurons=N, amplitude=amplitude, axonalDelay=axonalDelay, lam=lam,
                                 zero_muaps=zero_muaps)
    else:
        muaps, tmuap = gen_muaps(n_neurons=N, amplitude=popt, axonalDelay=axonalDelay, lam=lam,
                                 zero_muaps=zero_muaps)

    # Calculate the goodness of fit
    R2, NRMSE = gof(downsampled_muaps, muaps)

    if verbose:
        print("Peak amplitude (from 0 V to positive peak) of largest anatomical MUAP: ", np.max(max_peak), " V")
        print("Amplitude range (V):", muaps.max(axis=0).min(), "to", muaps.max(axis=0).max())
        print("max |muaps[0]| (V):", np.abs(muaps[0]).max())
        print("min |muaps[0]| (V):", np.abs(muaps[0]).min())
        print("lam shape:", lam.shape)
        print("lam range (ms):", lam.min(), "to", lam.max())
        print("delay range (ms):", np.nanmin(axonalDelay), "to", np.nanmax(axonalDelay))
        # print("The normalised RMSE is ", NRMSE)
        # print("The R^2 is ", R2)
        print("Median R^2: %.4f, median normalised RMSE: %.4f" % (np.nanmedian(R2), np.nanmedian(NRMSE)))

    if plotOn:
        plt.figure(figsize=(8, 5))
        for n in range(N):
            plt.plot(tmuap, 1e3 * muaps[:, n], label=f"MU {n + 1}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (mV)")
        plt.title("Simulated MUAP shapes (first-order Hermite-Rodriguez function)")
        plt.xlim([0, 20])
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.show()

        # Plot anatomical against synthetic MUAPs
        for n in range(N):
            plt.figure()
            plt.plot(downsampled_t, 1e3 * downsampled_muaps[:, n], "k", label="Anatomical MUAP")
            plt.plot(tmuap, 1e3 * muaps[:, n], "r", label="Synthetic MUAP")
            plt.xlim([0, 20])
            plt.title("Shape of MUAP {0}".format(n))
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (mV)")
            plt.legend()
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
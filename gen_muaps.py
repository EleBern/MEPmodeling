"""
gen_muaps.py
============

Generate Motor Unit Action Potential (MUAP) waveforms using the first-order
Hermite-Rodriguez function, exactly as described in:

    https://journals.physiology.org/doi/full/10.1152/jn.00626.2017

(Eqs. 4-5)
-----------------------

Difference from the paper:
    - The whole MUAP waveform is generated and saved, it is then sampled according to the motor neurons' spike times 
      in MEPmodel_bio_core.py and MEPmodel_pheno_core.py. For this purpose:
        - The heaviside function is not used
        - Motor neuron spike times are replaced by an array of times between 0 and 20 ms with 0.1 ms steps
        - The axonal delay is arbitrary. The peaks of the simulated MEPs are aligned to the peaks of the recorded
          MEPs, giving the axonal delay
    - The MUAP amplitude distribution should reproduce that of the anatomically derived MUAPs (https://pubmed.ncbi.nlm.nih.gov/31465437/).
      For this purpose:
        - A1 is set to 3.75457942e-06/2 
        - Am = 5.28518724e+02 * A1
"""

import os
import h5py
import numpy as np
from h5_helpers import load_h5_to_dict
from load_muap import amplitude_distribution


def gen_muaps(n_neurons=100, a=3.75457942e-06/2, b=5.28518724e+02):
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

    Returns
    -------
    muaps : ndarray, shape (200, N)
    tmuap : ndarray, shape (200,)
    """
    # Paper parameters
    lam = 2
    axonalDelay = 3 * lam + 1

    # Time vector
    tmuap = np.arange(0, 20, 0.1)
    t_mn = np.linspace(0, 20, 200)

    # Amplitude A_i: exponential (log-spaced) distribution over a n_neurons-fold
    # range, from A_1 = a (first-recruited MU) to A_M = b*a
    # (last-recruited MU). Motor units are assumed ordered by recruitment
    # order (column index 0 = first recruited).
    if n_neurons > 1:
        frac = np.arange(n_neurons) / (n_neurons - 1)
    else:
        frac = np.zeros(1)
    A = a * b ** frac  

    # Eq. 4: H_i(t) = A_i * (tau_i - t) * exp(-((tau_i - t)/lambda)^2) * u(tau_i - t)
    t_M = t_mn[:, None]      # (200, 1)
    t_D = np.ones(n_neurons) * axonalDelay  # shape (N,)
    t_D = t_D[None, :]      # (1, N)
    A = A[None, :]          # (1, N)
  
    z = axonalDelay - t_mn - tmuap  # (200, N)
    z = np.vstack((z,) * 100).T
    normalization_factor = np.max(z * np.exp(-(z / lam) ** 2)) # To ensure that Am = b * A1
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


    # Generate the synthetic MUAPs with that amplitude distribution
    N = 100

    muaps, tmuap = gen_muaps(n_neurons=N, a=popt[0], b=popt[1])

    print("muaps shape:", muaps.shape)  # (200, N)
    print("tmuap shape:", tmuap.shape)  # (200,)
    print("tmuap range:", tmuap[0], "to", tmuap[-1])
    print("Amplitude range (V):", muaps.max(axis=0).min(), "to", muaps.max(axis=0).max())
    print("max |muaps[0]| (V):", np.abs(muaps[0]).max())
    print("min |muaps[0]| (V):", np.abs(muaps[0]).min())

    plt.figure(figsize=(8, 5))
    for n in range(N):
        plt.plot(tmuap, 1e3 * muaps[:, n], label=f"MU {n + 1}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (mV)")
    plt.title("Simulated MUAP shapes (first-order Hermite-Rodriguez function)")
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
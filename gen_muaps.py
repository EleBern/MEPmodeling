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
          MEPs, giving the axonal delay. Here it is therefore reused to position the waveform in the time
          window, and is derived from lambda instead of being hard-coded: axonal_delay = 5.4937*lambda for the
          default tolerance t0_tol = 1e-12, i.e. the smallest delay for which the MUAP has already decayed to
          1e-12 of its peak at t = 0. This crops the leading run of near-zero samples for any lambda.
          See muap_axonal_delay().
    - The MUAP amplitude distribution should reproduce that of the anatomically derived MUAPs (https://pubmed.ncbi.nlm.nih.gov/31465437/).
      For this purpose:
        - A1 is set to 3.75457942e-06/2 
        - Am = 5.28518724e+02 * A1
"""

import os
import warnings

import numpy as np
import h5py


def muap_axonal_delay(lam, tol=1e-12):
    """
    Axonal delay [ms] that places the MUAP so it has just decayed to `tol`
    (relative to its own peak) at the start of the time window.

    In gen_muaps the argument of the Hermite-Rodriguez function is
    z = axonal_delay - t_mn - tmuap, and both time vectors start at 0, so at
    the first sample z[0] = axonal_delay exactly. The normalized waveform
    there is f(u)/f_max with

        f(u) = u * exp(-u**2),  u = axonal_delay / lam,
        f_max = exp(-1/2) / sqrt(2)

    so the requirement |MUAP(0)| / peak = tol becomes

        u * exp(-u**2) = tol * exp(-1/2) / sqrt(2),   u > 1/sqrt(2)

    i.e. the root on the decaying branch. Solved by Newton iteration on
    g(u) = ln(u) - u**2 - ln(tol*f_max), which is smooth and converges in a
    few steps. (Equivalently u = sqrt(-W_{-1}(-tol**2/e)/2) via the lower
    branch of the Lambert W function, but that would add a SciPy
    dependency.)

    The result is proportional to lam: axonal_delay = 5.4937*lam for the
    default tol = 1e-12. This replaces a hard-coded delay, so the waveform
    starts right at the beginning of the window for any lam instead of
    leaving a stretch of numerically-zero samples.

    Parameters
    ----------
    lam : float
        Duration factor lambda [ms].
    tol : float, optional
        Target amplitude at t = 0 as a fraction of the MUAP peak.
        Default 1e-12.

    Returns
    -------
    axonal_delay : float
        Axonal delay [ms]; scales linearly with lam.
    """
    if not (0.0 < tol < 1.0):
        raise ValueError(f"tol must lie in (0, 1); got {tol}")

    f_max = np.exp(-0.5) / np.sqrt(2.0)
    target = np.log(tol * f_max)

    # Initial guess: drop the ln(u) term, u ~ sqrt(-target).
    u = np.sqrt(max(-target, 1.0))
    for _ in range(100):
        g = np.log(u) - u * u - target
        dg = 1.0 / u - 2.0 * u
        u_new = u - g / dg
        # Stay on the decaying branch u > 1/sqrt(2).
        if u_new <= 1.0 / np.sqrt(2.0):
            u_new = 0.5 * (u + 1.0 / np.sqrt(2.0))
        if abs(u_new - u) < 1e-14 * max(1.0, u):
            u = u_new
            break
        u = u_new

    return lam * u


def gen_muaps(n_neurons=100, a=3.75457942e-06/2, b=5.28518724e+02, lam=2,
              axonal_delay=None, t0_tol=1e-4):
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
    lam : float, optional
        Time/duration factor lambda [ms] of the Hermite-Rodriguez function
        (default 2, the paper's value).
    axonal_delay : float or None, optional
        Axonal delay [ms]. If None (default) it is derived from lam via
        `muap_axonal_delay(lam, t0_tol)` rather than hard-coded, so the MUAP
        always starts at the very beginning of the time window whatever lam
        is. Pass a number to override.
    t0_tol : float, optional
        Amplitude at t = 0, as a fraction of the MUAP's own peak, used when
        axonal_delay is derived automatically. Default 1e-12: small enough
        that nothing meaningful is clipped off the leading lobe, large
        enough that the start of the window is not filled with
        numerically-zero samples. Larger values crop more aggressively.

    Returns
    -------
    muaps : ndarray, shape (200, N)
    tmuap : ndarray, shape (200,)
    """

    # Time vector
    tmuap = np.arange(0, 20, 0.1)
    t_mn = np.linspace(0, 20, 200)

    # Axonal delay: not hard-coded, but derived from lam. Since the paper's
    # tD_i is arbitrary here (it is fixed later by aligning simulated and
    # recorded MEP peaks), it is used to position the waveform in the
    # window: the smallest delay for which the MUAP has decayed to t0_tol of
    # its peak at t = 0. Too small a delay clips the leading lobe; too large
    # a delay wastes the front of the window on samples that are
    # indistinguishable from zero. Because it scales linearly with lam, the
    # waveform tracks the chosen duration automatically.
    if axonal_delay is None:
        axonal_delay = muap_axonal_delay(lam, t0_tol)

        # z decreases by ~2 steps per sample (t_mn and tmuap both advance),
        # so the waveform occupies t in [0, axonal_delay]. Warn if lam is so
        # large that the trailing lobe runs past the end of the window.
        if axonal_delay > tmuap[-1]:
            warnings.warn(
                f"lam={lam} ms requires an axonal delay of "
                f"{axonal_delay:.2f} ms, so the MUAP is truncated at the end "
                f"of the {tmuap[-1]:.1f} ms window. Reduce lam, raise "
                "t0_tol, or pass axonal_delay explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )

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
    A = A[None, :]          # (1, N)
  
    z = axonal_delay - t_mn - tmuap                 # (200,)
    z = np.vstack((z,) * n_neurons).T               # (200, N)
    normalization_factor = np.max(z * np.exp(-(z / lam) ** 2)) # To ensure that Am = b * A1
    muaps = A * (z * np.exp(-(z / lam) ** 2)) / normalization_factor 

    return muaps, tmuap


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 100

    muaps, tmuap = gen_muaps()

    print("muaps shape:", muaps.shape)  # (200, N)
    print("tmuap shape:", tmuap.shape)  # (200,)
    print("tmuap range:", tmuap[0], "to", tmuap[-1])
    peak = muaps.max(axis=0)
    print("Amplitude range (V):", peak.min(), "to", peak.max())
    print("Am / A1 =", peak.max() / peak.min())

    lam_used = 2
    print("derived axonal delay (ms):", muap_axonal_delay(lam_used, 1e-4))
    print("relative |MUAP| at t=0:", np.abs(muaps[0] / peak).max())
    n_dead = int((np.abs(muaps) / peak < 1e-4).all(axis=1).sum())
    print("samples below 1e-12 of peak:", n_dead, "of", muaps.shape[0])

    plt.figure(figsize=(8, 5))
    for n in range(N):
        plt.plot(tmuap, 1e3 * muaps[:, n], label=f"MU {n + 1}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (mV)")
    plt.title("Simulated MUAP shapes (first-order Hermite-Rodriguez function)")
    #plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()
    print("Saved demo plot to muaps_demo.png")

    # Save muaps and the corresponding time vector (t_muaps) to an HDF5
    out_dir = os.path.join(os.getcwd(), "data_MUAP")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pheno_muaps.h5")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("muaps", data=muaps)
        f.create_dataset("t", data=tmuap)

    print(f"Saved MUAPs to {out_path}")
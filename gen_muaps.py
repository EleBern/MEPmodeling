"""
gen_muaps.py
============

Generate Motor Unit Action Potential (MUAP) waveforms using the first-order
Hermite-Rodriguez function, exactly as described in:

    https://journals.physiology.org/doi/full/10.1152/jn.00626.2017

Model (paper Eqs. 4-5)
-----------------------
Surface EMG is modeled as the linear summation of MUAPs. The shape of the
MUAP produced by the i-th motor unit is given by the first-order
Hermite-Rodriguez function:

    H_i(t) = A_i * (tau_i - t) * exp(-((tau_i - t) / lambda)**2) * u(tau_i - t)   (Eq. 4)

    tau_i = tm_i + tD_i                                                          (Eq. 5, see note below)

where:
    A_i     : scale (amplitude) factor for motor unit i
    lambda  : time (duration) factor, identical for every motor unit (2 ms)
    tm_i    : time of a motoneuron spike
    tD_i    : motoneuron-to-muscle conduction delay ("axonalDelay")
    u(.)    : Heaviside step function (u(x) = 1 for x >= 0, else 0)

Amplitude scaling (A_i)
------------------------
A_i is simulated to vary over a 100-fold range across the motor unit pool:
the first-recruited motor unit has amplitude A_1 = a, and the last-recruited
motor unit has amplitude A_M = 100*a, where a is a scale constant (paper
value: a = 3.5e-4, in mV, calibrated to match experimental mean MEP
amplitude). As in Li et al. (2012), the distribution of A_i across the pool
is exponential, i.e. logarithmically (not linearly) spaced between A_1 and
A_M, so that many motor units have low amplitude and few have high
amplitude:

    A_i = a * 100 ** ((i - 1) / (M - 1)),   i = 1 ... M

Duration (lambda)
------------------
lambda is fixed at 2 ms and identical for every motor unit -- all FDI motor
units are modeled with MUAPs of the same duration (only amplitude differs
across the pool).

Conduction delay (tD_i / axonalDelay)
--------------------------------------
The paper sets this delay to 10 ms (within the biological range for
motoneuron-to-FDI conduction in humans).

Note on Eq. 5 sign convention
------------------------------
The paper text gives "tau_i = tm_i - tD_i". Taken completely literally with
tD_i > 0, this would place the (causal) support of H_i(t) -- which by
construction is only nonzero for t <= tau_i -- entirely *before* the
motoneuron fires, i.e. before tm_i itself, which contradicts the
accompanying description ("... it causes its MUAP to occur after a time
delay tD_i"). To stay faithful to the physiological description (MUAP
occurs at the muscle a time tD_i *after* the motoneuron spike) this
implementation uses:

    tau_i = tm_i + tD_i

which reduces to the same functional form (Eq. 4) and the same causal,
single-lobe Hermite-Rodriguez pulse shape, but keeps the delay acting in the
physiologically-described direction. If a literal sign match to the printed
Eq. 5 is required instead, flip the sign of `axonalDelay` before calling
`gen_muaps`, or edit the `tau` computation below.

Function
--------
gen_muaps(axonalDelay, spike_times, a=3.5e-4, lam=2.0) -> (muaps, tmuap)

Parameters
----------
axonalDelay : float or array_like, shape (N,)
    Motoneuron-to-muscle conduction delay tD_i [ms] (paper value: 10 ms).
    A scalar is broadcast to all N motor units; an array of length N gives
    a per-motor-unit delay.
spike_times : array_like, shape (x, N)
    Motor neuron spike times/train matrix. x is the number of spikes for a
    given neuron, N is the total number of motor units (neurons) in the
    pool. Only the number of columns (N) is used here to determine how
    many MUAP waveforms to generate; the shape itself (Eq. 4) is generated
    for a single nominal spike (tm_i = 0) per motor unit, i.e. this
    function returns each motor unit's MUAP *template* rather than a full
    spike-train-convolved EMG signal.
a : float, optional
    Amplitude scale constant [mV] for the first-recruited motor unit
    (paper value: 3.5e-4). Default 3.5e-4.
lam : float, optional
    Time/duration factor lambda [ms], identical for all motor units
    (paper value: 2.0 ms). Default 2.0.

Returns
-------
muaps : ndarray, shape (200, N)
    MUAP waveform (mV) for each of the N motor units, sampled at the time
    points given by tmuap.
tmuap : ndarray, shape (200,)
    Time vector for the MUAP waveforms, from 0 to 19.9 ms in steps of
    0.1 ms (200 samples).
"""

import os
import numpy as np
import h5py


def gen_muaps(spike_times, axonalDelay=7, a=1.2e-5, lam=2.0):
    """
    Generate MUAP waveforms using the first-order Hermite-Rodriguez
    function described in the paper (Eqs. 4-5). See module docstring for
    full details.

    Parameters
    ----------
    axonalDelay : float or array_like, shape (N,)
        Motoneuron-to-muscle conduction delay tD_i [ms].
    spike_times : array_like, shape (x, N)
        Motor neuron spike times/train matrix; only used to infer N (the
        number of motor units).
    a : float, optional
        Amplitude scale constant [mV] for the first motor unit (default
        3.5e-4, the paper's value).
    lam : float, optional
        Duration factor lambda [ms], identical for all motor units
        (default 2.0, the paper's value).

    Returns
    -------
    muaps : ndarray, shape (200, N)
    tmuap : ndarray, shape (200,)
    """
    spike_times = np.asarray(spike_times)
    if spike_times.ndim != 2:
        raise ValueError(
            "spike_times must have shape (x, N) with x spikes and N neurons; "
            f"got array with shape {spike_times.shape}"
        )
    n_neurons = spike_times.shape[1]

    # Time vector: 0 to 19.9 ms in 0.1 ms steps -> 200 samples.
    tmuap = np.arange(0, 20, 0.1)

    # Broadcast axonalDelay (tD_i) to a (N,) vector.
    axonalDelay = np.asarray(axonalDelay, dtype=float)
    if axonalDelay.ndim == 0:
        tD = np.full(n_neurons, float(axonalDelay))
    else:
        tD = axonalDelay.reshape(-1)
        if tD.shape[0] != n_neurons:
            raise ValueError(
                "axonalDelay must be a scalar or have length N "
                f"(N={n_neurons}); got length {tD.shape[0]}"
            )

    # Amplitude A_i: exponential (log-spaced) distribution over a 100-fold
    # range, from A_1 = a (first-recruited MU) to A_M = 100*a
    # (last-recruited MU). Motor units are assumed ordered by recruitment
    # order (column index 0 = first recruited).
    if n_neurons > 1:
        frac = np.arange(n_neurons) / (n_neurons - 1)
    else:
        frac = np.zeros(1)
    #A = a * 100.0 ** frac  # shape (N,)
    A = a * 5.28518724e+02 ** frac  # shape (N,)

    # Nominal single motoneuron spike (tm_i = 0) for MUAP-template
    # generation; tau_i = tm_i + tD_i = tD_i (see docstring note on the
    # Eq. 5 sign convention).
    tau = tD  # shape (N,)

    # Eq. 4: H_i(t) = A_i * (tau_i - t) * exp(-((tau_i - t)/lambda)^2) * u(tau_i - t)
    t_col = tmuap[:, None]      # (200, 1)
    tau_row = tau[None, :]      # (1, N)
    A_row = A[None, :]          # (1, N)

    z = tau_row - t_col                     # (200, N)
    heaviside = (-z >= 0).astype(float)      # u(tau_i - t)
    muaps = A_row * z * np.exp(-(z / lam) ** 2) #* heaviside

    return muaps, tmuap


if __name__ == "__main__":
    # Simple demonstration / sanity check.
    import matplotlib
    #matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = 100
    # Fake spike train matrix: 50 spikes for each of N neurons (values
    # unused by gen_muaps other than for shape).
    fake_spike_times = np.cumsum(np.random.exponential(50, size=(50, N)), axis=0)
    from scipy.stats import gamma as gamma_dist
    scale       = 1.0 / 0.25
    quantiles   = np.linspace(0, 0.99, N)
    spike_times = gamma_dist.ppf(quantiles, a=0.83, scale=scale)

    # Paper value: 10 ms motoneuron-to-muscle conduction delay for all
    # motor units.
    axonalDelay = 100#7.0

    muaps, tmuap = gen_muaps(fake_spike_times, a=3.75457942e-06/2)

    print("muaps shape:", muaps.shape)  # (200, N)
    print("tmuap shape:", tmuap.shape)  # (200,)
    print("tmuap range:", tmuap[0], "to", tmuap[-1])
    print("Amplitude range (V):", muaps.max(axis=0).min(), "to", muaps.max(axis=0).max())

    plt.figure(figsize=(8, 5))
    for n in range(N):
        plt.plot(tmuap, 1e3 * muaps[:, n], label=f"MU {n + 1}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (mV)")
    plt.title("Simulated MUAP shapes (first-order Hermite-Rodriguez function)")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()
    print("Saved demo plot to muaps_demo.png")

    # Save muaps and the corresponding time vector (t_muaps) to an HDF5
    # file, in a "data_MUAP" folder created (if needed) in the directory
    # from which the script is run.
    out_dir = os.path.join(os.getcwd(), "data_MUAP")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pheno_muaps.h5")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("muaps", data=muaps)
        f.create_dataset("t", data=tmuap)

    print(f"Saved MUAPs to {out_path}")
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
    - The MUAP amplitude distribution should reproduce that of the anatomically derived MUAPs (https://pubmed.ncbi.nlm.nih.gov/31465437/).
      For this purpose:
        - A1 is set to 3.75457942e-06/2 
        - Am = 5.28518724e+02 * A1
    - lambda and the axonal delay are no longer fixed:
      both are fitted so that the synthetic MUAP matches the anatomically derived MUAPs as
      closely as possible (see fit_lam()).

Fitting lambda and the axonal delay
-----------------------------------
Within this model lambda sets the MUAP *shape*, the axonal delay sets its *position* in time and
A_i only scales it. The fit is therefore done on the normalised waveform, with one gain (and one
sign, since the polarity of a recorded MUAP is arbitrary) free per anatomical MUAP:

    min_{lambda, delay, c_j}  sum_j || w_j - c_j * g(t; lambda, delay) ||^2

With c_j optimal, the residual of MUAP j is  ||w_j||^2 - <w_j, g>^2 / ||g||^2, i.e. 1 - R_j^2,
and the cost is the mean of 1 - R_j^2 over all anatomical MUAPs.

The delay enters only as a time shift of g, so for a given lambda the cost for *every* candidate
delay is obtained in one pass with a cross-correlation; the delay search is therefore free and
only lambda is scanned (coarse geometric grid + local refinement, no scipy dependency).
"""

import os
import warnings

import h5py
import numpy as np
from h5_helpers import load_h5_to_dict
from load_muap import amplitude_distribution


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
# Sampling interval of the anatomical MUAPs in ms. Only used if muap.h5 does not contain a
# time vector. CHANGE THIS if the anatomical MUAPs are not sampled at 10 kHz.
DT_ANAT_MS = 0.1

# Search range for lambda [ms]. The upper bound keeps the whole biphasic waveform inside the
# 20 ms window (with the double-time z the waveform spans ~6*lambda/2 ms around its centre).
LAM_BOUNDS = (0.05, 5.0)

# Search range for the axonal delay [ms]. None = automatic: 0 .. fold*20, where fold
# (= z_time_fold()) is how much delay corresponds to 1 ms of waveform shift (~2 with the
# double-time z, 1 otherwise). The fit anyway only accepts delays that keep the whole
# waveform inside the 0-20 ms window.
DELAY_BOUNDS = None


# --------------------------------------------------------------------------------------
# MUAP generation
# --------------------------------------------------------------------------------------
def _time_axes():
    """The two 0-20 ms axes used by gen_muaps()."""
    tmuap = np.arange(0, 20, 0.1)          # (200,)
    t_mn = np.linspace(0, 20, 200)         # (200,)
    return tmuap, t_mn


def _z_of_t(axonal_delay):
    """Argument z of the Hermite-Rodriguez function, on the tmuap axis."""
    tmuap, t_mn = _time_axes()
    return axonal_delay - t_mn - tmuap


def z_time_fold():
    """
    -dz/dt on the tmuap axis, i.e. how many ms of delay correspond to 1 ms of waveform shift.
    ~2.005 with the double-time z, 1 with the single-time z.
    """
    tmuap, _ = _time_axes()
    z = _z_of_t(0.0)
    return -(z[-1] - z[0]) / (tmuap[-1] - tmuap[0])


def gen_muaps(n_neurons=100, a=1.04495487e-02/2, b=1.59019399e+02, lam=2.0,
              axonal_delay=None):
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
        Time constant lambda [ms] of the Hermite-Rodriguez function
        (default 2.0, the paper value).
    axonal_delay : float, optional
        Axonal delay [ms]. Default None sets axonal_dely = 3*lam + 1. 
        Use fit_lam() to obtain the lambda / delay pair that
        best matches the anatomically derived MUAPs.

    Returns
    -------
    muaps : ndarray, shape (200, N)
    tmuap : ndarray, shape (200,)
    """
    # Paper parameters
    if axonal_delay is None:
        axonal_delay = 3 * lam + 1

    # Time vector
    tmuap, _ = _time_axes()

    # Amplitude A_i: exponential (log-spaced) distribution over a n_neurons-fold
    # range, from A_1 = a (first-recruited MU) to A_M = b*a
    # (last-recruited MU). Motor units are assumed ordered by recruitment
    # order (column index 0 = first recruited).
    if n_neurons > 1:
        frac = np.arange(n_neurons) / (n_neurons - 1)
    else:
        frac = np.zeros(1)
    A = a * b ** frac
    A = A[None, :]          # (1, N)

    # Eq. 4: H_i(t) = A_i * (tau_i - t) * exp(-((tau_i - t)/lambda)^2) * u(tau_i - t)
    z = _z_of_t(axonal_delay)                     # (200,)
    if z.max() < lam / np.sqrt(2):
        warnings.warn(
            f"axonal_delay={axonal_delay:.3f} ms is too short for lam={lam:.3f} ms: the "
            "positive peak of the waveform falls before t = 0, so the peak amplitudes are "
            "no longer equal to A_i (the A_M/A_1 ratio is preserved).", RuntimeWarning)
    z = np.tile(z[:, None], (1, n_neurons))       # (200, N)

    normalization_factor = np.max(z * np.exp(-(z / lam) ** 2))  # To ensure that Am = b * A1
    muaps = A * (z * np.exp(-(z / lam) ** 2)) / normalization_factor

    return muaps, tmuap


def muap_shape(lam, dt=0.1, axonal_delay=None, rel_tol=1e-4):
    """
    Normalised (unit peak) MUAP waveform as generated by gen_muaps(), resampled at `dt` ms
    and trimmed to its support. This is the template used to fit lambda and the delay.

    Returns
    -------
    t : ndarray
        Absolute time [ms] (same clock as tmuap) of the trimmed template.
    g : ndarray
        Template, peak amplitude 1.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m, t = gen_muaps(n_neurons=1, a=1.0, b=1.0, lam=lam, axonal_delay=axonal_delay)
    g = m[:, 0]

    t_new = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    g = np.interp(t_new, t, g)

    peak = np.max(np.abs(g))
    if peak <= 0 or not np.isfinite(peak):
        return t_new, np.zeros_like(t_new)

    keep = np.abs(g) >= rel_tol * peak
    i0 = int(np.argmax(keep))
    i1 = len(keep) - int(np.argmax(keep[::-1]))
    return t_new[i0:i1], g[i0:i1] / peak


# --------------------------------------------------------------------------------------
# Fitting lambda and the axonal delay to the anatomical MUAPs
# --------------------------------------------------------------------------------------
def _prepare_anatomical(anatomical_muaps, t_anat=None, dt_anat=DT_ANAT_MS, dt_fit=None):
    """
    Bring the anatomical MUAPs into the form used by the fit: samples along axis 0,
    baseline removed, resampled onto a common (possibly finer) grid so that the delay /
    shift search has sub-sample resolution.

    Returns (W, t_fit, dt_fit) with W of shape (n_samples, n_valid_muaps).
    """
    W = np.atleast_2d(np.asarray(anatomical_muaps, dtype=float))
    if W.shape[0] == 1:                     # a single MUAP given as a row
        W = W.T

    t_src = None
    if t_anat is not None:
        t_anat = np.asarray(t_anat, dtype=float).ravel()
        if W.shape[0] != t_anat.size and W.shape[1] == t_anat.size:
            W = W.T                         # samples were along axis 1
        if t_anat.size == W.shape[0] and t_anat.size > 1:
            t_src = t_anat
            dt_anat = float(np.median(np.diff(t_anat)))

    if not np.isfinite(dt_anat) or dt_anat <= 0:
        dt_anat = DT_ANAT_MS
    if t_src is None:
        t_src = np.arange(W.shape[0]) * dt_anat     # assumes the MUAPs start at t = 0
    if dt_fit is None:
        dt_fit = min(dt_anat, 0.05)

    t_fit = np.arange(t_src[0], t_src[-1] + 0.5 * dt_fit, dt_fit)

    cols = []
    for j in range(W.shape[1]):
        w = W[:, j]
        if not np.all(np.isfinite(w)):
            continue
        w = w - np.median(w)                # remove baseline offset
        if np.allclose(w, 0):
            continue
        cols.append(np.interp(t_fit, t_src, w))

    if not cols:
        raise ValueError("No usable anatomical MUAPs (all empty or non-finite).")

    return np.column_stack(cols), t_fit, dt_fit


def _free_shift_residuals(lam, W, dt, ww):
    """
    1 - R^2 per anatomical MUAP when each MUAP gets its own time shift (delay not fitted).
    """
    n_samples, n_muaps = W.shape
    _, g = muap_shape(lam, dt=dt)
    gg = float(g @ g)
    if g.size < 3 or gg <= 0 or g.size > 4 * n_samples:
        return np.ones(n_muaps)             # degenerate template -> worst possible fit

    r = np.empty(n_muaps)
    for j in range(n_muaps):
        cc = np.correlate(W[:, j], g, mode="full")   # <w, g_shift> for every shift
        r[j] = max(0.0, 1.0 - float(np.max(cc ** 2)) / (gg * ww[j]))
    return r


def _shared_delay_scan(lam, W, t_fit, dt, ww, delay_bounds):
    """
    For a fixed lambda, evaluate the cost (mean 1 - R^2, one gain and sign free per MUAP,
    but a single common delay) for every candidate delay in one cross-correlation pass.

    Returns (delays, cost) sorted by increasing delay, restricted to delay_bounds and to
    shifts for which the template lies inside the anatomical window.
    """
    n_samples, n_muaps = W.shape
    d_ref = 3 * lam + 1                     # delay used to build the reference template
    tg, g = muap_shape(lam, dt=dt, axonal_delay=d_ref)
    m = g.size
    gg = float(g @ g)
    if m < 3 or gg <= 0:
        return np.array([d_ref]), np.array([1.0])

    # S[k] = mean_j <w_j, g_shift>^2 / ||w_j||^2  for shift = k - (m - 1) samples
    S = np.zeros(n_samples + m - 1)
    for j in range(n_muaps):
        S += np.correlate(W[:, j], g, mode="full") ** 2 / ww[j]
    cost = np.maximum(0.0, 1.0 - S / (n_muaps * gg))

    shifts = np.arange(S.size) - (m - 1)
    # time shift applied to the reference template, and the delay it corresponds to
    tau = (t_fit[0] + shifts * dt) - tg[0]
    delays = d_ref + z_time_fold() * tau

    inside = (shifts >= 0) & (shifts + m <= n_samples)   # template fully in the window
    ok = (delays >= delay_bounds[0]) & (delays <= delay_bounds[1])
    mask = inside & ok
    if not mask.any():                      # window too short for this lambda: allow overhang
        mask = ok
    if not mask.any():
        return np.array([d_ref]), np.array([1.0])

    delays, cost = delays[mask], cost[mask]
    order = np.argsort(delays)
    return delays[order], cost[order]


def fit_lam(anatomical_muaps, t_anat=None, dt_anat=DT_ANAT_MS, lam_bounds=LAM_BOUNDS,
            fit_delay=True, delay_bounds=DELAY_BOUNDS, n_grid=60, n_refine=4, verbose=True):
    """
    Find the lambda (and, if fit_delay, the axonal delay) that make the synthetic MUAP fit
    the anatomical MUAPs best. See the module docstring for the cost function and for what
    fitting the delay assumes about the anatomical MUAPs.

    Parameters
    ----------
    anatomical_muaps : array_like, shape (n_samples, n_muaps)
        Anatomically derived MUAPs.
    t_anat : array_like, optional
        Time vector of the anatomical MUAPs [ms]. If given it sets both the sampling interval
        and the time origin used for the delay.
    dt_anat : float, optional
        Sampling interval [ms] used when `t_anat` is not given (default DT_ANAT_MS); the MUAPs
        are then assumed to start at t = 0.
    lam_bounds : (float, float), optional
        Search interval for lambda [ms].
    delay_bounds : (float, float) or None, optional
        Search interval for the axonal delay [ms]; None = automatic (see DELAY_BOUNDS).
    fit_delay : bool, optional
        True (default): fit a single common axonal delay together with lambda.
        False: give every MUAP its own time shift (only lambda is fitted) and return the
        paper delay 3*lambda + 1.
    n_grid, n_refine : int, optional
        Coarse grid size and number of local refinement passes for lambda.

    Returns
    -------
    lam_opt : float
        Best-fitting lambda [ms].
    delay_opt : float
        Best-fitting axonal delay [ms] (3*lam_opt + 1 if fit_delay is False).
    info : dict
        'r2' (mean R^2), 'r2_per_muap', 'cost', 'lam_grid', 'cost_grid',
        'delay_grid'/'delay_cost' (cost vs delay at lam_opt, fit_delay only),
        'delay_at_lam' (best delay for every lambda of the coarse grid),
        'W', 't_fit', 'dt_fit', 'n_muaps', 'fit_delay'.
    """
    if delay_bounds is None:
        delay_bounds = (0.0, z_time_fold() * 20.0)

    W, t_fit, dt_fit = _prepare_anatomical(anatomical_muaps, t_anat=t_anat, dt_anat=dt_anat)
    ww = np.einsum('ij,ij->j', W, W)         # ||w_j||^2

    def cost_of_lam(lam):
        """(cost, best delay) for one lambda."""
        if fit_delay:
            delays, cost = _shared_delay_scan(lam, W, t_fit, dt_fit, ww, delay_bounds)
            k = int(np.argmin(cost))
            return float(cost[k]), float(delays[k])
        return float(_free_shift_residuals(lam, W, dt_fit, ww).mean()), 3 * lam + 1

    lo, hi = float(lam_bounds[0]), float(lam_bounds[1])
    lam_grid = np.geomspace(lo, hi, int(n_grid))
    evaluated = [cost_of_lam(l) for l in lam_grid]
    cost_grid = np.array([c for c, _ in evaluated])
    delay_at_lam = np.array([d for _, d in evaluated])

    lams, costs, delays_best = lam_grid, cost_grid, delay_at_lam
    for _ in range(int(n_refine)):
        k = int(np.argmin(costs))
        left = lams[max(k - 1, 0)]
        right = lams[min(k + 1, lams.size - 1)]
        if right <= left:
            break
        lams = np.linspace(left, right, 21)
        ev = [cost_of_lam(l) for l in lams]
        costs = np.array([c for c, _ in ev])
        delays_best = np.array([d for _, d in ev])

    k = int(np.argmin(costs))
    lam_opt = float(lams[k])
    delay_opt = float(delays_best[k])

    # Goodness of fit per MUAP at the optimum
    if fit_delay:
        tg, g = muap_shape(lam_opt, dt=dt_fit, axonal_delay=delay_opt)
        gh = np.zeros(W.shape[0])
        i0 = int(round((tg[0] - t_fit[0]) / dt_fit))
        s0, s1 = max(i0, 0), min(i0 + g.size, W.shape[0])
        gh[s0:s1] = g[s0 - i0:s1 - i0]
        gg = float(gh @ gh)
        proj = W.T @ gh
        r2_per_muap = np.clip(proj ** 2 / (gg * ww), 0.0, 1.0) if gg > 0 else np.zeros(ww.size)
        delay_grid, delay_cost = _shared_delay_scan(lam_opt, W, t_fit, dt_fit, ww, delay_bounds)
    else:
        r2_per_muap = 1.0 - _free_shift_residuals(lam_opt, W, dt_fit, ww)
        delay_grid = delay_cost = None

    info = {
        "lam": lam_opt,
        "axonal_delay": delay_opt,
        "r2": float(np.mean(r2_per_muap)),
        "r2_per_muap": r2_per_muap,
        "cost": float(costs[k]),
        "lam_grid": lam_grid,
        "cost_grid": cost_grid,
        "delay_at_lam": delay_at_lam,
        "delay_grid": delay_grid,
        "delay_cost": delay_cost,
        "W": W,
        "t_fit": t_fit,
        "dt_fit": dt_fit,
        "n_muaps": W.shape[1],
        "fit_delay": bool(fit_delay),
    }

    if verbose:
        print(f"Fitted lambda      : {lam_opt:.4f} ms  (paper value: 2; "
              f"search range {lo}-{hi} ms)")
        if fit_delay:
            print(f"Fitted axonal delay: {delay_opt:.4f} ms  (paper relation 3*lam+1 = "
                  f"{3 * lam_opt + 1:.4f} ms; search range "
                  f"{delay_bounds[0]:.1f}-{delay_bounds[1]:.1f} ms)")
        else:
            print(f"Axonal delay       : {delay_opt:.4f} ms  (not fitted, = 3*lam+1; "
                  "each MUAP was fitted with its own time shift)")
        print(f"Mean R^2 over {W.shape[1]} anatomical MUAPs: {info['r2']:.4f} "
              f"(min {r2_per_muap.min():.4f}, max {r2_per_muap.max():.4f})")
        print(f"Fit sampling interval: {dt_fit} ms, anatomical time origin "
              f"{t_fit[0]:.3f} ms")
        if lam_opt < lo * 1.01 or lam_opt > hi * 0.99:
            print("WARNING: lambda sits at the edge of lam_bounds - widen the range "
                  "(but check that the waveform still fits in the 20 ms window).")
        if fit_delay and (delay_opt < delay_bounds[0] * 1.01
                          or delay_opt > delay_bounds[1] * 0.99):
            print("WARNING: the delay sits at the edge of delay_bounds - widen the range, "
                  "or check that the anatomical MUAPs really share the tmuap time origin.")

    return lam_opt, delay_opt, info


def align_to_template(w, g):
    """
    Best shift and gain (sign included) of template `g` for the single MUAP `w`.
    Used for the free-shift diagnostics. Returns (g_fitted, shift_samples, gain).
    """
    cc = np.correlate(w, g, mode="full")
    k = int(np.argmax(cc ** 2))
    shift = k - (g.size - 1)
    gain = float(cc[k]) / float(g @ g)

    out = np.zeros_like(w)
    i0 = max(shift, 0)
    i1 = min(shift + g.size, w.size)
    if i1 > i0:
        out[i0:i1] = gain * g[i0 - shift:i1 - shift]
    return out, shift, gain


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    FIT_DELAY = True         # False -> free per-MUAP shift, delay = 3*lam + 1 (see docstring)

    # Import anatomical MUAPs and calculate the amplitude distribution
    root    = os.path.dirname(os.path.realpath(__file__))
    h5_path = os.path.join(root, "data_MUAP", "muap.h5")

    anatomical_muaps = None
    t_anat = None
    popt = None
    lam_opt = 2.0                      
    delay_opt = 3 * lam_opt + 1
    info = None

    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            tmp = load_h5_to_dict(f)

        anatomical_muaps = tmp["muaps"]   # [n_samples x n_muaps]
        for key in ("t", "time", "tmuap", "t_muap", "t_muaps"):
            if key in tmp:
                t_anat = np.asarray(tmp[key]).ravel()
                break
        if t_anat is None:
            print(f"No time vector in {h5_path}: assuming dt = {DT_ANAT_MS} ms and t0 = 0 "
                  f"for the anatomical MUAPs.")

    # Calculate and fit MUAPs amplitude distribution
        popt = amplitude_distribution(anatomical_muaps)

    # Fit lambda (and the axonal delay) to the anatomical MUAPs
        lam_opt, delay_opt, info = fit_lam(anatomical_muaps, t_anat=t_anat,
                                          dt_anat=DT_ANAT_MS, fit_delay=FIT_DELAY)
    else:
        print(f"{h5_path} not found: using the default values lambda = {lam_opt}, "
              f"delay = {delay_opt} and the default amplitude parameters.")

    # Generate the synthetic MUAPs with that amplitude distribution, lambda and delay
    N = 100

    if popt is not None:
        muaps, tmuap = gen_muaps(n_neurons=N, a=popt[0]/2, b=popt[1],
                                 lam=lam_opt, axonal_delay=delay_opt)
    else:
        muaps, tmuap = gen_muaps(n_neurons=N, lam=lam_opt, axonal_delay=delay_opt)

    print("lambda used:", lam_opt)
    print("axonal delay used:", delay_opt)
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
    plt.title(f"Simulated MUAP shapes (Hermite-Rodriguez, lambda = {lam_opt:.3f} ms, "
              f"delay = {delay_opt:.3f} ms)")
    plt.xlim([0, 20])
    # plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()

    # Diagnostics of the fit
    if info is not None:
        W, t_fit, dt_fit = info["W"], info["t_fit"], info["dt_fit"]

        fig, ax = plt.subplots(1, 4, figsize=(18, 4))

        ax[0].plot(info["lam_grid"], 1.0 - info["cost_grid"], '-')
        ax[0].axvline(lam_opt, color='r', ls='--', label=f"{lam_opt:.3f} ms")
        ax[0].set_xscale('log')
        ax[0].set_xlabel("lambda (ms)")
        ax[0].set_ylabel("mean $R^2$")
        ax[0].set_title("Fit vs lambda (best delay each)")
        ax[0].legend()

        if info["delay_grid"] is not None:
            ax[1].plot(info["delay_grid"], 1.0 - info["delay_cost"], '-')
            ax[1].axvline(delay_opt, color='r', ls='--', label=f"{delay_opt:.3f} ms")
            ax[1].set_xlabel("axonal delay (ms)")
            ax[1].set_ylabel("mean $R^2$")
            ax[1].set_title(f"Fit vs delay (lambda = {lam_opt:.3f} ms)")
            ax[1].legend()
        else:
            ax[1].axis('off')

        # Overlay of the normalised anatomical MUAPs and the fitted waveform
        tg, g = muap_shape(lam_opt, dt=dt_fit, axonal_delay=delay_opt)
        for j in range(W.shape[1]):
            w = W[:, j]
            peak = np.max(np.abs(w))
            if info["fit_delay"]:
                # absolute time, sign chosen by the fit
                i0 = int(round((tg[0] - t_fit[0]) / dt_fit))
                gh = np.zeros(W.shape[0])
                s0, s1 = max(i0, 0), min(i0 + g.size, W.shape[0])
                gh[s0:s1] = g[s0 - i0:s1 - i0]
                s = np.sign(w @ gh) or 1.0
                ax[2].plot(t_fit, s * w / peak, color='0.7', lw=0.8)
            else:
                _, shift, gain = align_to_template(w, g)
                s = np.sign(gain) or 1.0
                ax[2].plot(t_fit - shift * dt_fit + tg[0], s * w / peak,
                           color='0.7', lw=0.8)
        ax[2].plot(tg, g, 'r', lw=2, label="fitted MUAP")
        ax[2].set_xlabel("Time (ms)")
        ax[2].set_ylabel("Normalised amplitude")
        ax[2].set_title("Anatomical MUAPs vs fit")
        ax[2].legend()

        ax[3].hist(info["r2_per_muap"], bins=20)
        ax[3].set_xlabel("$R^2$ per MUAP")
        ax[3].set_ylabel("count")
        ax[3].set_title(f"mean $R^2$ = {info['r2']:.3f}")

        fig.tight_layout()

    plt.show()

    # Save muaps and the corresponding time vector (t_muaps) to an HDF5
    out_dir = os.path.join(os.getcwd(), "data_MUAP")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pheno_muaps.h5")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("muaps", data=muaps)
        f.create_dataset("t", data=tmuap)
        f.create_dataset("lam", data=float(lam_opt))
        f.create_dataset("axonal_delay", data=float(delay_opt))
        if info is not None:
            f.create_dataset("fit_r2", data=float(info["r2"]))

    print(f"Saved MUAPs to {out_path}")
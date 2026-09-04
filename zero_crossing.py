"""Zero-crossing detection for MUAP waveforms stored in muap.h5.

The file holds `muaps` with shape (n_samples, n_waveforms) and a shared
time vector `t` of length n_samples.

Main entry points
-----------------
zero_crossings(t, y)          -> crossings of one waveform
zero_crossings_all(t, muaps)  -> list of crossings, one per waveform
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class ZeroCrossings(NamedTuple):
    """Result of a zero-crossing search on one waveform.

    times    : crossing times, in the units of `t` (sub-sample resolution)
    indices  : index of the last sample *before* each crossing
    rising   : True where the waveform goes negative -> positive
    """

    times: np.ndarray
    indices: np.ndarray
    rising: np.ndarray

    def __len__(self) -> int:
        return self.times.size


def _locate(t: np.ndarray, y: np.ndarray, a: int, b: int) -> tuple[float, int]:
    """Locate the crossing inside [a, b], where y[a] and y[b] have opposite signs.

    Returns (crossing time, index of the sample before the crossing).
    """
    j = np.arange(a, b)
    # Sample pairs that straddle zero. Strict '<' skips pairs touching zero,
    # which are handled by the zero-run fallback below.
    brackets = j[y[j] * y[j + 1] < 0]

    if brackets.size:
        # With several brackets (noise wiggling about zero) take the steepest
        # one: it is the transition that actually carries the waveform across.
        k = int(brackets[np.argmax(np.abs(y[brackets + 1] - y[brackets]))])
        t0, t1, y0, y1 = t[k], t[k + 1], y[k], y[k + 1]
        return float(t0 - y0 * (t1 - t0) / (y1 - y0)), k

    # No strict bracket => the sign flip happens across a run of exact zeros.
    # Report the centre of that run.
    zeros = np.arange(a, b + 1)[y[a : b + 1] == 0]
    return float(0.5 * (t[zeros[0]] + t[zeros[-1]])), int(zeros[0])


def zero_crossings(t, y, rel_threshold: float = 0.0) -> ZeroCrossings:
    """Find the zero crossings of a single waveform.

    Parameters
    ----------
    t : (n,) array
        Time vector.
    y : (n,) array
        Waveform samples.
    rel_threshold : float, default 0.0
        Noise gate, as a fraction of the waveform's peak absolute amplitude.
        A sign change is only reported once the waveform reaches
        ``rel_threshold * max(|y|)`` on both sides of it, so excursions that
        never leave the noise band are ignored. 0.0 reports every genuine
        sign change; 0.02-0.05 is a reasonable range for noisy recordings.

    Returns
    -------
    ZeroCrossings
        Named tuple of `times`, `indices` and `rising`. Empty for a flat or
        all-zero waveform.

    Notes
    -----
    Samples that are exactly zero are not treated as crossings in their own
    right: the zero padding at the ends of these MUAPs would otherwise
    register as spurious events. A zero-valued sample only counts when the
    waveform genuinely changes sign across it.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t = t[1000:]
    y = y[1000:]

    if t.shape != y.shape:
        raise ValueError(f"t and y must have the same shape, got {t.shape} and {y.shape}")
    if not 0.0 <= rel_threshold < 1.0:
        raise ValueError("rel_threshold must be in [0, 1)")

    empty = ZeroCrossings(
        np.empty(0, dtype=float), np.empty(0, dtype=int), np.empty(0, dtype=bool)
    )
    if y.size < 2:
        return empty

    peak = np.max(np.abs(y))
    if peak == 0:                       # dead / all-zero channel
        return empty

    # Samples carrying a real excursion. With rel_threshold = 0 this simply
    # drops the exact zeros.
    sig = np.flatnonzero(np.abs(y) > rel_threshold * peak)
    if sig.size < 2:
        return empty

    signs = np.sign(y[sig])
    flips = np.flatnonzero(signs[:-1] != signs[1:])
    if flips.size == 0:
        return empty

    times, indices = zip(*(_locate(t, y, a, b) for a, b in zip(sig[flips], sig[flips + 1])))

    return ZeroCrossings(
        times=np.asarray(times, dtype=float),
        indices=np.asarray(indices, dtype=int),
        rising=signs[flips] < 0,
    )


def zero_crossings_all(t, muaps, time_axis: int = 0, rel_threshold: float = 0.0):
    """Apply `zero_crossings` to every waveform in a 2-D array.

    `muaps` is (n_samples, n_waveforms) by default; pass ``time_axis=1`` if
    the waveforms are stored as rows. Returns a list of ZeroCrossings, one
    per waveform, in column (or row) order.
    """
    muaps = np.asarray(muaps, dtype=float)
    if muaps.ndim != 2:
        raise ValueError(f"expected a 2-D array, got {muaps.ndim}-D")
    if time_axis == 1:
        muaps = muaps.T
    elif time_axis != 0:
        raise ValueError("time_axis must be 0 or 1")

    return [zero_crossings(t, col, rel_threshold) for col in muaps.T]


def load_muaps(path: str = "muap.h5"):
    """Read (t, muaps) from the HDF5 file."""
    import h5py

    with h5py.File(path, "r") as f:
        return f["t"][:], f["muaps"][:]


if __name__ == "__main__":
    import sys
    import os
    import h5py
    from h5_helpers import load_h5_to_dict

    path = sys.argv[1] if len(sys.argv) > 1 else "data_MUAP/Dist1_Monopolar_Rest_NormalCV_New.hdf5"
    #t, muaps = load_muaps(path)
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            tmp = load_h5_to_dict(f)

    muaps = tmp["MUAPShapes"] 
    t = np.linspace(0, 20, 20001)
    cond = np.sum(np.abs(muaps), axis=1) != 0
    first_nonzero = np.argmax(cond)      # 0-based index of first True (assumes at least one nonzero row)
    idx = first_nonzero - 1              # keep one zero row before the signal starts, like MATLAB's idx = find(...)-1

    muaps = -muaps[idx:, :]              # flipped (sign)
    t = t[idx:]
    t = t - t.min()
    crossings = zero_crossings_all(t, muaps)

    for i, zc in enumerate(crossings):
        if len(zc) == 0:
            print(f"waveform {i:3d}: no zero crossings (flat waveform)")
        else:
            marks = ", ".join(
                f"{x:.3f} ({'up' if r else 'down'})" for x, r in zip(zc.times, zc.rising)
            )
            print(f"waveform {i:3d}: {len(zc)} crossing(s) at t = {marks}")

    counts = np.array([len(zc) for zc in crossings])
    print(f"\n{len(counts)} waveforms, {counts.sum()} crossings total")
    for n in np.unique(counts):
        print(f"  {int((counts == n).sum()):3d} waveform(s) with {n} crossing(s)")

    import matplotlib.pyplot as plt
    for i in [42,52,72]:#range(np.shape(muaps)[1]):
        plt.figure()
        plt.plot(t, muaps[:,i])
        plt.plot(np.linspace(0,20,50), np.zeros(50), "k--", linewidth=0.5)
        if len(crossings[i].times) == 1:
            plt.plot(crossings[i].times, 0, "r*")
        elif len(crossings[i].times) > 1:
            j = np.argwhere(crossings[i].rising == False)
            plt.plot(crossings[i].times[j], 0, "r*")
        plt.title(i)
        plt.show()
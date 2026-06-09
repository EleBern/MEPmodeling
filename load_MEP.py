import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from load_h5 import load_h5_to_dict

HDF5_PATH = "/home/bernasconi/Documents/Programma/MEP_Python/MEPmodeling/data_diLazarro/DiLazarro_di_wave_data.hdf5"

# Mapping from subject number to year (and subject index within that year)
SUBJ_TO_YEAR = {
    1: ('2004', 1),   # subject # 1 of 2004
    2: ('2004', 2),   # subject # 2 of 2004
    3: ('2007', None),
    4: ('2013', None),
    5: ('2020', None),
}


def _find_subject_group(thr_type_group, year, subj_num_in_year):
    """
    Return the HDF5 group for the requested subject inside a threshold_type
    group.  Subject folders end in '# N' for 2004 data; for other years there
    is only one subject folder so we take the first one.
    """
    keys = list(thr_type_group.keys())

    if subj_num_in_year is not None:
        # 2004: match the folder whose name ends in '# N'
        pattern = re.compile(r'#\s*' + str(subj_num_in_year) + r'\s*$')
        matches = [k for k in keys if pattern.search(k)]
        if not matches:
            raise KeyError(
                f"No subject folder ending in '# {subj_num_in_year}' found. "
                f"Available: {keys}"
            )
        return thr_type_group[matches[0]]
    else:
        # Single subject years: take the first (and only) folder
        if not keys:
            raise KeyError("No subject folders found.")
        return thr_type_group[keys[0]]


def _first_nan_sample(mep):
    """
    For each trial, return the index of the first NaN sample across all
    intensity levels, or n_time if the trial is clean.

    Parameters
    ----------
    mep : (n_intensities, n_time, n_trials)

    Returns
    -------
    first_nan : (n_trials,)  int array; value is n_time when no NaN exists
    """
    n_time, n_trials = mep.shape[1], mep.shape[2]
    # True where any intensity has a NaN: (n_time, n_trials)
    any_nan = np.any(np.isnan(mep), axis=0)
    # For each trial, argmax on the time axis finds the first True;
    # when there is no NaN, argmax returns 0 – fix with np.where.
    has_nan   = np.any(any_nan, axis=0)               # (n_trials,)
    first_nan = np.argmax(any_nan, axis=0)            # (n_trials,)
    first_nan = np.where(has_nan, first_nan, n_time)  # n_time means "clean"
    return first_nan


def _is_trailing_nan(mep, trial_idx):
    """
    Return True if all NaN values in a trial are contiguous and end-aligned
    (i.e. no NaN appears before the first NaN sample).

    Parameters
    ----------
    mep        : (n_intensities, n_time, n_trials)
    trial_idx  : int
    """
    trial   = mep[:, :, trial_idx]             # (n_intensities, n_time)
    any_nan = np.any(np.isnan(trial), axis=0)  # (n_time,)
    first   = int(np.argmax(any_nan))
    if first == 0:
        return False # Discard trial rather than cropping entire signal if the whole trial is NaNs
    # Trailing: everything from `first` onward is NaN, nothing before it is
    return bool(np.all(any_nan[first:]) and not np.any(any_nan[:first]))


def _clean_nan_trials(mep, times, subj):
    """
    Handle NaN-contaminated trials in mep.

    Strategy
    --------
    * Trials whose NaNs are strictly trailing (all NaNs form a contiguous
      block at the end of the time axis) are kept; mep and times are
      cropped to the shortest valid length across all such trials.
    * Trials with scattered NaNs are discarded entirely.

    A trial is evaluated across *all* intensity levels jointly so that the
    array stays rectangular.

    Parameters
    ----------
    mep   : (n_intensities, n_time, n_trials)
    times : (n_time,)
    subj  : str  – used only for warning messages

    Returns
    -------
    mep_clean   : (n_intensities, n_time_out, n_trials_out)
    times_clean : (n_time_out,)
    """
    n_trials  = mep.shape[2]
    first_nan = _first_nan_sample(mep)        # (n_trials,) index or n_time

    nan_trials = np.where(first_nan < mep.shape[1])[0]  # trials with any NaN

    if len(nan_trials) == 0:
        return mep, times                     # nothing to do

    trailing_mask = np.array(
        [_is_trailing_nan(mep, k) for k in nan_trials]
    )
    trailing_trials  = nan_trials[ trailing_mask]
    scattered_trials = nan_trials[~trailing_mask]

    # -- 1. discard scattered-NaN trials ------------------------------------
    keep = np.ones(n_trials, dtype=bool)
    if len(scattered_trials) > 0:
        keep[scattered_trials] = False
        print(
            f"[load_MEP] WARNING – Subject {subj!r}: discarding "
            f"{len(scattered_trials)}/{n_trials} trial(s) with scattered NaN values."
        )

    mep       = mep[:, :, keep]
    first_nan = first_nan[keep]

    # -- 2. crop trailing-NaN trials ----------------------------------------
    # Crop to the shortest valid length among trials that had trailing NaNs
    # (clean trials have first_nan == n_time, so they never drive the crop)
    trailing_first_nan = first_nan[first_nan < mep.shape[1]]
    if len(trailing_first_nan) > 0:
        crop_end = int(np.min(trailing_first_nan))
        print(
            f"[load_MEP] WARNING – Subject {subj!r}: {len(trailing_trials)} trial(s) "
            f"have trailing NaNs; cropping time axis to {crop_end} samples "
            f"(was {mep.shape[1]}). "
            f"Time vector (before cropping) now ends at {np.round(times[crop_end], 2)} ms."
        )
        mep   = mep[:, :crop_end, :]
        times = times[:crop_end]

    return mep, times


def load_MEP(subj, iidx=None, tcrop=[20, 50], plotOn=1):
    """
    Load MEP data from DiLazzaro_di_wave_data_by_year_merged.hdf5.

    Parameters
    ----------
    subj : str
        Subject identifier in the format '#PA' or '#LM', where # is 1–5.
        Examples: '1PA', '3LM', '5PA'
    iidx : array-like or None
        Indices of intensity levels to keep. None = all.
    tcrop : [float, float]
        Time window (ms) to crop for the averaged signal (relative to TMS).
    plotOn : bool / int
        Whether to produce plots.

    Returns
    -------
    y        : (n_intensities, n_time) mean MEP traces, baseline-corrected
    t        : (n_time,)              cropped time vector (ms, relative to TMS)
    mep      : (n_intensities, n_time, n_trials)  all trials, baseline-corrected
    intensities : (n_intensities,)   stimulation intensities
    times    : (n_time_full,)         full time vector (ms, relative to TMS stimulus per row)
    yall     : (n_intensities, n_time_cropped, n_trials)  cropped trials
    """
    # ------------------------------------------------------------------
    # Parse subject string, e.g. '3LM' -> subj_num=3, orientation='LM'
    # ------------------------------------------------------------------
    m = re.fullmatch(r'(\d+)(PA|LM)', str(subj).strip(), re.IGNORECASE)
    if not m:
        raise ValueError(
            f"subj must be in the format '#PA' or '#LM' (e.g. '3LM'). Got: {subj!r}"
        )
    subj_num      = int(m.group(1))
    orientation   = m.group(2).upper()

    if subj_num not in SUBJ_TO_YEAR:
        raise ValueError(f"subj number must be 1–5. Got: {subj_num}")

    year, subj_num_in_year = SUBJ_TO_YEAR[subj_num]

    # ------------------------------------------------------------------
    # Load from HDF5
    # ------------------------------------------------------------------
    with h5py.File(HDF5_PATH, 'r') as f:
        thr_type_group = f[year][orientation]['RMT']
        subj_group     = _find_subject_group(thr_type_group, year, subj_num_in_year)

    
        intensities = np.array(subj_group['intensities']).flatten()
        mep         = np.array(subj_group['EMG']['mep'])   # (n_intensities, n_time, n_trials)
        time_raw    = np.array(subj_group['time'])          # (n_thr, n_time)
        tidx = np.where((time_raw[0] >= tcrop[0]) & (time_raw[0] < tcrop[1]))[0]
        times_cropped = time_raw[:, tidx]   # (n_thr, n_crop)
        t = times_cropped[iidx, :]
        times = time_raw[iidx, :]
        # Check all rows are equal after shifting and cropping
        if not np.all(np.isclose(times_cropped, times_cropped[0, :], atol=1e-9)):
            diffs = [
                f"  row {i}: first={times_cropped[i, 0]:.4f} last={times_cropped[i, -1]:.4f}"
                for i in range(times_cropped.shape[0])
            ]
            msg = (
                f"Time arrays differ across threshold values after TSTIM correction "
                f"and cropping to [{tcrop[0]}, {tcrop[1]}) ms:\n"
                + "\n".join(diffs)
            )
            raise ValueError(msg)

        times = time_raw[0, :]    
        t = times_cropped[0, :]

    # ------------------------------------------------------------------
    # Select intensities
    # ------------------------------------------------------------------
    if iidx is None:
        iidx = np.arange(len(intensities))

    mep         = mep[iidx, :, :]
    intensities = intensities[iidx]

    # ------------------------------------------------------------------
    # Clean NaN trials (crop trailing NaNs; discard scattered-NaN trials)
    # ------------------------------------------------------------------
    mep, times = _clean_nan_trials(mep, times, subj)
 
    tidx = np.where((times >= tcrop[0]) & (times < tcrop[1]))[0] # Recompute in case mep and times were cropped due to trailing NaNs
    t = times[tidx]

    yall = mep[:, tidx, :]
    y    = np.mean(yall, axis=2)

    # ------------------------------------------------------------------
    # Remove baseline [-20, -10] ms
    # ------------------------------------------------------------------
    baseline_idx = np.where((times >= -20) & (times < -10))[0]
    baseline = np.mean(
        np.mean(mep[:, baseline_idx, :], axis=2),
        axis=1
    )  

    y   = y   - baseline[:, np.newaxis]
    mep = mep - baseline[:, np.newaxis, np.newaxis]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    if plotOn:
        n = len(intensities)
        ncols = min(5, n)
        nrows = int(np.ceil(n / ncols))

        # Figure 1: all trials
        plt.figure()
        for i in range(n):
            plt.subplot(nrows, ncols, i + 1)
            plt.plot(times, mep[i, :, :], 'c', linewidth=0.8)
            plt.grid(True)
            plt.plot(times, np.mean(mep[i, :, :], axis=1), 'k', linewidth=1.5)
            plt.xlim([20, 50])
            plt.ylim([-2, 5])
            plt.title(f"{intensities[i]}% MSO")
            plt.xlabel("msec")

        # Figure 2: averaged signals
        plt.figure()
        for i in range(n):
            plt.plot(t, y[i, :], linewidth=1.5)
        plt.grid(True)
        plt.title(f"Subject {subj}")
        plt.xlabel("Time (msec)")
        plt.ylabel("Amplitude (mV)")
        plt.xlim([20, 50])
        plt.legend([f"{v}% MSO" for v in intensities])
        plt.show()

    return y, t, mep, intensities, times, yall
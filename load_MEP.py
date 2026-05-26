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
    6: ('2050', None),
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


def load_MEP(subj, iidx=None, tcrop=[20, 50], plotOn=1):
    """
    Load MEP data from DiLazzaro_di_wave_data_by_year_merged.hdf5.

    Parameters
    ----------
    subj : str
        Subject identifier in the format '#PA' or '#LM', where # is 1–6.
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
        raise ValueError(f"subj number must be 1–6. Got: {subj_num}")

    year, subj_num_in_year = SUBJ_TO_YEAR[subj_num]

    # ------------------------------------------------------------------
    # Load from HDF5
    # ------------------------------------------------------------------
    with h5py.File(HDF5_PATH, 'r') as f:
        thr_type_group = f[year][orientation]['RMT']
        subj_group     = _find_subject_group(thr_type_group, year, subj_num_in_year)

        if subj_num == 6:
            intensities = np.array([140])
            mep = np.array(subj_group['EMG']['signal_full'])   # (n_intensities, n_time, n_trials)
            times    = np.array(subj_group['time']).flatten()          # (n_thr, n_time)
            tidx = np.where((times >= tcrop[0]) & (times < tcrop[1]))[0]
            t = times[tidx]   # (n_thr, n_crop)
        else:
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
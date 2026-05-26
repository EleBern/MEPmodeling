"""
Generate two output files from DiLazarro_di_wave_data_by_year_merged.hdf5:

  DiLazzaro_di_wave_data_up.hdf5
  DiLazzaro_di_wave_data_down.hdf5

Both files receive identical processing. For each subject:

  1. Read sample_frequency from EMG/ and time from the subject folder.
  2. If fs differs across thresholds → copy everything unchanged (ignored).
  3. If fs is uniform but time vectors differ across thresholds →
       resample every threshold's mep (at the same fs) so that all
       thresholds share the time vector of the first threshold.
       Update time to store the new (uniform) time vectors.
       Only mep and time are modified; all other data is copied verbatim.
  4. If both fs and time are uniform → copy everything unchanged.

Usage:
    python resample_di_wave.py
"""

import h5py
import numpy as np
import os

SRC_PATH  = "/home/bernasconi/Documents/Programma/MEP_Python/MEPmodeling/data_diLazarro/DiLazarro_di_wave_data_by_year_merged.hdf5"
UP_PATH   = "/home/bernasconi/Documents/Programma/MEP_Python/MEPmodeling/data_diLazarro/DiLazzaro_di_wave_data_up.hdf5"
DOWN_PATH = "/home/bernasconi/Documents/Programma/MEP_Python/MEPmodeling/data_diLazarro/DiLazzaro_di_wave_data_down.hdf5"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def copy_item(src_item, dst_parent, name, skip_datasets=None):
    """
    Recursively copy an HDF5 item (group or dataset) from src_item into
    dst_parent under the given name, preserving all attributes.

    skip_datasets : set of dataset names to omit (used to drop signal_mean).
    """
    skip_datasets = skip_datasets or set()

    if isinstance(src_item, h5py.Dataset):
        if name in skip_datasets:
            print(f"      [drop] {name}")
            return
        if name in dst_parent:
            return
        ds = dst_parent.create_dataset(
            name,
            data=src_item[()],
            compression=src_item.compression,
            compression_opts=src_item.compression_opts,
        )
        for k, v in src_item.attrs.items():
            ds.attrs[k] = v

    elif isinstance(src_item, h5py.Group):
        grp = dst_parent.require_group(name)
        for k, v in src_item.attrs.items():
            grp.attrs[k] = v
        for child_name in src_item.keys():
            copy_item(src_item[child_name], grp, child_name, skip_datasets)


def copy_group_contents(src_grp, dst_grp, skip_datasets=None):
    """Copy all children of src_grp into dst_grp."""
    skip_datasets = skip_datasets or set()
    for k, v in src_grp.attrs.items():
        dst_grp.attrs[k] = v
    for name in src_grp.keys():
        copy_item(src_grp[name], dst_grp, name, skip_datasets)


def fs_uniform(fs_array):
    """
    Return (is_uniform: bool, unique_values: sorted list) for a flat or
    2-D array of sampling frequencies.
    """
    vals = np.unique(np.round(fs_array.flatten(), decimals=4))
    return len(vals) == 1, vals.tolist()


def time_uniform(time_array):
    """
    Return True if all rows of time_array (n_thr, n_time) are identical.
    """
    if time_array.ndim == 1:
        return True
    return bool(np.all(np.isclose(time_array, time_array[0], atol=1e-9)))


def resample_to_target_time(mep_i, time_i, time_target, fs):
    """
    Resample mep_i (n_time_i, n_trials) from time_i onto time_target
    at the same sampling frequency fs, using scipy.signal.resample_poly.

    Strategy: both time vectors are sampled at the same fs, so they differ
    only in start/end time or length.  We find the overlap, extract the
    matching segment from mep_i, and interpolate onto time_target using
    numpy interp (sample-accurate for same-fs data).

    Returns resampled mep of shape (len(time_target), n_trials).
    """
    n_trials = mep_i.shape[1]
    out = np.full((len(time_target), n_trials), np.nan)
    for trial in range(n_trials):
        out[:, trial] = np.interp(time_target, time_i, mep_i[:, trial],
                                   left=np.nan, right=np.nan)
    return out


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def process_subject(src_subj_grp, dst_subj_grp):
    """
    Copy the subject group into dst_subj_grp.

    Cases:
      - fs not uniform across thresholds  → copy everything unchanged (ignored).
      - fs uniform, time uniform          → copy everything unchanged.
      - fs uniform, time NOT uniform      → resample every threshold's mep
                                            onto the first threshold's time
                                            vector; update time; copy all
                                            other data verbatim.
    """
    emg_grp  = src_subj_grp['EMG']
    fs_raw   = np.array(emg_grp['sample_frequency'])  # (n_thr,) or (n_thr, n_trials)
    mep_raw  = np.array(emg_grp['mep'])               # (n_thr, n_time, n_trials)
    time_raw = np.array(src_subj_grp['time'])         # (n_thr, n_time)

    uniform_fs, unique_fs = fs_uniform(fs_raw)
    uniform_time          = time_uniform(time_raw)

    # ------------------------------------------------------------------ #
    # Decide action                                                        #
    # ------------------------------------------------------------------ #
    if not uniform_fs:
        print(f"        fs not uniform ({unique_fs} Hz) — copying unchanged")
        copy_group_contents(src_subj_grp, dst_subj_grp)
        return

    if uniform_time:
        print(f"        fs and time uniform — copying unchanged")
        copy_group_contents(src_subj_grp, dst_subj_grp)
        return

    fs = float(unique_fs[0])
    print(f"        fs uniform ({fs} Hz), time differs — resampling to row-0 time")

    # ------------------------------------------------------------------ #
    # Copy everything except mep and time (written explicitly below)      #
    # ------------------------------------------------------------------ #
    dst_emg_grp = dst_subj_grp.require_group('EMG')
    for k, v in emg_grp.attrs.items():
        dst_emg_grp.attrs[k] = v
    for name in emg_grp.keys():
        if name not in {'mep', 'signal_mean'}:
            copy_item(emg_grp[name], dst_emg_grp, name)

    for k, v in src_subj_grp.attrs.items():
        dst_subj_grp.attrs[k] = v
    for name in src_subj_grp.keys():
        if name in ('EMG', 'time'):
            continue
        copy_item(src_subj_grp[name], dst_subj_grp, name)

    # ------------------------------------------------------------------ #
    # Resample every threshold row onto the first threshold's time vector #
    # ------------------------------------------------------------------ #
    time_target = time_raw[0]           # 1-D reference: (n_time_target,)
    n_thr       = mep_raw.shape[0]
    n_trials    = mep_raw.shape[2]
    n_t_target  = len(time_target)

    mep_out  = np.full((n_thr, n_t_target, n_trials), np.nan)
    time_out = np.tile(time_target, (n_thr, 1))   # (n_thr, n_time_target)

    for i in range(n_thr):
        time_i = time_raw[i] if time_raw.ndim == 2 else time_raw
        mep_i  = mep_raw[i]       # (n_time_i, n_trials)

        if np.allclose(time_i, time_target, atol=1e-9):
            mep_out[i] = mep_i    # already aligned, no interpolation needed
        else:
            mep_out[i] = resample_to_target_time(mep_i, time_i, time_target, fs)
            print(f"          thr row {i}: time {time_i[0]:.3f}–{time_i[-1]:.3f} ms "
                  f"-> {time_target[0]:.3f}–{time_target[-1]:.3f} ms")

    dst_emg_grp.create_dataset('mep',  data=mep_out)
    dst_subj_grp.create_dataset('time', data=time_out)

    print(f"        mep  {mep_raw.shape} -> {mep_out.shape}")
    print(f"        time {time_raw.shape} -> {time_out.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    for out_path in (UP_PATH, DOWN_PATH):
        if os.path.exists(out_path):
            os.remove(out_path)
            print(f"Removed existing: {out_path}")

    with h5py.File(SRC_PATH, 'r') as src, \
         h5py.File(UP_PATH,   'w') as up, \
         h5py.File(DOWN_PATH, 'w') as down:

        for year in sorted(src.keys()):
            for orientation in sorted(src[year].keys()):
                for thr_type in sorted(src[year][orientation].keys()):
                    thr_grp = src[year][orientation][thr_type]

                    for subj_key in sorted(thr_grp.keys()):
                        subj_grp = thr_grp[subj_key]

                        # Skip if this is not a subject group (no EMG subfolder)
                        if 'EMG' not in subj_grp or 'mep' not in subj_grp['EMG']:
                            # Copy verbatim into both files
                            for dst in (up, down):
                                dst_grp = dst.require_group(
                                    f"{year}/{orientation}/{thr_type}")
                                copy_item(subj_grp, dst_grp, subj_key)
                            continue

                        path = f"{year}/{orientation}/{thr_type}/{subj_key}"
                        print(f"\n  Processing: {path}")

                        # Both output files receive identical processing
                        for dst_file in (up, down):
                            dst_grp  = dst_file.require_group(
                                f"{year}/{orientation}/{thr_type}")
                            dst_subj = dst_grp.require_group(subj_key)
                            process_subject(subj_grp, dst_subj)

    print("\nDone.")
    print(f"  UP   -> {UP_PATH}")
    print(f"  DOWN -> {DOWN_PATH}")


if __name__ == '__main__':
    run()
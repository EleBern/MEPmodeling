"""
Merge threshold_value subgroups inside each threshold_type group,
preserving the subject/channel/EMG layout.

For every (year, orientation, threshold_type) combination (year 2004 skipped):
  - All datasets inside each channel group are merged across threshold values,
    stacked in ascending threshold order.
  - The EMG subfolder (at the same level as the channel folders) is merged
    the same way.
  - Stacking is shape-aware:
      scalar            ->  1-D array of length n_thr
      (n,)              ->  (n_thr, n)
      (1, entries, trials) ->  (n_thr, entries, trials)   [axis-0 squeeze then stack]
      any other shape   ->  np.stack along a new axis 0

Output structure (mirrors source, threshold_value folders removed):
    /{year}/{orientation}/{threshold_type}/{subject}/
        intensities               (n_thr,)  -- shared across all channels
        {channel}/
            <every dataset>       shape: stacked as above
        EMG/
            mep                   (n_thr, ...)  -- renamed from signal_full
            <other datasets>      shape: stacked as above

Usage:
    python merge_thresholds.py
"""

import re
import h5py
import numpy as np
import os
from collections import defaultdict

SRC_PATH = "/home/bernasconi/Documents/Programma/MEP_Python/MEPmodeling/data_diLazarro/DiLazarro_di_wave_data_by_year.hdf5"
DST_PATH = "/home/bernasconi/Documents/Programma/MEP_Python/MEPmodeling/data_diLazarro/DiLazarro_di_wave_data_by_year_merged.hdf5"

SKIP_YEARS = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def subject_number(folder_name):
    """
    Extract the trailing '# N' subject number from a folder name.
    Returns int N, or None if absent.
    """
    m = re.search(r'#\s*(\d+)\s*$', folder_name)
    return int(m.group(1)) if m else None


def grouping_key(thr_type, path_parts):
    """
    Hashable key: (thr_type, subject_id, *rest)
    subject_id is the '# N' integer when present, else the raw folder name.
    This keeps RMT/MSO, and subject #1/#2, strictly separate.
    """
    if not path_parts:
        return (thr_type,)
    subject_part = path_parts[0]
    num  = subject_number(subject_part)
    rest = tuple(path_parts[1:])
    if num is not None:
        return (thr_type, num) + rest
    return (thr_type, subject_part) + rest


def pad_to_shape(arr, target_shape):
    """
    Pad arr with NaN (float) or 0 (int/bool) along every axis so that its
    shape matches target_shape.  target_shape must be >= arr.shape on every axis.
    """
    arr = np.asarray(arr)
    if arr.shape == target_shape:
        return arr
    # Cast integer/bool arrays to float so NaN can be used as a sentinel
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        arr = arr.astype(float)
    fill = np.nan if np.issubdtype(arr.dtype, np.floating) else 0
    pad_width = [(0, t - s) for s, t in zip(arr.shape, target_shape)]
    return np.pad(arr, pad_width, mode='constant', constant_values=fill)


def stack_arrays(arrays):
    """
    Stack a list of arrays (one per threshold value) in a shape-aware way.
    When arrays differ in shape across threshold values (e.g. different number
    of trials), all arrays are padded with NaN/0 to the maximum size along
    each axis before stacking.

    scalar                ->  1-D array  (n_thr,)
    (n,)                  ->  (n_thr, n)
    (1, entries, trials)  ->  (n_thr, entries, trials)  [singleton squeezed first]
    any other shape       ->  (n_thr, *max_shape)
    """
    converted = [np.asarray(a) for a in arrays]
    a0 = converted[0]

    # scalar
    if a0.ndim == 0:
        return np.array([a.item() for a in converted])

    # (1, entries, trials) — squeeze the leading singleton before stacking
    if a0.ndim == 3 and a0.shape[0] == 1:
        converted = [a[0] for a in converted]   # each -> (entries, trials)

    # Compute the maximum shape across all arrays (element-wise)
    max_shape = tuple(
        max(a.shape[i] for a in converted) for i in range(converted[0].ndim)
    )

    # Pad each array to max_shape, then stack along a new axis 0
    padded = [pad_to_shape(a, max_shape) for a in converted]
    return np.stack(padded, axis=0)


def collect_leaf_groups(thr_val_group):
    """
    Walk the subject/channel tree and return the set of all relative paths
    to 'leaf' groups — groups that contain at least one dataset directly
    (i.e. channel groups and EMG groups, not the subject group itself).

    Also returns subject-level groups that hold only datasets (flat layout).

    Returns
    -------
    list of str  — relative paths from thr_val_group to each leaf group
    """
    leaf_paths = []

    def _visit(name, obj):
        if not isinstance(obj, h5py.Group):
            return
        # A group is a leaf if it directly contains at least one dataset
        has_direct_dataset = any(
            isinstance(obj[k], h5py.Dataset) for k in obj.keys()
        )
        if has_direct_dataset:
            leaf_paths.append(name)

    thr_val_group.visititems(_visit)
    return leaf_paths


def dataset_names_in_group(group):
    """Return names of all datasets directly inside group (not recursive)."""
    return [k for k in group.keys() if isinstance(group[k], h5py.Dataset)]


def copy_group_attrs(src_group, dst_group):
    for k, v in src_group.attrs.items():
        dst_group.attrs[k] = v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def merge_thresholds(src_path, dst_path):
    if os.path.exists(dst_path):
        os.remove(dst_path)
        print(f"Removed existing file: {dst_path}")

    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:

        for year in sorted(src.keys()):
            if year in SKIP_YEARS:
                print(f"[skip] year {year}")
                continue

            for orientation in sorted(src[year].keys()):
                for thr_type in sorted(src[year][orientation].keys()):

                    src_thr_type = src[year][orientation][thr_type]
                    dst_thr_type = dst.require_group(f"{year}/{orientation}/{thr_type}")

                    thr_values_str = sorted(src_thr_type.keys(), key=lambda x: float(x))

                    # ----------------------------------------------------------
                    # Pass 1 — discover every leaf group path and accumulate
                    # per-dataset arrays across threshold values.
                    #
                    # data_store[key][dataset_name] = [(thr_val, array), ...]
                    # key_to_path[key]              = canonical output sub-path
                    # ----------------------------------------------------------
                    # key: grouping_key(thr_type, path_parts_to_leaf)
                    data_store = defaultdict(lambda: defaultdict(list))
                    key_to_path = {}

                    for thr_val_str in thr_values_str:
                        thr_val = float(thr_val_str)
                        thr_val_group = src_thr_type[thr_val_str]

                        leaf_paths = collect_leaf_groups(thr_val_group)

                        if not leaf_paths:
                            print(f"  [warn] no leaf groups under "
                                  f"{year}/{orientation}/{thr_type}/{thr_val_str}")
                            continue

                        for leaf_path in leaf_paths:
                            leaf_group = thr_val_group[leaf_path]
                            parts = leaf_path.split('/')
                            key   = grouping_key(thr_type, parts)

                            if key not in key_to_path:
                                key_to_path[key] = leaf_path

                            for ds_name in dataset_names_in_group(leaf_group):
                                arr = leaf_group[ds_name][()]
                                data_store[key][ds_name].append((thr_val, arr))

                    # ----------------------------------------------------------
                    # Pass 2 — stack and write
                    # ----------------------------------------------------------
                    # Track which subject groups have already received
                    # their 'intensities' dataset (written once per subject,
                    # not once per channel/EMG leaf).
                    written_intensities = set()

                    for key, ds_dict in sorted(data_store.items()):
                        out_path   = key_to_path[key]
                        dst_group  = dst_thr_type.require_group(out_path)

                        # Collect threshold values from whichever dataset is
                        # present (all share the same threshold list)
                        any_entries = next(iter(ds_dict.values()))
                        thr_vals    = np.array([e[0] for e in
                                                sorted(any_entries, key=lambda x: x[0])])

                        # 'intensities' lives one level up (subject folder),
                        # shared across all channel/EMG leaves of that subject.
                        subject_path = out_path.split('/')[0]
                        if subject_path not in written_intensities:
                            dst_subject = dst_thr_type.require_group(subject_path)
                            dst_subject.create_dataset('intensities', data=thr_vals)
                            written_intensities.add(subject_path)
                            print(f"  {year}/{orientation}/{thr_type}/{subject_path}"
                                  f"/intensities  ->  {thr_vals}")

                        # Determine whether this leaf is inside an EMG folder
                        path_parts = out_path.split('/')
                        in_emg = 'EMG' in path_parts

                        for ds_name, entries in ds_dict.items():
                            entries.sort(key=lambda x: x[0])   # ascending threshold
                            arrays  = [e[1] for e in entries]
                            stacked = stack_arrays(arrays)

                            # Rename signal_full -> mep inside EMG groups
                            out_name = 'mep' if (in_emg and ds_name == 'signal_full') else ds_name

                            dst_group.create_dataset(out_name, data=stacked)

                            print(f"  {year}/{orientation}/{thr_type}/{out_path}"
                                  f"/{out_name}  ->  {stacked.shape}  "
                                  f"thr={thr_vals}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def print_structure(hdf5_path, max_depth=7):
    def _print(name, obj):
        depth = name.count('/')
        if depth >= max_depth:
            return
        indent = '  ' * depth
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name.split('/')[-1]}  "
                  f"[dataset] shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"{indent}{name.split('/')[-1]}  [group]")

    print(f"\nStructure of: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        f.visititems(_print)


if __name__ == '__main__':
    print(f"Source     : {SRC_PATH}")
    print(f"Destination: {DST_PATH}\n")

    merge_thresholds(SRC_PATH, DST_PATH)

    print_structure(DST_PATH, max_depth=7)
    print("\nDone.")
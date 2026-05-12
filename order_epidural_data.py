"""
Merge threshold_value subgroups inside each threshold_type group,
preserving the subject/channel layout, keeping multiple subjects separate,
and keeping RMT and MSO signals strictly separate.

Starting from the restructured file:
    /{year}/{orientation}/{threshold_type}/{threshold_value}/{subject}/[{channel}/]signal_short

For every (year, orientation, threshold_type) combination:
  1. Collect all threshold_value folders, sorted numerically (ascending).
  2. Walk the subject -> channel hierarchy to discover all unique sub-paths
     that lead to a signal_short dataset.
  3. Group signals by (threshold_type, subject_number, *channel) so that:
       - RMT and MSO are never merged together.
       - Subject # 1 and subject # 2 are never merged together.
       - Different channels of the same subject are never merged together.
  4. Write the merged data preserving the original layout:
       /{year}/{orientation}/{threshold_type}/{subject}/[{channel}/]signal_short
       /{year}/{orientation}/{threshold_type}/{subject}/[{channel}/]threshold_values

Usage:
    python merge_thresholds.py
"""

import re
import h5py
import numpy as np
import os
from collections import defaultdict

SRC_PATH = "/home/bernasconi/Documents/Programma/MEP_Python/MEPmodeling/data_diLazzaro/DiLazarro_di_wave_data_by_year.hdf5"
DST_PATH = "/home/bernasconi/Documents/Programma/MEP_Python/MEPmodeling/data_diLazzaro/DiLazarro_di_wave_data_by_year_merged.hdf5"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_signal_paths(thr_val_group):
    """
    Return a list of relative paths from thr_val_group to every dataset
    named 'signal_short'.

    Examples
    --------
    ['subject_0/signal_short']
    ['subject_0/channel_0/signal_short', 'subject_0/channel_1/signal_short']
    ['PA 120 RMT # 1/signal_short', 'PA 120 RMT # 2/signal_short']
    """
    paths = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name.split('/')[-1] == 'signal_short':
            paths.append(name)

    thr_val_group.visititems(_visit)
    return paths


def subject_number(folder_name):
    """
    Extract the trailing subject number from a folder name ending in '# N'.
    Returns the integer N, or None if no such suffix is present.

    Examples
    --------
    'PA 120 RMT # 1'  ->  1
    'LM 140 MSO # 2'  ->  2
    'subject_0'       ->  None
    """
    m = re.search(r'#\s*(\d+)\s*$', folder_name)
    return int(m.group(1)) if m else None


def grouping_key(thr_type, path_parts):
    """
    Build a hashable grouping key from the threshold type and the path
    components between the threshold_value folder and 'signal_short'
    (e.g. thr_type='RMT', path_parts=['PA 120 RMT # 2', 'channel_0']).

    The key encodes:
      - thr_type   : 'RMT' or 'MSO' — ensures RMT and MSO are never mixed.
      - subject id : the '# N' number from the subject folder name, so that
                     subject # 1 and # 2 are always kept separate even if the
                     rest of their folder name varies across threshold values.
      - channel(s) : all remaining path components verbatim.

    Returns a tuple, e.g. ('RMT', 1, 'channel_0').
    """
    if not path_parts:
        return (thr_type,)

    subject_part = path_parts[0]
    num = subject_number(subject_part)
    rest = tuple(path_parts[1:])

    if num is not None:
        # Key on (thr_type, subject_number, *channels)
        return (thr_type, num) + rest
    else:
        # No '# N' suffix — use the folder name verbatim
        return (thr_type, subject_part) + rest


def copy_non_signal_datasets(src_group, dst_group):
    """
    Recursively copy every dataset that is NOT named signal_short or
    threshold_values (e.g. time_short), preserving attributes.
    Key-existence checks prevent errors when called repeatedly for the
    same dst_group across multiple threshold values.
    """
    skip = {'signal_short', 'threshold_values'}
    for attr_key, attr_val in src_group.attrs.items():
        dst_group.attrs[attr_key] = attr_val

    for key in src_group.keys():
        item = src_group[key]
        if isinstance(item, h5py.Dataset):
            if key not in skip and key not in dst_group:
                dst_group.create_dataset(
                    key,
                    data=item[()],
                    compression=item.compression,
                    compression_opts=item.compression_opts,
                )
                for attr_key, attr_val in item.attrs.items():
                    dst_group[key].attrs[attr_key] = attr_val
        elif isinstance(item, h5py.Group):
            child = dst_group.require_group(key)
            copy_non_signal_datasets(item, child)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def merge_thresholds(src_path, dst_path):
    if os.path.exists(dst_path):
        os.remove(dst_path)
        print(f"Removed existing file: {dst_path}")

    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:

        for year in sorted(src.keys()):
            if year == '2004':
                print(f"  [skip] year 2004 — skipping as requested")
                continue
            for orientation in sorted(src[year].keys()):
                for thr_type in sorted(src[year][orientation].keys()):  # 'MSO', 'RMT'

                    src_thr_type = src[year][orientation][thr_type]
                    dst_thr_type = dst.require_group(f"{year}/{orientation}/{thr_type}")

                    # Sort threshold value folders numerically
                    thr_values_str = sorted(src_thr_type.keys(), key=lambda x: float(x))

                    # ----------------------------------------------------------
                    # Pass 1: collect signals keyed by
                    #         (thr_type, subject_number, *channel_parts)
                    #
                    # channel_data : key -> [(thr_val, signal_array), ...]
                    # key_to_path  : key -> canonical output sub-path string
                    #                (taken from the first threshold_value that
                    #                 sees this key; used to name the dst group)
                    # ----------------------------------------------------------
                    channel_data = defaultdict(list)
                    key_to_path  = {}

                    for thr_val_str in thr_values_str:
                        thr_val = float(thr_val_str)
                        thr_val_group = src_thr_type[thr_val_str]

                        signal_paths = find_signal_paths(thr_val_group)
                        if not signal_paths:
                            print(f"  [warn] no signal_short under "
                                  f"{year}/{orientation}/{thr_type}/{thr_val_str} — skipping")
                            continue

                        for sig_path in signal_paths:
                            # e.g. 'PA 120 RMT # 2/channel_0/signal_short'
                            parts     = sig_path.split('/')
                            sub_parts = parts[:-1]          # drop 'signal_short'
                            key       = grouping_key(thr_type, sub_parts)
                            signal_array = thr_val_group[sig_path][()]
                            channel_data[key].append((thr_val, signal_array))

                            # Record the canonical output path on first encounter
                            if key not in key_to_path:
                                key_to_path[key] = '/'.join(sub_parts)

                        # Copy auxiliary datasets (e.g. time_short) once per
                        # threshold value; key-existence checks prevent duplicates
                        copy_non_signal_datasets(thr_val_group, dst_thr_type)

                    # ----------------------------------------------------------
                    # Pass 2: stack signals and write per (subject, channel)
                    # ----------------------------------------------------------
                    for key, entries in sorted(channel_data.items()):
                        entries.sort(key=lambda x: x[0])        # ascending threshold
                        thr_vals = np.array([e[0] for e in entries])
                        signals  = np.vstack([e[1] for e in entries])  # (n_thr, 100)

                        out_path   = key_to_path[key]
                        dst_parent = dst_thr_type.require_group(out_path)
                        dst_parent.create_dataset("signal_short",     data=signals)
                        dst_parent.create_dataset("threshold_values", data=thr_vals)

                        print(f"  {year}/{orientation}/{thr_type}/{out_path}  ->  "
                              f"signal_short {signals.shape},  "
                              f"threshold_values {thr_vals}")


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def print_structure(hdf5_path, max_depth=6):
    """Print the hierarchy of an HDF5 file for quick verification."""
    def _print(name, obj):
        depth = name.count('/')
        if depth < max_depth:
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

    print_structure(DST_PATH, max_depth=6)
    print("\nDone.")
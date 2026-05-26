"""
Copy the subject from year 2020, threshold 140, threshold_type RMT
from DiLazarro_di_wave_data_by_year.hdf5 into
DiLazarro_di_wave_data_by_year_merged.hdf5 under year 2050.

The source group is copied verbatim (all datasets, all sub-groups,
all attributes) for every orientation (PA, LM) that has a 2020/RMT/140
entry in the source file.

The destination path mirrors the merged file's convention:
    /2050/{orientation}/RMT/{subject_key}/...

Usage:
    python add_2020_140_as_2050.py
"""

import h5py
import numpy as np

SRC_PATH = "/home/bernasconi/Documents/Programma/MEP_Python/MEPmodeling/data_diLazarro/DiLazarro_di_wave_data_by_year.hdf5"
DST_PATH = "/home/bernasconi/Documents/Programma/MEP_Python/MEPmodeling/data_diLazarro/DiLazarro_di_wave_data_by_year_merged.hdf5"

SRC_YEAR      = '2020'
SRC_THR_TYPE  = 'RMT'
SRC_THR_VAL   = '140'
DST_YEAR      = '2050'


def copy_item(src, dst):
    """
    Recursively copy all datasets and groups from src into dst,
    preserving attributes at every level.
    """
    # Copy group-level attributes
    for k, v in src.attrs.items():
        dst.attrs[k] = v

    for key in src.keys():
        item = src[key]
        if isinstance(item, h5py.Dataset):
            if key in dst:
                print(f"    [skip] dataset '{key}' already exists in destination")
                continue
            dst.create_dataset(
                key,
                data=item[()],
                compression=item.compression,
                compression_opts=item.compression_opts,
            )
            for k, v in item.attrs.items():
                dst[key].attrs[k] = v
            print(f"    copied dataset '{key}'  shape={item.shape}  dtype={item.dtype}")

        elif isinstance(item, h5py.Group):
            child_dst = dst.require_group(key)
            copy_item(item, child_dst)


def print_structure(h5file, root_path, max_depth=6):
    """Print the subtree rooted at root_path for verification."""
    if root_path not in h5file:
        print(f"  (path not found: {root_path})")
        return

    def _print(name, obj):
        depth = name.count('/')
        if depth >= max_depth:
            return
        indent = '  ' * depth
        if isinstance(obj, h5py.Dataset):
            print(f"  {indent}{name.split('/')[-1]}  "
                  f"[dataset] shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"  {indent}{name.split('/')[-1]}  [group]")

    h5file[root_path].visititems(_print)


def run():
    with h5py.File(SRC_PATH, 'r') as src, h5py.File(DST_PATH, 'a') as dst:

        if SRC_YEAR not in src:
            raise KeyError(f"Year '{SRC_YEAR}' not found in source file.")

        src_year_grp = src[SRC_YEAR]
        orientations = list(src_year_grp.keys())   # e.g. ['LM', 'PA']

        copied_any = False

        for orientation in orientations:
            # Check this orientation has the requested threshold type and value
            try:
                src_thr_val_grp = src_year_grp[orientation][SRC_THR_TYPE][SRC_THR_VAL]
            except KeyError:
                print(f"  [skip] {SRC_YEAR}/{orientation}/{SRC_THR_TYPE}/{SRC_THR_VAL} "
                      f"not found in source — skipping orientation '{orientation}'")
                continue

            subject_keys = list(src_thr_val_grp.keys())
            if not subject_keys:
                print(f"  [skip] no subjects under "
                      f"{SRC_YEAR}/{orientation}/{SRC_THR_TYPE}/{SRC_THR_VAL}")
                continue

            for subject_key in subject_keys:
                src_subj_grp = src_thr_val_grp[subject_key]
                dst_path = f"{DST_YEAR}/{orientation}/{SRC_THR_TYPE}/{subject_key}"

                if dst_path in dst:
                    print(f"  [skip] destination '{dst_path}' already exists — "
                          f"delete it first if you want to overwrite.")
                    continue

                print(f"\nCopying  {SRC_YEAR}/{orientation}/{SRC_THR_TYPE}"
                      f"/{SRC_THR_VAL}/{subject_key}")
                print(f"     ->  {dst_path}")

                dst_subj_grp = dst.require_group(dst_path)
                copy_item(src_subj_grp, dst_subj_grp)
                copied_any = True

        if not copied_any:
            print("\nNothing was copied.")
            return

        # ----------------------------------------------------------------
        # Verification: print what was written under /2050
        # ----------------------------------------------------------------
        print(f"\n--- Structure written under /{DST_YEAR} ---")
        print_structure(dst, DST_YEAR, max_depth=6)
        print("\nDone.")


if __name__ == '__main__':
    print(f"Source : {SRC_PATH}")
    print(f"Dest   : {DST_PATH}")
    print(f"Copying: {SRC_YEAR}/*/{ SRC_THR_TYPE}/{SRC_THR_VAL}  ->  {DST_YEAR}\n")
    run()
"""
Restructure DiLazarro_di_wave_data.hdf5

Original structure:
    /{orientation}/{threshold_type}/{threshold_value}/{year}/{subject_data}

New structure:
    /{year}/{orientation}/{threshold_type}/{threshold_value}/{subject_data}

Usage:
    python restructure_hdf5.py
"""

import h5py
import os

SRC_PATH = "/home/bernasconi/Documents/Programma/MEP_Python/MEPmodeling/data_diLazzaro/DiLazarro_di_wave_data.hdf5"
DST_PATH = "/home/bernasconi/Documents/Programma/MEP_Python/MEPmodeling/data_diLazzaro/DiLazarro_di_wave_data_by_year.hdf5"


def copy_group(src_group, dst_group):
    """Recursively copy all datasets and attributes from src_group into dst_group."""
    # Copy group-level attributes
    for attr_key, attr_val in src_group.attrs.items():
        dst_group.attrs[attr_key] = attr_val

    for key in src_group.keys():
        item = src_group[key]
        if isinstance(item, h5py.Dataset):
            # Copy dataset with its attributes
            dst_group.create_dataset(key, data=item[()], compression=item.compression,
                                     compression_opts=item.compression_opts)
            for attr_key, attr_val in item.attrs.items():
                dst_group[key].attrs[attr_key] = attr_val
        elif isinstance(item, h5py.Group):
            child = dst_group.require_group(key)
            copy_group(item, child)


def restructure(src_path, dst_path):
    if os.path.exists(dst_path):
        os.remove(dst_path)
        print(f"Removed existing file: {dst_path}")

    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        # Original layout: orientation → threshold_type → threshold_value → year → subject
        for orientation in src.keys():               # e.g. 'LM', 'PA'
            orient_group = src[orientation]
            for threshold_type in orient_group.keys():           # e.g. 'RMT', 'MSO'
                thr_type_group = orient_group[threshold_type]
                for threshold_value in thr_type_group.keys():    # e.g. '100', '120', ...
                    thr_val_group = thr_type_group[threshold_value]
                    for year in thr_val_group.keys():            # e.g. '2007', '2020', ...
                        year_group = thr_val_group[year]

                        # Build the new path: year / orientation / threshold_type / threshold_value
                        new_path = f"{year}/{orientation}/{threshold_type}/{threshold_value}"
                        dst_group = dst.require_group(new_path)

                        # Copy each subject group into the new location
                        for subject_key in year_group.keys():
                            if subject_key not in dst_group:
                                subject_dst = dst_group.require_group(subject_key)
                                copy_group(year_group[subject_key], subject_dst)
                            else:
                                print(f"  [skip] {new_path}/{subject_key} already exists")

                        print(f"  Copied  {orientation}/{threshold_type}/{threshold_value}/{year}"
                              f"  →  {new_path}")

    print(f"\nDone. New file written to:\n  {dst_path}")


def print_structure(hdf5_path, max_depth=5):
    """Print the top levels of an HDF5 file for quick verification."""
    def _print(name, obj):
        depth = name.count('/')
        if depth < max_depth:
            indent = '  ' * depth
            kind = 'dataset' if isinstance(obj, h5py.Dataset) else 'group'
            print(f"{indent}{name.split('/')[-1]}  ({kind})")

    print(f"\nStructure of: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        f.visititems(_print)


if __name__ == '__main__':
    print(f"Source : {SRC_PATH}")
    print(f"Destination: {DST_PATH}\n")

    restructure(SRC_PATH, DST_PATH)

    # Quick sanity check — print first few levels of the new file
    print_structure(DST_PATH, max_depth=4)
"""
Walks through all subfolders of "fitted_results", finds .h5 files, and for
each file checks if a dataset "ref/intensity_idx" exists. If it exists and
does not start from 0, it is updated in place to (intensity_idx - 1).

The file is opened in read/write mode ("r+") and only the target dataset is
modified, so all other contents of the file are preserved exactly, and the
file is saved under the same name/path as the original.
"""

import os
import h5py
import numpy as np

ROOT_DIR = "fitted_results"
DATASET_PATH = "ref/intensity_idx"


def process_file(filepath):
    try:
        with h5py.File(filepath, "r+") as f:
            if DATASET_PATH not in f:
                return

            dset = f[DATASET_PATH]
            data = dset[()]

            # Only handle array-like data
            arr = np.asarray(data)

            if arr.size == 0:
                return

            min_val = arr.min()

            if min_val != 0:
                new_arr = arr - 1

                # Preserve dtype where possible
                if np.issubdtype(dset.dtype, np.integer) and not np.issubdtype(new_arr.dtype, np.integer):
                    new_arr = new_arr.astype(dset.dtype)

                dset[...] = new_arr
                print(f"Updated {DATASET_PATH} in: {filepath} (was min={min_val}, subtracted 1)")
            else:
                print(f"Skipped (already starts at 0): {filepath}")

    except OSError as e:
        print(f"Could not open {filepath}: {e}")


def main():
    if not os.path.isdir(ROOT_DIR):
        print(f"Root directory '{ROOT_DIR}' not found.")
        return

    for dirpath, _, filenames in os.walk(ROOT_DIR):
        for filename in filenames:
            if filename.lower().endswith(".h5"):
                filepath = os.path.join(dirpath, filename)
                process_file(filepath)


if __name__ == "__main__":
    main()
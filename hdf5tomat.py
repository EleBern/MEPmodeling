"""
hdf5_to_mat.py
--------------
Extracts two datasets from DILazarro_di_wave_data.hdf5 and saves them
to a MATLAB .mat file (v5 format, compatible with MATLAB R2006a+):

    HDF5 path                                                   → mat key
    ---------------------------------------------------------   --------
    2020/LM/RMT/Di Lazarro 2020 LM/EMG/mep                    → mep
    2020/LM/RMT/Di Lazarro 2020 LM/time                        → t

Usage
-----
    python hdf5_to_mat.py                        # uses defaults below
    python hdf5_to_mat.py input.hdf5 output.mat  # explicit paths
"""

import sys
import h5py
import scipy.io
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
DEFAULT_HDF5 = "data_diLazarro/DiLazarro_di_wave_data.hdf5"
DEFAULT_MAT  = "S15.mat"

HDF5_DATASETS = {
    # HDF5 dataset path                              : variable name in .mat
    "2013/PA/RMT/Di Lazarro 2013 PA & LM/EMG/mep": "mep",
    "2013/PA/RMT/Di Lazarro 2013 PA & LM/time":    "t",
    "2013/PA/RMT/Di Lazarro 2013 PA & LM/intensities":    "intensities",
}

# ── main ─────────────────────────────────────────────────────────────────────
def convert(hdf5_path: str, mat_path: str) -> None:
    mat_data: dict[str, np.ndarray] = {}

    with h5py.File(hdf5_path, "r") as f:
        for hdf5_key, mat_key in HDF5_DATASETS.items():
            if hdf5_key not in f:
                raise KeyError(
                    f"Dataset not found in HDF5 file: '{hdf5_key}'\n"
                    f"Available top-level keys: {list(f.keys())}"
                )
            data = f[hdf5_key][()]           # load fully into memory
            if mat_key == "t":
                data = data[0]               # keep only the first row (shape: 1000,)
            mat_data[mat_key] = np.array(data)
            print(f"  {hdf5_key!r}  →  '{mat_key}'  shape={data.shape}  dtype={data.dtype}")

    scipy.io.savemat(mat_path, mat_data, do_compression=True)
    print(f"\nSaved → {mat_path}")
    print(f"Variables in .mat: {list(mat_data.keys())}")


if __name__ == "__main__":
    hdf5_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HDF5
    mat_path  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MAT

    print(f"Reading : {hdf5_path}")
    print(f"Writing : {mat_path}\n")
    convert(hdf5_path, mat_path)
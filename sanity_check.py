import os
import h5py
import numpy as np
from h5_helpers import load_h5_to_dict

"""
Check that "muap.h5" contains the same data as "muap1.h5" so I know I processed the MUAPs correctly
"""

root    = os.path.dirname(os.path.realpath(__file__))
h5_path = os.path.join(root, "data_MUAP", "muap.h5")

if os.path.exists(h5_path):
    with h5py.File(h5_path, 'r') as f:
        tmp = load_h5_to_dict(f)

muaps = tmp["muaps"]   # [n_samples x n_muaps]
t     = tmp["t"].T     # [n_samples x 1]

h5_path = os.path.join(root, "data_MUAP", "muap1.h5")

if os.path.exists(h5_path):
    with h5py.File(h5_path, 'r') as f:
        tmp = load_h5_to_dict(f)

muaps1 = tmp["muaps"]   # [n_samples x n_muaps]
t1     = tmp["t"].T     # [n_samples x 1]

if np.allclose(muaps, muaps1):
    print("Muaps all close")

if np.allclose(t, t1):
    print("Time vectors all close")
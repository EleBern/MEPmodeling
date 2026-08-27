import os
import h5py
import numpy as np
from scipy.interpolate import interp1d
from h5_helpers import load_h5_to_dict, _save_dict_to_h5

"""
Preprocess MUAPs of 5 hand models to save the data in hdf5 files instead of in MATLAB format.
Follow the preprocessing done originally (https://github.com/vscChien/MEPmodeling/blob/main/load_muap.m)
"""

root    = os.path.dirname(os.path.realpath(__file__))
for i in range(1, 6):
    h5_path = os.path.join(root, "data_MUAP", "Dist{0}_Monopolar_Rest_NormalCV_New.hdf5".format(i))
    output_path = os.path.join(root, "data_MUAP", "muap{0}.h5".format(i))

    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            tmp = load_h5_to_dict(f)

    muaps = tmp["MUAPShapes"]
    tmuap = np.linspace(0, 20, 20001) # 0~20 ms, dt = 0.001 msec

    # ---cut zeros (around first 2000 points)----
    cond = np.sum(np.abs(muaps), axis=1) != 0
    first_nonzero = np.argmax(cond)      # 0-based index of first True (assumes at least one nonzero row)
    idx = first_nonzero - 1              # keep one zero row before the signal starts, like MATLAB's idx = find(...)-1

    muaps = -muaps[idx:, :]              # flipped (sign)
    tmuap = tmuap[idx:]
    tmuap = tmuap - tmuap.min()

    # ------------------------------------
    # downsample to dt = 0.1 msec
    dt = 0.1  # ms
    t = np.arange(0, 20, dt)
    f = interp1d(tmuap, muaps, axis=0, kind='linear', bounds_error=False, fill_value=0)
    muaps = f(t)

    output_dict = {"muaps": muaps, "t": t}

    with h5py.File(output_path, 'w') as f:
        _save_dict_to_h5(f, output_dict)
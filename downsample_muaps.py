import os
import h5py
import numpy as np
from scipy.interpolate import interp1d
from h5_helpers import load_h5_to_dict

def load_unprocessed_muaps(h5_path="Dist1_Monopolar_Rest_NormalCV_New.hdf5"):
    """
        Funtion that loads and processes anatomical MUAPs. 
    
        Parameters
        ----------
        h5_path   : string, file name of the original hand model
                    
    
        Returns
        -------
        tmuap : np.array 
            time vector of the anatomical MUAPs
        muaps : np.array
            MUAPs waveform
        downsampled_t : np.array
            downsampled time vector with 0.1 ms dt
        downsampled_muaps : np.array
            downsampled MUAPs waveform
    """

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

    # downsample to dt = 0.1 msec
    dt = 0.1  # ms
    downsampled_t = np.arange(0, 20, dt)
    f = interp1d(tmuap, muaps, axis=0, kind='linear', bounds_error=False, fill_value=0)
    downsampled_muaps = f(downsampled_t)

    return tmuap, muaps, downsampled_t, downsampled_muaps
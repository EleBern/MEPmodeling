import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from load_h5 import load_h5_to_dict


def load_muap(plotOn=0):
    """
    Load motor unit action potentials (MUAPs) and optionally add white
    Gaussian noise.

    Parameters
    ----------
    plotOn     : int or bool
                 If truthy, plot all MUAPs (default 0).
    noise_std  : float
                 Noise level as a percentage of the RMS amplitude of the
                 loaded MUAPs.  For example, noise_std=5 adds noise with
                 std = 5 % × RMS(muaps).  Default is 0 (no noise).
    noise_seed : int or None
                 Random seed for reproducibility.  If None (default) the
                 noise is different on every call.

    Returns
    -------
    muaps : np.ndarray  [n_samples x n_muaps]  MUAPs with noise added
    t     : np.ndarray  [n_samples x 1]         time vector (ms)
    """
    root    = os.getcwd()
    h5_path = os.path.join(root, "data_MUAP", "muap.h5")

    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            tmp = load_h5_to_dict(f)

        muaps = tmp["muaps"]   # [n_samples x n_muaps]
        t     = tmp["t"].T     # [n_samples x 1]

    if plotOn:
        fig = plt.figure()
        for i in range(1, 21):   # 1:20
            ax = fig.add_subplot(4, 5, i)
            start_col = (i - 1) * 5
            end_col   = i * 5

            ax.plot(t, 1e6 * muaps[:, start_col:end_col], linewidth=1.5)
            ax.grid(True)

            labels = [str(val) for val in range(start_col + 1, end_col + 1)]
            ax.legend(labels, loc="upper right")

        plt.xlabel("ms")
        plt.ylabel("μV")
        plt.show()

    return muaps, t
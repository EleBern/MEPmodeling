import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from zero_crossing import zero_crossings_all
from downsample_muaps import load_unprocessed_muaps

root    = os.path.dirname(os.path.realpath(__file__))
h5_path = os.path.join(root, "data_MUAP", "Dist5_Monopolar_Rest_NormalCV_New.hdf5")

t, anatomical_muaps, downsampled_t, downsampled_muaps = load_unprocessed_muaps(h5_path)
dt = t[1] - t[0]
crossings = zero_crossings_all(t, anatomical_muaps)

# Calculate the amplitude of the anatomical MUAPs
for i in range(100):
    # Find peaks with a minimum height of 1/5 of the maximum voltage
    # Peaks must be at least 1.5 ms apart, if not they are considered from the same oscillation
    height = np.max(1e6 * anatomical_muaps[:, i]) / 5
    pos_peaks, _ = find_peaks(1e6 * anatomical_muaps[:, i], height=height, distance=1.5/dt) 
    neg_peaks, _ = find_peaks(- 1e6 * anatomical_muaps[:, i], height=height, distance=1.5/dt)
    peaks = np.sort(np.concatenate([pos_peaks, neg_peaks]))

    # If multiple crossing times find the crossing time between the 2 largest peaks
    if len(crossings[i].times) > 1:
        max = np.abs(anatomical_muaps[peaks[1], i] - anatomical_muaps[peaks[0], i])
        max_index = 0
        for k in range(1, len(peaks)-1):
            if np.abs(anatomical_muaps[peaks[k+1], i] - anatomical_muaps[peaks[k], i]) > max:
                max = np.abs(anatomical_muaps[peaks[k+1], i] - anatomical_muaps[peaks[k], i])
                max_index = k
        delay_index = np.argwhere(crossings[i].times > t[peaks[max_index]])
        delay = crossings[i].times[delay_index[0]]
    else:
        delay = crossings[i].times

    plt.figure()
    plt.plot(t, 1e6 * anatomical_muaps[:, i])
    plt.plot(t[peaks], 1e6 * anatomical_muaps[peaks, i], "r*")
    plt.plot(t, np.zeros(len(t)), "k--", linewidth=0.5)
    plt.plot(crossings[i].times, np.zeros(len(crossings[i].times)), "m*")
    plt.plot(delay, np.zeros(len(delay)), "go")
    plt.xlim([0, t[-1]])
    plt.title("Shape of MUAP {0}".format(i))
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (uV)")
    plt.show()

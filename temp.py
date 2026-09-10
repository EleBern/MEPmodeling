import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from zero_crossing import crossing_times
from downsample_muaps import load_unprocessed_muaps
from helper_f_genmuaps import amplitude_distribution, find_peak_times

root    = os.path.dirname(os.path.realpath(__file__))
h5_path = os.path.join(root, "data_MUAP", "Dist5_Monopolar_Rest_NormalCV_New.hdf5")

t, anatomical_muaps, downsampled_t, downsampled_muaps = load_unprocessed_muaps(h5_path)
# Calculate the zero-crossing of the anatomical MUAPs
dt = t[1] - t[0]
delay = crossing_times(t, anatomical_muaps)
# Calculate the amplitude of the anatomical MUAPs
popt, amplitude = amplitude_distribution(anatomical_muaps, t[1]-t[0])
amplitude = 1e6 * amplitude
for i in [1,18,32,35,49,76,92]:#range(100):
    # Find the peaks of the anatomical MUAPs
    peak_times = find_peak_times(anatomical_muaps[:, i], dt)
    # Plot the anatomical MUAPs with highlighting the zero-crossing, the peaks and the amplitude
    plt.figure()
    plt.plot(t, 1e6 * anatomical_muaps[:, i])
    plt.plot(t[peak_times], 1e6 * anatomical_muaps[peak_times, i], "r*")
    plt.plot(t, np.zeros(len(t)), "k--", linewidth=0.5)
    #plt.plot(crossings[i].times, np.zeros(len(crossings[i].times)), "m*")
    plt.plot(delay[i], 0, "go")
    plt.plot([delay[i], delay[i]], [-amplitude[i], amplitude[i]], "--k")
    plt.xlim([0, t[-1]])
    plt.title("Shape of MUAP {0}".format(i))
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (uV)")
    plt.show()

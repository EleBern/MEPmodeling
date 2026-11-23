import os
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from h5_helpers import load_h5_to_dict

subject = "1"
input_dir = os.path.join("/home/bernasconi/Documents/Programma/fparam_distribution", "NoNoise", "S"+subject) 
batch = os.listdir(input_dir)
total_spike_num = np.zeros(len(batch))
spike_num = np.zeros((len(batch), 100))
coeff_determination = np.zeros(len(batch))



for i in range(len(batch)):
    file_path = os.path.join(input_dir, batch[i], "result_bio_s{0}.h5".format(subject))
    with h5py.File(file_path, 'r') as f:
        tmp = load_h5_to_dict(f)
    coeff_determination[i] = tmp["ref"]["R2"]
    spike_times = tmp['ref']['sim']['spike_times']
    temp = np.sum(~np.isnan(spike_times),axis=1)
    spike_num[i] = np.sum(temp, axis=1) # number of spikes per each MN
    total_spike_num[i] = np.sum(~np.isnan(spike_times)) # number of spikes summed across all MNs

threshold = 0.93
# coeff_determination = coeff_determination[np.nonzero(coeff_determination > threshold)]
# total_spike_num = total_spike_num[np.nonzero(coeff_determination > threshold)]

print(total_spike_num)
print(spike_num[1])
print(spike_num[0])

plt.figure()
plt.boxplot(total_spike_num, tick_labels=["1-100"])
plt.ylabel("Number of spikes")
plt.xlabel("MN index")
plt.title("Distribution of MN spikes")

plt.figure()
plt.boxplot(spike_num)
plt.ylabel("Number of spikes")
plt.xlabel("MN index")
plt.title("Distribution of MN spikes")

plt.show()
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from h5_helpers import load_h5_to_dict

subject = "1"
input_dir = os.path.join("/home/bernasconi/Documents/Programma/Reduce_MUAPs", "S"+subject) 
batch = sorted(os.listdir(input_dir))
global_boxplot_coeff_determination = {}
global_boxplot_spikes = {}

for i in range(10, 90, 10):
    spike_num = np.zeros((10, i+10))
    total_spike_num = np.zeros(10)
    coeff_determination = np.zeros(10)

    for j in range(10):
        file_path = os.path.join(input_dir, batch[i+j-10], "result_bio_s{0}.h5".format(subject))
        if os.path.isfile(file_path):
            with h5py.File(file_path, 'r') as f:
                tmp = load_h5_to_dict(f)
            # print(j, file_path[-20:-18])
            coeff_determination[j] = tmp["ref"]["R2"]
            spike_times = tmp['ref']['sim']['spike_times']
            temp = np.sum(~np.isnan(spike_times),axis=1)
            spike_num[j] = np.sum(temp, axis=1) # number of spikes per each MN
            total_spike_num[j] = np.sum(~np.isnan(spike_times)) # number of spikes summed across all MNs

    global_boxplot_coeff_determination[i] = coeff_determination
    global_boxplot_spikes[i] = total_spike_num

    # print(total_spike_num)
    # print(spike_num[1])
    # print(spike_num[0])

    # plt.figure()
    # plt.title("Coefficient of determination distribution for {0} MUAPs".format(i+10))
    # plt.boxplot(coeff_determination, tick_labels=["R²"])


    # plt.figure()
    # plt.boxplot(total_spike_num, tick_labels=["1-{0}".format(i+10)])
    # plt.ylabel("Number of spikes")
    # plt.xlabel("MN index")
    # plt.title("Distribution of MN spikes")

    # plt.figure()
    # plt.boxplot(spike_num)
    # plt.ylabel("Number of spikes")
    # plt.xlabel("MN index")
    # plt.title("Distribution of MN spikes")

    # plt.show()

plt.figure()
plt.title("Distribution of R²")
d = pd.DataFrame.from_dict(global_boxplot_coeff_determination)
plt.boxplot(pd.DataFrame.from_dict(global_boxplot_coeff_determination), tick_labels=np.arange(20,100,10))
plt.ylabel("R²")
plt.xlabel("Number of MUAPs")

plt.figure()
plt.boxplot(pd.DataFrame.from_dict(global_boxplot_spikes), tick_labels=np.arange(20,100,10))
plt.ylabel("Number of spikes")
plt.xlabel("Number of MUAPs")
plt.title("Distribution of MN spikes")

plt.show()
"""
Example file to load EMG data from the existing data set 'DiLazarro_di_wave_data.hdf5'
a measurement dictionary for the data set 2020-RMT 140%-PA orientation is loaded
the correct path to the hdf5 file needs to be supplied in hdf5_path
plotting is done via matplotlib, additional smoothing can be done via scipy.ndimage.gaussian_filter1d
"""
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

def plot_emg(hdf5_path, orientation, threshold, year):
    # test if file exists in location
    if not os.path.exists(hdf5_path):
        raise ValueError('hdf5_path does not exist')

    with h5py.File(hdf5_path, 'r') as h5file:
        name_h5group = h5file[str(year)][orientation]['RMT']

        name_keys = list(name_h5group.keys())
        if threshold == None:
            threshold = np.array(list(h5file[str(year)]["PA"]["RMT"][name_keys[0]]["intensities"]))
            th_idx = np.arange(len(threshold))
        else:
            th_idx = np.argwhere(np.array(list(h5file[str(year)]["PA"]["RMT"][name_keys[0]]["intensities"])) == 140)[0]
        emg_data_full = np.array(name_h5group[name_keys[0]]['EMG']['mep'])
        emg_data_mean = np.mean(emg_data_full, axis=2)   
        time = np.array(name_h5group[name_keys[0]]['time'])

    for i in th_idx:
        plt.figure()
        plt.plot(time[i, :], emg_data_mean[i, :], label='mean')
        for j in range(emg_data_full.shape[-1]):
            if j ==0:
                # add name for legend on first occurence
                plt.plot(time[i, :], emg_data_full[i, :, j], c='k', alpha=0.3, zorder=-1, label='trials')
            else:
                plt.plot(time[i, :], emg_data_full[i, :, j], c='k', alpha=0.3, zorder=-1)
        plt.xlim([min(time[i]), max(time[i])])
        plt.xlabel('time ( ms)')
        plt.ylabel('EMG (mV)')
        if isinstance(threshold, int):
            title_string = 'Year ' + str(year) + ' th ' + str(threshold)
        else:
            title_string = 'Year ' + str(year) + ' th ' + str(threshold[i])
        plt.title(title_string)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    plot_emg()
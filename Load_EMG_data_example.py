"""
Example file to load EMG data from the existing data set 'DiLazarro_di_wave_data.hdf5'
a measurement dictionary for the data set 2020-RMT 140%-PA orientation is loaded
the correct path to the hdf5 file needs to be supplied in hdf5_path
plotting is done via matplotlib, additional smoothing can be done via scipy.ndimage.gaussian_filter1d
"""
import numpy as np
import h5py
import os
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
# hdf5_path = 'C:\\Users\\User\\nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\DIwaves_Di_Lazzaro'
hdf5_path = 'data_diLazzaro/DiLazarro_di_wave_data.hdf5'



measurement_dict = dict(orientation='PA', threshold=140, year=2020, hdf5_path=hdf5_path, sigma=1.0)

# test if file exists in location
if not os.path.exists(hdf5_path):
    raise ValueError('hdf5_path does not exist')

with h5py.File(hdf5_path, 'r') as h5file:
    name_h5group = h5file[measurement_dict['orientation']]['RMT'][str(measurement_dict['threshold'])][
        str(measurement_dict['year'])]
    name_keys = list(name_h5group.keys())
    # pick first name here (index 0) since in the dataset there is only one recording per orientation/RMT/year combination
    emg_data_full = np.array(name_h5group[name_keys[0]]['EMG']['signal_full'])
    emg_data_mean = np.array(name_h5group[name_keys[0]]['EMG']['signal_mean'])
    time = np.array(name_h5group[name_keys[0]]['time'])
emg_data_full = emg_data_full[0].T # formatting from the original .mat files into a useful shape
plt.plot(time, emg_data_mean, label='mean')
for i in range(emg_data_full.shape[0]):
    if i ==0:
        # add name for legend on first occurence
        plt.plot(time,emg_data_full[i], c='k', alpha=0.3, zorder=-1, label='trials')
    else:
        plt.plot(time, emg_data_full[i], c='k', alpha=0.3, zorder=-1)
plt.xlabel('time ( ms)')
plt.ylabel('EMG (mV)')
plt.legend()
plt.show()
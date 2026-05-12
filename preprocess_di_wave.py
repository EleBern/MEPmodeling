import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import scipy
import scipy.ndimage
import scipy.signal
from Utils import DI_wave_test_function, butter_highpass_filter, detrend
matplotlib.use('TkAgg')


def preprocess_di_wave(hdf5_path, file_args=None, t=None, dt=0.01, T=50,
                    plot=False, from_file=False,
                    enable_high_pass=False, detrend_signal=False,
                    detrend_distance=1, plot_detrend=False,
                    highpass=False, hp_cutoff=1.5,
                    plot_d_wave_detection=False):
    """
    Load a DI-wave test signal from an HDF5 file or generate a toy model.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 file containing the DI-wave data.
    file_args : dict, optional
        Dictionary with keys describing which dataset to load:
            orientation  – 'PA' or 'LM'
            threshold    – integer, e.g. 120
            year         – integer, e.g. 2020
            sigma        – Gaussian smoothing sigma
            channel      – channel index (0-based)
            subject      – subject index (0-based), default 0
            threshold_type – default 'RMT'
    t : np.ndarray, optional
        Time vector. Created from dt/T when not provided.
    dt : float
        Time step in ms (default 0.01).
    T : float
        Total duration in ms (default 20).
    plot : bool
        Plot the resulting target signal.
    from_file : bool
        If True, load from the HDF5 file; otherwise return the toy model.
    enable_high_pass : bool
        Apply a high-pass filter to the loaded signal.
    detrend_signal : bool
        Apply detrending to the loaded signal.
    detrend_distance : float
        Distance in ms used for peak detection during detrending.
    plot_detrend : bool
        Plot intermediate detrending result.
    highpass : bool
        Additional flag to trigger high-pass filtering (overridden by enable_high_pass).
    hp_cutoff : float
        High-pass filter cutoff in Hz.
    plot_d_wave_detection : bool
        Plot the processed signal alongside the raw data.

    Returns
    -------
    target : np.ndarray
        The target DI-wave signal interpolated onto t.
    t : np.ndarray
        The time vector used.
    """

    if t is None:
        t = np.arange(0, T, dt)

    # Default parameters for the toy model (used when from_file=False)
    test_func_intensity = 1.5
    test_func_t0 = 0.2
    test_func_dt = 1.5
    test_func_width = 0.3

    if enable_high_pass and not highpass:
        highpass = True

    # ------------------------------------------------------------------ #
    # Toy model — no file required                                         #
    # ------------------------------------------------------------------ #
    if not from_file:
        target = DI_wave_test_function(t,
                                       intensity=test_func_intensity,
                                       t0=test_func_t0,
                                       dt=test_func_dt,
                                       width=test_func_width)
        if plot:
            plt.plot(t, target)
            plt.xlabel('time in ms')
            plt.ylabel('v in mV')
            plt.grid()
            plt.show()
        return target, t

    # ------------------------------------------------------------------ #
    # Load from HDF5 file                                                  #
    # ------------------------------------------------------------------ #
    # Default data_dict; updated with any caller-supplied file_args
    data_dict = dict(orientation='PA', threshold=100, year=2020,
                     threshold_type='RMT', channel=0, subject=0,
                     sigma=1)
    if file_args is not None:
        data_dict.update(file_args)

    with h5py.File(hdf5_path, 'r') as h5file:
        name_h5group = (h5file[data_dict['orientation']]
                               [data_dict['threshold_type']]
                               [str(data_dict['threshold'])]
                               [str(data_dict['year'])])
        name_dict = dict(name_h5group)
        name_keys = name_dict.keys()

        di_signals = []
        times = []

        subject = data_dict['subject']
        if not isinstance(subject, list):
            subject = [subject]

        for i_key, key in enumerate(name_keys):
            if i_key in subject:
                subject_i = name_h5group[key]
                times.append(np.array(subject_i['time_short']))
                channel_keys = dict(subject_i).keys()
                single_channels = []
                channel = data_dict['channel']
                if not isinstance(channel, list):
                    channel = [channel]
                for i_ch, ch_key in enumerate(channel_keys):
                    if i_ch in channel:
                        # Read signal_short into a numpy array while the file is still open
                        single_channels.append(np.array(subject_i[ch_key]['signal_short']))
                di_signals.append(single_channels)

    measurement_data_original = di_signals[0][0]

    # ------------------------------------------------------------------ #
    # Detrend branch                                                       #
    # ------------------------------------------------------------------ #
    if detrend_signal:
        t_data = times[0]
        detrend_thr = 0.001
        d_wave_width = 1.5

        # Dataset-specific settings
        orient = data_dict['orientation']
        year   = data_dict['year']
        thr    = data_dict['threshold']
        ch     = data_dict['channel']

        if orient == 'PA' and year == 2020 and thr == 140 and ch == 0:
            idx_start, idx_end = 0, 87
            height_d_wave = 1
            d_wave_width = 1.0
            do_detrend = False
        elif orient == 'PA' and year == 2020 and thr == 120 and ch == 0:
            idx_start, idx_end = 0, t_data.shape[0]
            height_d_wave = 1
            do_detrend = True
        elif orient == 'PA' and year == 2020 and thr == 100 and ch == 0:
            idx_start, idx_end = 0, 90
            height_d_wave = 1.05
            do_detrend = False
        elif orient == 'PA' and year == 2007 and thr == 120 and ch == 0:
            idx_start, idx_end = 0, t_data.shape[0]
            height_d_wave = 0.5
            do_detrend = False
        elif orient == 'PA' and year == 2007 and thr == 150 and ch == 0:
            idx_start, idx_end = 0, t_data.shape[0]
            height_d_wave = 1.0
            do_detrend = True
            detrend_thr = 0.0002
        elif orient == 'PA' and year == 2004 and thr == 154:
            idx_start, idx_end = 0, t_data.shape[0]
            height_d_wave = 0.4
            do_detrend = False
        elif orient == 'PA' and year == 2004 and thr == 146:
            idx_start, idx_end = 0, t_data.shape[0]
            height_d_wave = 0.1
            do_detrend = True
            detrend_thr = 1e-4
        elif orient == 'LM':
            height_d_wave = measurement_data_original.max() * 0.7
            idx_start, idx_end = 0, t_data.shape[0]
            do_detrend = False
        else:
            idx_start, idx_end = 0, t_data.shape[0]
            height_d_wave = 1
            do_detrend = False
            d_wave_width = 2.0

        measurement_data = measurement_data_original[idx_start:idx_end].copy()
        t_data = t_data[idx_start:idx_end]

        measurement_data_filtered = scipy.ndimage.gaussian_filter1d(
            measurement_data, sigma=data_dict['sigma'])

        d_wave_idx = scipy.signal.find_peaks(
            measurement_data, height=height_d_wave)[0][0]

        if highpass:
            measurement_data_filtered = butter_highpass_filter(
                measurement_data_filtered, cutoff=hp_cutoff,
                fps=int(1 / dt))

        t_d_wave = t_data[d_wave_idx]
        d_wave_end_idx = np.where(t_data > t_d_wave + (d_wave_width / 2))[0][0]

        if do_detrend and detrend_signal:
            measurement_data_filtered = detrend(
                t_data, measurement_data_filtered,
                find_peaks_args=dict(threshold=detrend_thr), plot=False)
            measurement_data_filtered[measurement_data_filtered < 0] = 0

        measurement_data_filtered[:d_wave_end_idx] = 0
        measurement_data_filtered[-1] = 0

        if plot_d_wave_detection:
            plt.plot(t_data, measurement_data_filtered)
            plt.plot(t_data, measurement_data_original[idx_start:idx_end],
                     alpha=0.4, color='k', linestyle='--')
            plt.xlabel('t (ms)')
            plt.ylabel('v (µV)')
            plt.title(f"{data_dict['orientation']} {data_dict['threshold']} "
                      f"{data_dict['year']} {data_dict['channel'] + 2}")
            plt.show()

    # ------------------------------------------------------------------ #
    # High-pass branch (no detrend)                                        #
    # ------------------------------------------------------------------ #
    else:
        t_data = times[0]
        dt_data = np.diff(t_data)[0]
        d_wave_time = 3.5
        if data_dict['orientation'] == 'PA' and data_dict['year'] == 2007:
            d_wave_time = 3.0
        idx_t_dwave_end = np.where(t_data > d_wave_time)[0][0]

        measurement_data_i_waves = measurement_data_original.copy()

        if measurement_data_i_waves.max() > 6:
            measurement_data_i_waves /= 3

        measurement_data_smooth = scipy.ndimage.gaussian_filter1d(
            measurement_data_i_waves, sigma=data_dict['sigma'])
        measurement_data_filtered = butter_highpass_filter(
            measurement_data_smooth, cutoff=0.1, fps=int(1 / dt_data))

        if plot_d_wave_detection:
            plt.plot(t_data, measurement_data_filtered, label='filtered data')
            plt.plot(t_data, measurement_data_original,
                     alpha=0.4, color='k', linestyle='--', label='raw data')
            plt.xlabel('t (ms)')
            plt.ylabel('v (µV)')
            plt.title(f"{data_dict['orientation']} {data_dict['threshold']} "
                      f"{data_dict['year']} {data_dict['channel'] + 2}")
            plt.legend()
            plt.show()

    target = np.interp(t, t_data, measurement_data_filtered, left=0, right=0)

    if True:
        plt.plot(t, target)
        plt.xlabel('time in ms')
        plt.ylabel('v in mV')
        plt.grid()
        plt.show()

    return target, t
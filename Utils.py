import numpy as np
import time
import re
import h5py
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
from scipy.ndimage import gaussian_filter1d, shift
from scipy.signal import correlate, find_peaks
matplotlib.use('TkAgg')

def sigmoid(x, x0, r, amp):
    """
    Parametrized sigmoid function.

    .. math::
        y = \\frac{amp}{1+e^{-r(x-x_0)}}

    Parameters
    ----------
    x : np.ndarray of float
        (N_x) X-values the function is evaluated in.
    x0 : float
        Horizontal shift along the abscissa.
    r : float
        Slope parameter (steepness).
    amp : float
        Maximum value the sigmoid converges to.

    Returns
    -------
    y : np.ndarray of float
        (N_x) Function value at argument x.
    """
    y = amp / (1 + np.exp(-r * (x - x0)))
    return y


def DI_wave_test_function(t, intensity, t0=5, dt=1.4, width=0.25):
    """
    Determines cortical DI waves from TMS

    Parameters
    ----------
    t: ndarray of float [n_t]
        Time axis in ms
    intensity: float
        Stimulator intensity w.r.t resting motor threshold (typical range: [0 ... 2])
    t0: float
        offset time
    dt: float
        Spacing of waves in ms
    width: float
        Width of waves

    Returns
    -------
    y: ndarray of float [n_t]
        DI waves
    """

    waves = ["D", "I1", "I2", "I3", "I4"]

    x0 = dict()
    x0["D"] = 1.6952640144480995
    x0["I1"] = 1.314432218728424
    x0["I2"] = 1.4421623825084195
    x0["I3"] = 1.31643163560532
    x0["I4"] = 1.747079479469914

    amp = dict()
    amp["D"] = 12.83042571812661 / 35.46534715796085
    amp["I1"] = 35.46534715796085 / 35.46534715796085
    amp["I2"] = 26.15109003222628 / 35.46534715796085
    amp["I3"] = 15.491215097559184 / 35.46534715796085
    amp["I4"] = 10.461195366965226 / 35.46534715796085

    r = dict()
    r["D"] = 13.945868670402973
    r["I1"] = 8.707029476168504
    r["I2"] = 7.02266347578131
    r["I3"] = 16.74855628350182
    r["I4"] = 17.85806255278076

    y = np.zeros(len(t))

    for i, w in enumerate(waves):
        y_ = np.exp(-(t - t0 - i * dt) ** 2 / (2 * width ** 2))
        y_ = y_ / np.max(y_)
        y_ = y_ * sigmoid(intensity, amp=amp[w], r=r[w], x0=x0[w])
        y = y + y_

    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.9)
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fps, order=5):
    b, a = butter_highpass(cutoff, fps, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def detrend(x, y, find_peaks_args=dict(threshold=0.05, distance=1), plot=False, start_from_first_peak=False):
    """
    Function that does detrending by finding the peaks of the negative signal and
    interpolating a line for the lower bound of the signal over time. This lower bound
    is then subtracted from the original signal to detrend it. If peaks are not found, this returns the
    original signal
    :param x: np.ndarray, gird (i.e. time) of the original signal
    :param y: np.ndarray, original signal
    :return: y_detrend, np.ndarray, same shape as y: detrended version of the original signal
    """
    find_peaks_args['x'] = -y
    neg_peaks_idxs = find_peaks(**find_peaks_args)[0]
    if len(neg_peaks_idxs)>0:
        y_low = np.interp(x, x[neg_peaks_idxs], y[neg_peaks_idxs])
        if start_from_first_peak:
            start_idx = neg_peaks_idxs[0]
        else:
            start_idx = 0
        y_detrend = y.copy()
        y_detrend[start_idx:] = y[start_idx:] - y_low[start_idx:]
    else:
        y_detrend = y
    if plot:
        plt.plot(x, y)
        plt.plot(x, y_detrend)
        plt.scatter(x[neg_peaks_idxs], y[neg_peaks_idxs], color='red', marker='x')
        plt.legend(['original', 'detrended'])
        plt.show()
    return y_detrend
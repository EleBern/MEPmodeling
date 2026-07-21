import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

parentDir = os.path.dirname(os.getcwd())
sys.path.append(parentDir)

from gen_DIwave import gen_DIwave
from generate_EP import generate_EP
from deconv_DIwave import deconvreg

# ---------fig_method_2.png----------


def sigmoid(x, x0, r, a):
    return a / (1 + np.exp(r * (x0 - x)))


# Define x values and scale
x = 3 / 80 * np.arange(-3, 31, 3) + 1  # AMT average
x = x * 0.75                            # AMT to RMT
z = np.linspace(0.55, 1.7, 1000)        # Continuous range for fitting

# Data
I1 = np.array([0, 2.609, 2.958, 10.57, 16.79, 18.82, 30.33, 30.97,
               37.46, 33.35, 36.22, 36.85])
I2 = np.array([0, 0, 0, 10.06, 9.113, 11.17, 14.89, 18.86,
               22.56, 21.34, 25.60, 29.58])
I3 = np.array([0, 0, 0, 0, 7.659, 10.90, 16.10, 15.69,
               16.69, 13.47, 16.71, 21.34])
I4 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11.44, 8.991])
D = np.array([0, 0, 0, 0, 0, 0, 0, 0, 6.389, 9.444, 10.00, 14.17])
data = [D, I1, I2, I3, I4]

colors = np.array([
    [64, 121, 184],   # blue
    [238, 125, 0],    # orange
    [79, 160, 14],    # green
    [196, 31, 25],    # red
    [144, 103, 193],  # purple
]) / 255
labels = ['D', 'I1', 'I2', 'I3', 'I4']

fig = plt.figure()
width, height = 10, 8  # cm
fig.set_size_inches(width/1.5, height/1.5)

gs = fig.add_gridspec(3, 2)
ax1 = fig.add_subplot(gs[:, 0])
handles = []
for i, y in enumerate(data):
    c = colors[i]
    h = ax1.scatter(x, y, s=20, color=c)
    handles.append(h)

    p0 = [1.5, 10, 20]
    lb = [1, 0, 0]
    ub = [2.5, 500, 40]
    try:
        p, _ = curve_fit(sigmoid, x, y, p0=p0, bounds=(lb, ub), maxfev=10000)
    except RuntimeError:
        p = p0
    print(f'Fitted parameters for {labels[i]}: {list(p)}')
    ax1.plot(z, sigmoid(z, *p), color=c, linewidth=1.5)

ax1.set_ylim([-1, 40])
ax1.legend(handles, labels, loc='upper left')
ax1.set_xlabel('TMS intensity (RMT)')
ax1.set_ylabel(r'Amplitude ($\mu$V)')

# ========================================
dt = 0.01  # ms
t = np.arange(0, 20, dt)
TMS = [0.8, 1, 1.5]  # in %RMT
meancurve2 = np.column_stack([gen_DIwave(t, tms) for tms in TMS])
times = t

# ---------EP------------
d = 0.1
EP, t_ep, _ = generate_EP(d, 0, 1)  # get EP
dt2 = t_ep[1] - t_ep[0]
EP = -EP

# --------DIwave-------------
times2 = np.arange(times[0], times[-1] + dt2, dt2)  # ms
interp_func = interp1d(times, meancurve2, axis=0, fill_value='extrapolate')
meancurve3 = interp_func(times2)
tidx = np.where(times2 > 0)[0]
times2 = times2[tidx]
padding = 1000
meancurve3 = meancurve3[tidx, :]
meancurve3 = np.vstack([np.tile(meancurve3[0, :], (padding, 1)), meancurve3])
times3 = np.concatenate([np.full(padding, np.nan), times2])

# --------decov(DIwave,EP)-----------
lam = 100

rate = deconvreg(meancurve3[:,0], EP, lam)
for i in range(1, np.shape(meancurve3)[1]):
    tmp = deconvreg(meancurve3[:,i], EP, lam)
    rate = np.vstack([rate, tmp])
rate = rate.T

# -----------------------------------
xlimits = [0, 15]

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(times, meancurve2 / np.max(meancurve2), linewidth=1.5)
ax2.set_xlim(xlimits)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylabel('Potential')
ax2.set_yticks([0, 1])
ax2.set_ylim([-0.1, 1])
ax2.set_title('DI-waves')
ax2.legend([str(tms) for tms in TMS])

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t_ep - t_ep[0], EP, 'k', linewidth=1)
ax3.set_xlim(xlimits)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_ylabel('Potential')
ax3.set_yticks([0, 1])
ax3.set_ylim([-0.5, 1])
ax3.set_title(f'EP (d/c={d:g})')

ax4 = fig.add_subplot(gs[2, 1])
ax4.plot(times3, rate / np.max(rate), linewidth=1.5)
ax4.set_xlim(xlimits)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.set_xlabel('Time (msec)')
ax4.set_ylabel('Firing rate')
ax4.set_yticks([0, 1])
ax4.set_ylim([-0.1, 1])
ax4.set_title('Deconvoled DI-waves')

fig.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig(os.path.join('figures', 'fig_method_2.png'))

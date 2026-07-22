import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

parentDir = os.path.dirname(os.getcwd())
sys.path.append(parentDir)

# ---------fig_method_4.png----------
fig, axes = plt.subplots(1, 3, figsize=(15 , 5))#figsize=(15 / 2.54, 5 / 2.54))
fig.subplots_adjust(wspace=0.2)

# -----gamma-----
ax = axes[0]
shape = [1, 3, 5]        # shape
rate = [0.5, 0.5, 0.5]   # rate
handles = []
legend_entries = []
x = np.linspace(0, 30, 1000)
for a, r in zip(shape, rate):
    pdf_vals = gamma.pdf(x, a, scale=1 / r)
    h, = ax.plot(x, pdf_vals, linewidth=1.5)
    handles.append(h)
    legend_entries.append(rf'$\alpha={a:.1f}, \lambda={r:.1f}$')
ax.set_xlabel('Time (ms)', fontsize=15, fontname='calibri')
ax.set_ylabel('Prob. density', fontsize=15, fontname='calibri')
ax.set_xlim([0, 30])
ax.set_ylim([0, 0.5])
ax.set_title('Gamma')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(handles, legend_entries, loc='best')

# -----Trigger time (varying N)-----
ax = axes[1]
shape = [1, 1, 1, 1]
rate = [0.5, 0.5, 0.5, 0.5]
shifts = [20, 21, 22, 23]  # ms
Ns = [100, 75, 50, 25]     # N
handles = []
legend_entries = []
for a, r, shift, N in zip(shape, rate, shifts, Ns):
    spike_times = gamma.ppf(np.linspace(0, 0.99, N), a, scale=1 / r)
    h = ax.scatter(spike_times + shift, np.arange(1, N + 1), s=20, marker='.')
    handles.append(h)
    legend_entries.append(f'N={N}')
ax.legend(handles, legend_entries, prop={'family': 'calibri', 'size': 10})
ax.set_yticks(np.arange(0, 101, 25))
ax.set_xlim([20, 50])
ax.set_ylim([0, 100])
ax.set_xlabel('Time (ms)', fontsize=15, fontname='calibri')
ax.set_ylabel('Motor unit', fontsize=15, fontname='calibri')
ax.set_title('Trigger time')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# -----Trigger time (shape=5)-----
ax = axes[2]
shape = [5, 5, 5, 5]
rate = [0.5, 0.5, 0.5, 0.5]
shifts = [20, 21, 22, 23]
Ns = [100, 75, 50, 25]
for a, r, shift, N in zip(shape, rate, shifts, Ns):
    spike_times = gamma.ppf(np.linspace(0, 0.99, N), a, scale=1 / r)
    ax.scatter(spike_times + shift, np.arange(1, N + 1), s=20, marker='.')
ax.legend(handles, legend_entries, prop={'family': 'calibri', 'size': 10}, loc='upper left')
ax.set_xlabel('Time (ms)', fontsize=15, fontname='calibri')
ax.set_yticks([])
ax.set_yticks([], minor=True)
ax.set_ylim([0, 100])
ax.set_xlim([20, 50])
ax.set_title('Trigger time')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

os.makedirs('figures', exist_ok=True)
fig.savefig(os.path.join('figures', 'fig_method_4.png'))

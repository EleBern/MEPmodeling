import os
import sys

import numpy as np
import matplotlib.pyplot as plt

parentDir = os.path.dirname(os.getcwd())
sys.path.append(parentDir)

from load_muap import load_muap

# ---------fig_method_3-detail.png----------
muaps, tmuap = load_muap()

fig, axes = plt.subplots(10, 10, figsize=(12 / 2.54, 15 / 2.54))
axes = axes.flatten()

for idx in range(1, 101):
    i = idx - 1
    ax = axes[i]
    ax.plot(tmuap, muaps[:, i] * 1000, 'b', linewidth=1.5, zorder=10)
    ax.plot([0, 10], [0, 0], 'k')
    ax.scatter([0, 5, 10], [0, 0, 0], s=8, marker='|', color='k', zorder=15)

    ax.set_xlim([-2, 10])
    if idx <= 20:
        group = muaps[:, 0:20]
        ytick = 0.03
    elif idx <= 40:
        group = muaps[:, 20:40]
        ytick = 0.06
    elif idx <= 60:
        group = muaps[:, 40:60]
        ytick = 0.1
    elif idx <= 80:
        group = muaps[:, 60:80]
        ytick = 0.2
    else:
        group = muaps[:, 80:100]
        ytick = 0.5

    ymax = np.max(np.abs(group)) * 1.1 * 1000
    ylimits = [-ymax, ymax]
    ax.set_ylim(ylimits)

    if idx in [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]:
        ax.plot([0, 0], [0, ytick], 'k')
        ax.scatter([0], [ytick], s=8, marker='_', color='k')
        ax.text(0, ylimits[1] / 2, f'{ytick} mV',
                 va='bottom', fontsize=6, fontname='calibri')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(str(idx))
    ax.axis('off')

fig.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig(os.path.join('figures', 'fig_method_3-detail.png'))

# ---------fig_method_3.png----------
muaps, tmuap = load_muap()
ymax = np.max(np.abs(muaps)) * 1.1 * 1000
ylimits = [-ymax, ymax]

fig2, axes2 = plt.subplots(2, 5, figsize=(15 / 2.54, 7 / 2.54))
axes2 = axes2.flatten()

for i in range(10):
    ax = axes2[i]
    cols = muaps[:, i * 10:(i + 1) * 10] * 1000
    ax.plot(tmuap, cols, 'k', linewidth=0.5)
    ax.set_ylim(ylimits)
    ax.grid(True)
    ax.set_xticks([0, 5, 10])
    ax.set_xlim([0, 10])
    ax.set_title(f'{i * 10 + 1} - {i * 10 + 10}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig2.tight_layout()
fig2.savefig(os.path.join('figures', 'fig_method_3.png'))

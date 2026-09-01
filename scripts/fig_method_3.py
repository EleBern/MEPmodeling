import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

parentDir = os.path.dirname(os.getcwd())
sys.path.append(parentDir)
from h5_helpers import load_h5_to_dict

# ---------fig_method_3-detail.png----------
root = ".."
h5_path = os.path.join(root, "data_MUAP", "pheno_muaps.h5")
if os.path.exists(h5_path):
    with h5py.File(h5_path, 'r') as f:
        tmp = load_h5_to_dict(f)

    p_muaps = tmp["muaps"]   # [n_samples x n_muaps]
    tmuap     = tmp["t"].T     # [n_samples x 1]

h5_path = os.path.join(root, "data_MUAP", "muap.h5")

if os.path.exists(h5_path):
    with h5py.File(h5_path, 'r') as f:
        tmp = load_h5_to_dict(f)

    a_muaps = tmp["muaps"]   # [n_samples x n_muaps]

fig, axes = plt.subplots(10, 10, figsize=(12 / 2.54, 15 / 2.54))
axes = axes.flatten()

for idx in range(1, 101):
    i = idx - 1
    ax = axes[i]
    ax.plot(tmuap, a_muaps[:, i] * 1000, 'b', linewidth=1.5)
    ax.plot(tmuap, p_muaps[:, i] * 1000, 'r', linewidth=1.5, zorder=10, alpha=0.7)
    ax.plot([0, 10], [0, 0], 'k')
    ax.scatter([0, 5, 10], [0, 0, 0], s=8, marker='|', color='k')

    ax.set_xlim([-2, 10])
    if idx <= 20:
        group = a_muaps[:, 0:20]
        #group = p_muaps[:, 0:20]
        ytick = 0.03
    elif idx <= 40:
        group = a_muaps[:, 20:40]
        #group = p_muaps[:, 20:40]
        ytick = 0.06
    elif idx <= 60:
        group = a_muaps[:, 40:60]
        #group = p_muaps[:, 40:60]
        ytick = 0.1
    elif idx <= 80:
        group = a_muaps[:, 60:80]
        #group = p_muaps[:, 60:80]
        ytick = 0.2
    else:
        group = a_muaps[:, 80:100]
        #group = p_muaps[:, 80:100]
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

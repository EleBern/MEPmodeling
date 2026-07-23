import os
import sys

import numpy as np
import h5py
import matplotlib.pyplot as plt

parentDir = os.path.dirname(os.getcwd())
sys.path.append(parentDir)

from h5_helpers import load_h5_to_dict

def load_h5(filename):
    """Open an .h5 file and recursively load it into a nested dict."""
    with h5py.File(filename, 'r') as f:
        return load_h5_to_dict(f)


# ---------fig_method_6.png----------

AMPAweights = np.arange(0.2, 0.8 + 1e-9, 0.1)
score = np.zeros((len(AMPAweights) + 1, 5, 10))  # [AMPA2NMDAratio, R2, In->MN1, MN1->RC, RC->MN1]

for subj in range(1, 11):
    print(subj)
    s = subj - 1
    for j, w in enumerate(AMPAweights):
        tmp = load_h5(os.path.join(
            parentDir, 'fitted_results', 'bio', 'fixed_AMPAweight',
            f'result_bio_s{subj}[{w:g}].h5'))
        ref = tmp['ref']
        score[j, :, s] = [
            ref['model']['AMPAweight'].item(),
            ref['R2'].item(),
            np.mean(ref['model']['R']),
            np.mean(ref['model']['Wexc']),
            np.mean(ref['model']['RWinh']),
        ]
    tmp = load_h5(os.path.join(
        parentDir, 'fitted_results', 'bio', f'result_bio_s{subj}.h5'))
    ref = tmp['ref']
    score[-1, :, s] = [
        ref['model']['AMPAweight'].item(),
        ref['R2'].item(),
        np.mean(ref['model']['R']),
        np.mean(ref['model']['Wexc']),
        np.mean(ref['model']['RWinh']),
    ]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))  
axes = axes.flatten()

for subj in range(1, 11):
    s = subj - 1
    ax = axes[s]
    tmp = score[:, :, s][np.argsort(score[:, 0, s])]

    h1, = ax.plot(tmp[:, 0], tmp[:, 1], '-bo', linewidth=1, markerfacecolor='none')  # R2
    ax.plot(score[-1, 0, s], score[-1, 1, s], 'ro', linewidth=1, markerfacecolor='none')
    ax.set_xlim([0, 1])
    ax.set_ylim([0.5, 1])
    ax.tick_params(axis='y', colors='blue')
    ax.spines['left'].set_color('blue')
    ax.yaxis.label.set_color('blue')
    if subj in [1, 6]:
        ax.set_ylabel('R2')

    ax2 = ax.twinx()
    h2, = ax2.plot(tmp[:, 0], tmp[:, 2], '-yo', linewidth=1, markerfacecolor='none')  # In->MN1
    h3, = ax2.plot(tmp[:, 0], tmp[:, 3], '-go', linewidth=1, markerfacecolor='none')  # MN1->RC
    h4, = ax2.plot(tmp[:, 0], tmp[:, 4], '-mo', linewidth=1, markerfacecolor='none')  # RC->MN1
    ax2.plot(score[-1, 0, s], score[-1, 2, s], 'ro', linewidth=1, markerfacecolor='none')
    ax2.plot(score[-1, 0, s], score[-1, 3, s], 'ro', linewidth=1, markerfacecolor='none')
    ax2.plot(score[-1, 0, s], score[-1, 4, s], 'ro', linewidth=1, markerfacecolor='none')
    ax2.set_ylim([-1, 40])
    ax2.spines['left'].set_visible(False)

    ax.set_title(f'S{subj}')
    ax.grid(True)
    if subj == 8:
        ax.set_xlabel('AMPAweight', size=12)
    if subj in [1, 6]:
        ax.legend([h1, h2, h3, h4], ['R2', 'R', 'Wexc', 'Winh'],
                  loc='center right', bbox_to_anchor=(-0.3, 0.5),
                  bbox_transform=ax.transAxes)

fig.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig(os.path.join('figures', 'fig_method_6.png'), bbox_inches='tight')
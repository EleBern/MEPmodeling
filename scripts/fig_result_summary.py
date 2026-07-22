import os
import sys

import numpy as np
import h5py
import matplotlib.pyplot as plt

parentDir = os.path.dirname(os.getcwd())
sys.path.append(parentDir)

# h5_helpers.py lives in the parent directory alongside the other helper
# scripts (ga_MEPmodel_bio.py, MEPmodel_bio.py, etc.).
from h5_helpers import load_h5_to_dict
from MEPmodel_pheno import MEPmodel_pheno
from MEPmodel_bio import MEPmodel_bio


def load_h5(filename):
    """Open an .h5 file and recursively load it into a nested dict."""
    with h5py.File(filename, 'r') as f:
        return load_h5_to_dict(f)


# ---------fig_result_summary.png----------
score = np.zeros((10, 3))  # [Pheno, Bio, BioNoRC]
for subj in range(1, 11):
    s = subj - 1

    tmp = load_h5(os.path.join(parentDir, 'fitted_results', 'pheno',
                                f'result_pheno_s{subj}.h5'))
    _, ref = MEPmodel_pheno(tmp['p_post'], tmp['ref'], 0)
    score[s, 0] = ref['R2']

    tmp = load_h5(os.path.join(parentDir, 'fitted_results', 'bio',
                                f'result_bio_s{subj}.h5'))
    _, ref = MEPmodel_bio(tmp['p_post'], tmp['ref'], 0)
    score[s, 1] = ref['R2']

    tmp = load_h5(os.path.join(parentDir, 'fitted_results', 'bioNoRC',
                                f'result_bioNoRC_s{subj}.h5'))
    _, ref = MEPmodel_bio(tmp['p_post'], tmp['ref'], 0)
    score[s, 2] = ref['R2']

fig = plt.figure()
markerColor = np.array([128, 128, 128]) / 255
boxColor = np.array([0, 139, 139]) / 255  # noqa: F841 (kept for parity with source)
face = (*boxColor, 0.2)
edge = (*boxColor, 1) 
width = 15
height = 12
fig.set_size_inches(width / 2.54, height / 2.54)

ax1 = fig.add_subplot(2, 2, (1, 3))
bp = ax1.boxplot(score, positions=[1, 2, 3], patch_artist=True)
plt.setp(bp["medians"], color=boxColor, linewidth=2)
plt.setp(bp["boxes"], facecolor=face, edgecolor=edge, linewidth=2)
plt.setp(bp["medians"], color=edge)


# for patch in bp['boxes']:
#     patch.set(facecolor=boxColor, alpha=0.5) 
ax1.plot([1, 2, 3], score.T, '-o', markerfacecolor='none', color=markerColor, linewidth=0.5)
ax1.set_xticks([1, 2, 3])
ax1.set_xticklabels(['Pheno.', 'Bio.', 'Bio.(w/o RC)'], fontname='calibri')
ax1.set_ylabel(r'$R^2$')
ax1.set_ylim([0.5, 1])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.text(-0.2, 1.02, 'A', transform=ax1.transAxes, fontsize=14,
          fontweight='bold', ha='right', va='bottom')

ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(score[:, 1], score[:, 0], color=markerColor, facecolors='none')
ax2.plot([0, 1], [0, 1], 'k')
ax2.set_xlim([0.8, 1])
ax2.set_ylim([0.8, 1])
ax2.set_ylabel(r'$R^2$ (Pheno.)')
ax2.set_xlabel(r'$R^2$ (Bio.)')
for i in range(10):
    if i+1 == 9:
        ax2.text(score[i, 1] + 0.005, score[i, 0] - 0.02, str(i + 1),
              ha='left', fontsize=9)
    elif i+1 == 4:
        ax2.text(score[i, 1] + 0.005, score[i, 0] + 0.005, str(i + 1),
              ha='left', fontsize=9)
    else:
        ax2.text(score[i, 1] + 0.007, score[i, 0] - 0.01, str(i + 1),
                ha='left', fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_box_aspect(1)
ax2.text(-0.4, 1.05, 'B', transform=ax2.transAxes, fontsize=14,
          fontweight='bold', ha='right', va='bottom')

ax3 = fig.add_subplot(2, 2, 4)
ax3.scatter(score[:, 1], score[:, 2], color=markerColor, facecolors='none')
ax3.plot([0, 1], [0, 1], 'k')
ax3.set_xlim([0.8, 1])
ax3.set_ylim([0.5, 1])
ax3.set_ylabel(r'$R^2$ (Bio. w/o RC)')
ax3.set_xlabel(r'$R^2$ (Bio.)')
for i in range(10):
    if i+1 == 9:
        ax3.text(score[i, 1] + 0.002, score[i, 2] - 0.04, str(i + 1),
              ha='left', fontsize=9)
    elif i+1 == 5:
        ax3.text(score[i, 1], score[i, 2] - 0.045, str(i + 1),
              ha='left', fontsize=9)
    else:
        ax3.text(score[i, 1] + 0.0052, score[i, 2] - 0.015, str(i + 1),
                ha='left', fontsize=9)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_box_aspect(1)
ax3.text(-0.4, 1.05, 'C', transform=ax3.transAxes, fontsize=14,
          fontweight='bold', ha='right', va='bottom')

fig.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig(os.path.join('figures', 'fig_result_summary.svg'))

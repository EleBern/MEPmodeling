import os
import sys

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

parentDir = os.path.dirname(os.getcwd())
sys.path.append(parentDir)

from h5_helpers import load_h5_to_dict
from MEPmodel_pheno import MEPmodel_pheno
from MEPmodel_bio import MEPmodel_bio


def load_h5(filename):
    """Open an .h5 file and recursively load it into a nested dict."""
    with h5py.File(filename, 'r') as f:
        return load_h5_to_dict(f)


# ---------fig_result_summary.png----------
score = np.zeros((10, 3))  # [Pheno, Bio, BioNoRC]
score2 = np.zeros((10, 3))
for subj in range(1, 11):
    s = subj - 1

    tmp = load_h5(os.path.join(parentDir, 'fitted_results', 'pheno',
                                f'result_pheno_s{subj}.h5'))
    _, ref = MEPmodel_pheno(tmp['p_post'], tmp['ref'], 0)
    score[s, 0] = ref['R2']
    score2[s, 0] = ref['gof']

    tmp = load_h5(os.path.join(parentDir, 'fitted_results', 'bio',
                                f'result_bio_s{subj}.h5'))
    _, ref = MEPmodel_bio(tmp['p_post'], tmp['ref'], 0)
    score[s, 1] = ref['R2']
    score2[s, 1] = ref['gof']

    tmp = load_h5(os.path.join(parentDir, 'fitted_results', 'bioNoRC',
                                f'result_bioNoRC_s{subj}.h5'))
    _, ref = MEPmodel_bio(tmp['p_post'], tmp['ref'], 0)
    score[s, 2] = ref['R2']
    score2[s, 2] = ref['gof']

# --------------------------------------------------------
fig = plt.figure()
markerColor = np.array([128, 128, 128]) / 255
boxColor = np.array([0, 139, 139]) / 255  
face = (*boxColor, 0.2)
edge = (*boxColor, 1) 
width = 22
height = 12#9
fig.set_size_inches(width / 2.54, height / 2.54)

gs = fig.add_gridspec(2, 4, width_ratios=[0.575, 1.375, 0.575, 1.375],
                       wspace=0.1, hspace=0.5,
                       left=0.08, right=0.99, top=0.9, bottom=0.25)

ax1 = fig.add_subplot(gs[:, 0])
bp = ax1.boxplot(score, positions=[1, 2, 3], patch_artist=True)
plt.setp(bp["medians"], color=boxColor, linewidth=2)
plt.setp(bp["boxes"], facecolor=face, edgecolor=edge, linewidth=2)
plt.setp(bp["medians"], color=edge)

ax1.plot([1, 2, 3], score.T, '-o', markerfacecolor='none', color=markerColor, linewidth=0.5)
ax1.set_xticks([1, 2, 3])
ax1.set_xticklabels(['Pheno.', 'RC+', 'RC-'], fontname='calibri', rotation=45)
ax1.set_ylabel(r'Waveform $R^2$')
ax1.set_ylim([0.5, 1])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.text(-0.2, 1.02, 'A', transform=ax1.transAxes, fontsize=14,
          fontweight='bold', ha='right', va='bottom')

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(score[:, 1], score[:, 0], color=markerColor, facecolors='none')
ax2.plot([0, 1], [0, 1], 'k')
ax2.set_xlim([0.8, 1])
ax2.set_ylim([0.8, 1])
ax2.set_ylabel(r'$R^2$ (Pheno.)')
ax2.set_xlabel(r'$R^2$ (RC+) model')
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

ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(score[:, 1], score[:, 2], color=markerColor, facecolors='none')
ax3.plot([0, 1], [0, 1], 'k')
ax3.set_xlim([0.8, 1])
ax3.set_ylim([0.5, 1])
ax3.set_ylabel(r'$R^2$ (RC-) model')
ax3.set_xlabel(r'$R^2$ (Bio.)')
for i in range(10):
    if i+1 == 9:
        ax3.text(score[i, 1] - 0.002, score[i, 2] - 0.05, str(i + 1),
              ha='left', fontsize=9)
    elif i+1 == 5:
        ax3.text(score[i, 1]-0.005, score[i, 2] - 0.052, str(i + 1),
              ha='left', fontsize=9)
    else:
        ax3.text(score[i, 1] + 0.0052, score[i, 2] - 0.02, str(i + 1),
                ha='left', fontsize=9)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_box_aspect(1)

ax4 = fig.add_subplot(gs[:, 2])
bp2 = ax4.boxplot(score2, positions=[1, 2, 3], patch_artist=True)
plt.setp(bp2["medians"], color=boxColor, linewidth=2)
plt.setp(bp2["boxes"], facecolor=face, edgecolor=edge, linewidth=2)
plt.setp(bp2["medians"], color=edge)

ax4.plot([1, 2, 3], score2.T, '-o', markerfacecolor='none', color=markerColor, linewidth=0.5)
ax4.set_xticks([1, 2, 3])
ax4.set_xticklabels(['Pheno.', 'RC+', 'RC-'], fontname='calibri', rotation=45)
ax4.set_ylabel(r'IO $R^2$')
ax4.set_ylim([0.5, 1])
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.text(-0.2, 1.02, 'B', transform=ax4.transAxes, fontsize=14,
          fontweight='bold', ha='right', va='bottom')

ax5 = fig.add_subplot(gs[0, 3])
ax5.scatter(score2[:, 1], score2[:, 0], color=markerColor, facecolors='none')
ax5.plot([0, 1], [0, 1], 'k')
ax5.set_xlim([0.8, 1])
ax5.set_ylim([0.8, 1])
ax5.set_ylabel(r'$R^2$ (Pheno.)')
ax5.set_xlabel(r'$R^2$ (RC+) model')
for i in range(10):
    if i+1 in [3, 4, 7]:
        ax5.text(score2[i, 1], score2[i, 0]+0.005, str(i + 1),
                    ha='left', fontsize=9)
    elif i+1 in [5, 10]:
            ax5.text(score2[i, 1]-0.01, score2[i, 0]-0.02, str(i + 1),
                        ha='left', fontsize=9)
    else:
        ax5.text(score2[i, 1] + 0.005, score2[i, 0]-0.01, str(i + 1),
                ha='left', fontsize=9)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.set_box_aspect(1)

ax6 = fig.add_subplot(gs[1, 3])
ax6.scatter(score2[:, 1], score2[:, 2], color=markerColor, facecolors='none')
ax6.plot([0, 1], [0, 1], 'k')
ax6.set_xlim([0.8, 1])
ax6.set_ylim([0.5, 1])
ax6.set_ylabel(r'$R^2$ (RC-) model')
ax6.set_xlabel(r'$R^2$ (Bio.)')
for i in range(10):
    if i + 1 in [1,4,9]:
        ax6.text(score2[i, 1] + 0.005, score2[i, 2] - 0.04, str(i + 1),
                    ha='left', fontsize=9)
    else:
        ax6.text(score2[i, 1] - 0.005, score2[i, 2] + 0.02, str(i + 1),
                ha='left', fontsize=9)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.set_box_aspect(1)

os.makedirs('figures', exist_ok=True)
fig.savefig(os.path.join('figures', 'fig_result_summary.svg'))

# --------------------------------------------------------
print('waveform R^2 [Pheno,RC+,RC-]')
print('--------------------------------')
print(score.mean(axis=0))

print('IO curve R^2 [Pheno,RC+,RC-]')
print('--------------------------------')
print(score2.mean(axis=0))

t1, p1 = ttest_rel(score[:, 0], score[:, 1])
print(f'waveform R^2 Pheno vs RC+ ttest: t={t1}, p={p1}')

t2, p2 = ttest_rel(score2[:, 1], score2[:, 2])
print(f'IO R^2 RC+ vs RC- ttest: t={t2}, p={p2}')

# --------------------------------------------------------
fig2 = plt.figure()
width2 = 15
height2 = 10
fig2.set_size_inches(width2 / 2.54, height2 / 2.54)

# ----------waveform R2-------------
ax7 = fig2.add_subplot(1, 3, 1)
bp3 = ax7.boxplot(score, positions=[1, 2, 3], patch_artist=True)
plt.setp(bp3["medians"], color=boxColor, linewidth=2)
plt.setp(bp3["boxes"], facecolor=face, edgecolor=edge, linewidth=2)
plt.setp(bp3["medians"], color=edge)

ax7.plot([1, 2, 3], score.T, '-o', markerfacecolor='none', color=markerColor, linewidth=0.5)
ax7.set_xticks([1, 2, 3])
ax7.set_xticklabels(['Pheno.', 'RC+', 'RC-'], fontname='calibri')
ax7.set_ylabel(r'Waveform $R^2$')
ax7.set_ylim([0.5, 1])
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)

# ----------IO R2-------------
ax8 = fig2.add_subplot(1, 3, 2)
bp4 = ax8.boxplot(score2, positions=[1, 2, 3], patch_artist=True)
plt.setp(bp4["medians"], color=boxColor, linewidth=2)
plt.setp(bp4["boxes"], facecolor=face, edgecolor=edge, linewidth=2)
plt.setp(bp4["medians"], color=edge)

ax8.plot([1, 2, 3], score2.T, '-o', markerfacecolor='none', color=markerColor, linewidth=0.5)
ax8.set_xticks([1, 2, 3])
ax8.set_xticklabels(['Pheno.', 'RC+', 'RC-'], fontname='calibri')
ax8.set_ylabel(r'IO $R^2$')
ax8.set_ylim([0.5, 1])
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)

# -------------diff---------------
ax9 = fig2.add_subplot(1, 3, 3)
x = score[:, 1] - score[:, 2]
y = score2[:, 1] - score2[:, 2]
ax9.scatter(x, y, color=markerColor, facecolors='none')
ax9.axhline(0, color='k')
ax9.set_ylabel(r'diff IO $R^2$ (RC$^+$-RC$^-$)')
ax9.set_xlabel('diff waveform $R^2$\n(RC$^+$-RC$^-$)', fontname='calibri')
for i in range(10):
    ax9.text(x[i] + 0.015, y[i], str(i + 1), ha='left', fontsize=8)
ax9.spines['top'].set_visible(False)
ax9.spines['right'].set_visible(False)

fig2.tight_layout()
fig2.savefig(os.path.join('figures', 'fig_result_summary2.svg'))
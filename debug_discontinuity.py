import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ga_MEPmodel_bio import ga_MEPmodel_bio

def run_model(subj, R1, R2, R3, R4, R5, E1, E2, I1, I2, RC_th, MU_T, AMPAw):

    p = [R1, R2, R3, R4, R5, E1, E2, I1, I2, RC_th, MU_T, AMPAw]
    ref = ga_MEPmodel_bio(subj, withRC=1, AMPAweight=[], reRun=0, p_post=p)

    return ref['R2'], ref['sim']['spike_times']


if __name__ == '__main__':
    # fixed settings 
    SUBJ       = 1
    R2 = 1.05927459
    R3 = 1.57477574
    R4 = 0.76463778
    R5 = 9.80825998
    E1 = 19.96654087
    E2 = 11.40240176
    I1 = 0.14947802
    I2 = 0.04018722
    RC_th = 3.42343178
    MU_T = 5.
    AMPAw = 0.14387608


    R1_array = np.arange(0, 2, 0.1)
    coeff_determination = np.zeros((len(R1_array)))
    spike_num = np.zeros((len(R1_array), 100))
    total_spike_num = np.zeros((len(R1_array)))

    for i in range(len(R1_array)):
        R1 = R1_array[i]
        # Output metric
        coeff_determination[i], spike_times = run_model(SUBJ, R1, R2, R3, R4, R5, E1, E2, I1, 0, RC_th, MU_T, AMPAw)
        temp = np.sum(~np.isnan(spike_times),axis=1)
        spike_num[i] = np.sum(temp, axis=1)
        total_spike_num[i] = np.sum(~np.isnan(spike_times))


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 3), sharex=True)

    sns.heatmap(coeff_determination[np.newaxis, :], xticklabels=np.round(R1_array, 1),
                yticklabels=['Coefficient of determination'], annot=True, cmap='hot', ax=ax1)

    sns.heatmap(total_spike_num[np.newaxis, :], xticklabels=np.round(R1_array, 1),
                yticklabels=['Total number of MN spikes'], annot=True, cmap='hot', ax=ax2)

    for ax in (ax1, ax2):
        ax.tick_params(axis='y', rotation=0)   # keep the row labels horizontal
    ax2.set_xlabel('R1')
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 12))   # small height = short y axis
    sns.heatmap(spike_num.T, xticklabels=np.round(R1_array, 1),
            yticklabels=True, cmap='hot', ax=ax)
    ax.set_xlabel('R1'); plt.tight_layout() 
    ax.set_title('Number of MN spikes')
    ax.set_ylabel('MN index'); plt.tight_layout() 

    plt.show()
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from h5_helpers import load_h5_to_dict

input_dir = os.path.join("/home/bernasconi/Documents/Programma/fparam_distribution", "Subject1_NoNoise", "Results") 
output_dir = os.path.join("/home/bernasconi/Documents/Programma/fparam_distribution", "Subject1_NoNoise")
batch = os.listdir(input_dir)

R1    = np.zeros(len(batch))
R2    = np.zeros(len(batch))
R3    = np.zeros(len(batch))
R4    = np.zeros(len(batch))
R5    = np.zeros(len(batch))
E1    = np.zeros(len(batch))
E2    = np.zeros(len(batch))
I1    = np.zeros(len(batch))
I2    = np.zeros(len(batch))
RCth  = np.zeros(len(batch))
Tmu   = np.zeros(len(batch))
AMPAw = np.zeros(len(batch))

for i in range(len(batch)):
    file_path = os.path.join(input_dir, batch[i], "result_bio_s1.h5")
    with h5py.File(file_path, 'r') as f:
        tmp = load_h5_to_dict(f)
        p = tmp['p_post'].flatten()

        R1[i]    = p[0]
        R2[i]    = p[0]
        R3[i]    = p[0]
        R4[i]    = p[0]
        R5[i]    = p[0]
        E1[i]    = p[5]
        E2[i]    = p[6]
        I1[i]    = p[7]
        I2[i]    = p[8]
        RCth[i]  = p[9]
        Tmu[i]   = p[10]
        AMPAw[i] = p[11]

plt.figure()
plt.boxplot([R1, R2, R3, R4, R5], tick_labels=["R1", "R2", "R3", "R4", "R5"])
plt.xlim([0, 10])
plt.show()

plt.figure()
plt.boxplot([E1, E2], tick_labels=["E1", "E2"])
plt.xlim([0, 20])
plt.show()

plt.figure()
plt.boxplot([I1, I2, RCth, Tmu, AMPAw], tick_labels=["I1", "I2", "RCth", "Tmu", "AMPAw"])
plt.show()


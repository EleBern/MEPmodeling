import os
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
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
coeff_determination = np.zeros(len(batch))

for i in range(len(batch)):
    file_path = os.path.join(input_dir, batch[i], "result_bio_s1.h5")
    with h5py.File(file_path, 'r') as f:
        tmp = load_h5_to_dict(f)
    p = tmp['p_post'].flatten()

    R1[i]    = p[0]
    R2[i]    = p[1]
    R3[i]    = p[2]
    R4[i]    = p[3]
    R5[i]    = p[4]
    E1[i]    = p[5]
    E2[i]    = p[6]
    I1[i]    = p[7]
    I2[i]    = p[8]
    RCth[i]  = p[9]
    Tmu[i]   = p[10]
    AMPAw[i] = p[11]

    coeff_determination[i] = tmp["ref"]["R2"]

plt.figure()
plt.title("Coefficient of determination distribution")
plt.boxplot(coeff_determination, tick_labels=["R²"])
plt.show()

threshold = 0.5
coeff_determination = coeff_determination[np.nonzero(coeff_determination > threshold)]
R1 = R1[np.nonzero(coeff_determination > threshold)]
R2 = R2[np.nonzero(coeff_determination > threshold)]
R3 = R3[np.nonzero(coeff_determination > threshold)]
R4 = R4[np.nonzero(coeff_determination > threshold)]
R5 = R5[np.nonzero(coeff_determination > threshold)]
E1 = E1[np.nonzero(coeff_determination > threshold)]
E2 = E2[np.nonzero(coeff_determination > threshold)]
I1 = I1[np.nonzero(coeff_determination > threshold)]
I2 = I2[np.nonzero(coeff_determination > threshold)]
RCth = RCth[np.nonzero(coeff_determination > threshold)]
Tmu = Tmu[np.nonzero(coeff_determination > threshold)]
AMPAw = AMPAw[np.nonzero(coeff_determination > threshold)]

print("Out of {0} simulations, {1} had a R² > {2}".format(len(batch), len(R1), threshold))

# Plot parameter distribution with ylim as parameter boundary
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

axs[0, 0].boxplot([R1, R2, R3, R4, R5], tick_labels=["R1", "R2", "R3", "R4", "R5"])
axs[0, 0].set_ylim([0, 10])

axs[0, 1].boxplot([E1, E2], tick_labels=["E1", "E2"])
axs[0, 1].set_ylim([0, 20])

axs[0, 2].boxplot([I1, I2], tick_labels=["I1", "I2"])
axs[0, 2].set_ylim([0, 10])

axs[1, 0].boxplot([RCth], tick_labels=["RCth"])
axs[1, 0].set_ylim([1, 10])

axs[1, 1].boxplot([Tmu], tick_labels=["Tmu"])
axs[1, 1].set_ylim([5, 10])

axs[1, 2].boxplot([AMPAw], tick_labels=["AMPAw"])
axs[1, 2].set_ylim([0, 1])

fig.suptitle("Fitted parameters distribution")
plt.show()


# Plot parameter distribution zoomed in 
fig, axs = plt.subplots(1, 3, figsize=(12, 6))

axs[0].boxplot([R1, R2, R3, R4, R5], tick_labels=["R1", "R2", "R3", "R4", "R5"])
axs[1].boxplot([E1, E2], tick_labels=["E1", "E2"])
axs[2].boxplot([I1, I2, RCth, Tmu, AMPAw], tick_labels=["I1", "I2", "RCth", "Tmu", "AMPAw"])

fig.suptitle("Fitted parameters distribution")
plt.show()

# Correlation between pararmeters
data = {"R1" : R1, 
    "R2": R2,
    "R3": R3,
    "R4": R4, 
    "R5": R5, 
    "E1": E1, 
    "E2": E2,    
    "I1": I1,    
    "I2": I2,    
    "RCth": RCth,  
    "Tmu": Tmu,   
    "AMPAw": AMPAw, 
    "coeff_determination": coeff_determination 
}

dataframe = pd.DataFrame(data, columns=["R1", "R2", "R3", "R4", "R5", "E1", "E2", "I1", "I2", "RCth", "Tmu", "AMPAw"])
correlation_matrix = dataframe.corr()

# Remove upper half of correlation matrix for plot
mask = np.zeros_like(correlation_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
correlation_matrix[mask] = np.nan

# Plot correlation matrix
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
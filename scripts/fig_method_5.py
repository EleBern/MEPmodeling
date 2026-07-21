import os
import sys

import numpy as np
import matplotlib.pyplot as plt

parentDir = os.path.dirname(os.getcwd())
sys.path.append(parentDir)

# ---------fig_method_5.png----------

# ------ Biexponential synaptic kernels (analytical)------
labels = ['In->MN(AMPA | 1ms, 5ms)',
          'In->MN(NMDA | 3ms, 50ms)',
          'MN->RC(AChR | 0.5ms, 3.6ms)',
          'MN->RC(AChR | 1.8ms, 20.2ms)',
          'RC->MN(GlyR | 1ms, 6ms)']
tau = np.array([[1, 5],
                [3, 50],
                [0.5, 3.6],
                [1.8, 20.2],
                [1, 6]])
h = [1.8692, 1.2731, 1.5977, 1.3908, 1.7175]  # normalizing terms

dt = 0.1        # ms
tlength = 50    # ms
t = np.arange(0, tlength, dt)
kernels = np.zeros((5, len(t)))
for i in range(5):
    kernels[i, :] = h[i] * (np.exp(-t / tau[i, 1]) - np.exp(-t / tau[i, 0]))

fig1, ax1 = plt.subplots()
ax1.plot(t, kernels.T, linewidth=2)
ax1.set_ylim([0, 1])
ax1.grid(True)
ax1.legend(labels)
ax1.set_title('Biexponential synaptic kernels (analytical)')

# ------ Biexponential synaptic kernels (numerical)------
kernels2 = np.zeros((5, len(t)))
h2 = [14.6481, 3.9770, 26.5924, 6.9913, 14.0482]  # normalizing terms
for i in range(5):
    v = np.zeros(len(t))
    v2 = np.zeros(len(t))
    In = np.zeros(len(t))
    In[0] = 1  # impulse
    for tt in range(len(t) - 1):
        dv = v2[tt]
        dv2 = (h2[i] * In[tt]
               - (tau[i, 0] + tau[i, 1]) * v2[tt] / tau[i, 0] / tau[i, 1]
               - v[tt] / tau[i, 0] / tau[i, 1])
        v[tt + 1] = v[tt] + dv * dt
        v2[tt + 1] = v2[tt] + dv2 * dt
    kernels2[i, :] = v

fig2, ax2 = plt.subplots()
ax2.plot(t, kernels2.T, linewidth=2)
ax2.set_ylim([0, 1.01])
ax2.set_xlim([0, 50])
ax2.grid(True)
ax2.legend(labels)
ax2.set_title('Biexponential synaptic kernels (numerical)')

# Note: like the original MATLAB script, only the figure created last
# (the numerical kernels) is saved to disk.
os.makedirs('figures', exist_ok=True)
fig2.savefig(os.path.join('figures', 'fig_method_5.png'))

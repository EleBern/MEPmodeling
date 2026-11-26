import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from MEPmodel_bio import MEPmodel_bio
from h5_helpers import load_h5_to_dict
from config_model_bio import config_model_bio

subject = "1"
input_dir = os.path.join("/home/bernasconi/Documents/Programma/fparam_distribution", "NoNoise", "SyntheticMUAPs", "S"+subject) 
batch = os.listdir(input_dir)

for i in [40]: # choose a random optimization run
    file_path = os.path.join(input_dir, batch[i], "result_bio_s{0}.h5".format(subject))
    with h5py.File(file_path, 'r') as f:
        tmp = load_h5_to_dict(f)
    p = tmp['p_post'].flatten()
    R2_original = tmp['ref']['R2']

def param_sweep(ref, param_name, param_index, param_range):
    R2 = np.zeros(101)
    changing_param = np.linspace(param_range[0], param_range[1], 101)
    changing_param[-1] = p[param_index]
    for i in range(101):
        p[param_index] = changing_param[i] # p[7] is I1
        _, ref = MEPmodel_bio(p, ref, 0)
        R2[i] = ref["R2"]


    plt.figure()
    sorted_indices = np.argsort(changing_param)
    plt.plot(changing_param[sorted_indices], R2[sorted_indices])
    plt.plot(changing_param[-1], R2[-1], "r*")
    plt.title("Goodness of fit")
    plt.ylabel("R²")
    plt.xlabel(param_name)
    plt.xlim([param_range[0], param_range[1]])
    plt.savefig(os.path.join(input_dir, param_name+".png"))
    plt.show()

    # Sanity check
    print(R2_original)
    print(R2[-1])

param_sweep(tmp["ref"], "I1", 7, [0, 10])
param_sweep(tmp["ref"], "I2", 8, [0, 10])
param_sweep(tmp["ref"], "E1", 5, [0, 20])
param_sweep(tmp["ref"], "E2", 6, [0, 20])

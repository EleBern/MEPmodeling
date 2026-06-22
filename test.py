from GA.ga_toolbox.selection_uniq import selection_uniq
import numpy as np
import scipy.io

P1 = scipy.io.loadmat("P1.mat")
P1=P1["P1"]
B = scipy.io.loadmat("B.mat")
B=B["B"].ravel(order="F")
LR = np.zeros(12)
LR[-2] = 5
LR[-3] = 1
UR = np.ones(12) *10
UR[-1] = 1
UR[5] = 20
UR[6] = 20
selection_uniq(P1, B, 60, 60, -1, LR, UR)
#############################################################
# from load_muap import load_muap
# load_muap(1)
#############################################################
# from load_MEP import load_MEP
# load_MEP("5LM")
#############################################################
# from Load_EMG_data_example import plot_emg
# plot_emg("data_diLazarro/DiLazarro_di_wave_data_by_year_merged.hdf5", "PA", None, 2013)
# plot_emg("data_diLazarro/DiLazzaro_di_wave_data_down.hdf5", "PA", None, 2013)
#############################################################
# import matplotlib.pyplot as plt
# from ga_MEPmodel_bio import ga_MEPmodel_bio
# #ga_MEPmodel_bio("5LM", 1, [] , 0)

# for i in range(20):
#     print("############################################")
#     print("This is the ", i, " iteration")
#     ga_MEPmodel_bio("2PA", 1, [] , 1)
#     plt.close("all")
    
#############################################################
# from gen_DIwave import gen_DIwave

##########################################################################
# Phenomenological model
# from config_model_pheno import config_model_pheno
# config_model_pheno(1)


# from ga_MEPmodel_pheno import ga_MEPmodel_pheno
# ga_MEPmodel_pheno("5PA", 1)
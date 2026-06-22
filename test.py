from GA.gradient_toolbox.selection_best import selection_best
import numpy as np
import scipy.io

R=scipy.io.loadmat("R")
R=R["R"]
P=scipy.io.loadmat("P")
P=P["P"]
E=scipy.io.loadmat("E")
E=E["E"]
selection_best(P, E, R, 60, -1)
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
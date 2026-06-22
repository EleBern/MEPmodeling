from GA.ga_toolbox.mutationV import mutationV
import numpy as np
import scipy.io


lowchance = 0.1
highchance = 0.9
LR = np.zeros(12)
LR[-2]= 5
LR[-3]= 1
UR = np.ones(12)*10
UR[-1]=1
UR[5]=20
UR[6]=20
P = scipy.io.loadmat("P.mat")
P=P["P"]
mutationV(P, lowchance, highchance, LR, UR)
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
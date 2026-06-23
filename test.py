from GA.gradient_toolbox.multi_lavenberg_regulization import multi_lavenberg_regulization
import numpy as np
import scipy.io

Para_E = np.array([1.0079,2.2095,5.1515,0.9768,6.7231,10.1484,0.8018,0.4232,0.0346,3.3845,5.0966,0.1311])
h_output = scipy.io.loadmat("h_output.mat")
h_output=h_output["h_output"]
J = scipy.io.loadmat("J.mat")
J=J["J"]
LR = np.zeros(12)
LR[-2] = 5
LR[-3] = 1
UR = np.ones(12) *10
UR[-1] = 1
UR[5] = 20
UR[6] = 20
multi_lavenberg_regulization(25, -12, 12, Para_E, J, h_output, LR, UR)
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
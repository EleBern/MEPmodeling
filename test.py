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

from ga_MEPmodel_bio import ga_MEPmodel_bio
#for i in range(10):
ga_MEPmodel_bio("5PA", 1, [] , 1)

#############################################################
# from gen_DIwave import gen_DIwave

##########################################################################
# Phenomenological model
# from config_model_pheno import config_model_pheno
# config_model_pheno(1)


# from ga_MEPmodel_pheno import ga_MEPmodel_pheno
# ga_MEPmodel_pheno("5PA", 1)
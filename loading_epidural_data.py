from preprocess_di_wave import preprocess_di_wave

save_figs = False

hdf5_path = "/home/bernasconi/Documents/Programma/MEP_Python/MEPmodeling/data_diLazarro/DiLazarro_di_wave_data.hdf5"

# PA data
measurement_dict_2020_140_PA_ch3 = dict(orientation='PA', threshold=140, year=2020, sigma=1.0)
measurement_dict_2020_120_PA_ch3 = dict(orientation='PA', threshold=120, year=2020, sigma=1.0)
measurement_dict_2020_100_PA_ch3 = dict(orientation='PA', threshold=100, year=2020, sigma=1.0)
measurement_dict_2013_110_PA_ch2 = dict(orientation='PA', threshold=110, year=2013, sigma=1.0, channel=0)
measurement_dict_2013_110_PA_ch3 = dict(orientation='PA', threshold=110, year=2013, sigma=1.0, channel=1)
measurement_dict_2007_120_PA_ch3 = dict(orientation='PA', threshold=120, year=2007, sigma=1.0)
measurement_dict_2007_150_PA_ch3 = dict(orientation='PA', threshold=150, year=2007, sigma=1.0)
measurement_dict_2004_154_PA_2_ch2 = dict(orientation='PA', threshold=154, year=2004, sigma=1.0)
measurement_dict_2004_146_PA_2_ch2 = dict(orientation='PA', threshold=146, year=2004, sigma=1.0)
measurement_dict_2004_150_PA_1_ch2 = dict(orientation='PA', threshold=150, year=2004, sigma=0.1)

# LM data
measurement_dict_2020_80_LM_ch3  = dict(orientation='LM', threshold=80,  year=2020, sigma=1.0, channel=0)
measurement_dict_2020_80_LM_ch4  = dict(orientation='LM', threshold=80,  year=2020, sigma=1.0, channel=1)
measurement_dict_2020_100_LM_ch3 = dict(orientation='LM', threshold=100, year=2020, sigma=1.0, channel=0)
measurement_dict_2020_100_LM_ch4 = dict(orientation='LM', threshold=100, year=2020, sigma=1.0, channel=1)
measurement_dict_2020_120_LM_ch3 = dict(orientation='LM', threshold=120, year=2020, sigma=1.0, channel=0)
measurement_dict_2020_120_LM_ch4 = dict(orientation='LM', threshold=120, year=2020, sigma=1.0, channel=1)
measurement_dict_2013_100_LM_ch2 = dict(orientation='LM', threshold=100, year=2013, sigma=1.0, channel=0)
measurement_dict_2013_100_LM_ch3 = dict(orientation='LM', threshold=100, year=2013, sigma=1.0, channel=1)
measurement_dict_2007_120_LM_ch2 = dict(orientation='LM', threshold=120, year=2007, sigma=1.0, channel=0)
measurement_dict_2004_140_LM_1_ch2 = dict(orientation='LM', threshold=140, year=2004, sigma=0.1, channel=0)

data_dicts_PA = [
    measurement_dict_2020_140_PA_ch3,
    measurement_dict_2020_120_PA_ch3,
    measurement_dict_2020_100_PA_ch3,
    measurement_dict_2013_110_PA_ch2,
    measurement_dict_2013_110_PA_ch3,
    measurement_dict_2007_120_PA_ch3,
    measurement_dict_2007_150_PA_ch3,
    measurement_dict_2004_154_PA_2_ch2,
    measurement_dict_2004_146_PA_2_ch2,
    measurement_dict_2004_150_PA_1_ch2,
]
data_dicts_LM = [
    measurement_dict_2020_120_LM_ch3,
    measurement_dict_2020_120_LM_ch4,
    measurement_dict_2004_140_LM_1_ch2,
]

for file_args in data_dicts_LM:
    target, t = preprocess_di_wave(
        hdf5_path=hdf5_path,
        file_args=file_args,
        from_file=True,
        enable_high_pass=True,
        detrend_signal=False,
        plot=False,
        plot_d_wave_detection=True,
    )
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import norm

from gen_muaps import gen_muaps
from zero_crossing import crossing_times
from downsample_muaps import load_unprocessed_muaps    
from helper_f_genmuaps import fit_lam, gof, amplitude_distribution, exponential

popt = {}
amplitude = {}
lam = {}
mean_lam = {}
axonalDelay = {}
mean_axonalDelay = {}
RMSE = {}
mean_RMSE = {}
R2 = {}
mean_R2 = {}

root    = os.path.dirname(os.path.realpath(__file__))
for i in range(1,6):
    h5_path = os.path.join(root, "data_MUAP", "Dist{0}_Monopolar_Rest_NormalCV_New.hdf5".format(i))

    t, anatomical_muaps, downsampled_t, downsampled_muaps = load_unprocessed_muaps(h5_path)

    # Calculate the amplitude and amplitude distribution of the anatomical MUAPs
    popt[i], amplitude[i] = amplitude_distribution(anatomical_muaps)

    # Find the zero-crossing of the anatomical MUAPs
    axonalDelay[i] = 2 * crossing_times(t, anatomical_muaps)
    mean_axonalDelay[i] = np.nanmean(axonalDelay[i])

    # Fit one lambda per anatomical MUAP (100 MUAPs -> 100 lambdas)
    lam[i] = fit_lam(downsampled_muaps, amplitude[i], axonalDelay[i])
    mean_lam[i] = np.nanmean(lam[i])

    # Generate synthetic MUAPs
    zero_muaps = np.all(anatomical_muaps == 0, axis=0)
    axonalDelay[i][zero_muaps] = np.nanmean(axonalDelay[i][~zero_muaps])
    zero_muaps = None
    muaps, _ = gen_muaps(100, amplitude[i], axonalDelay[i], lam[i], zero_muaps=zero_muaps)

    # Calculate goodness of fit
    R2[i], RMSE[i] = gof(downsampled_muaps, muaps)
    mean_R2[i] = np.nanmean(R2[i])
    mean_RMSE[i] = np.nanmean(RMSE[i])

# Lower and upper bounds for the amplitude distribution
k = list(amplitude.keys())
total_a = np.zeros((len(amplitude[k[0]]), len(k)))
for i in range(len(k)):
    total_a[:, i] = amplitude[k[i]]

lower_a = 1e6 * np.min(total_a, axis=1)
print(np.shape(lower_a))
upper_a = 1e6 * np.max(total_a, axis=1)
print(np.shape(upper_a))

x = np.arange(len(lower_a)) / (len(lower_a)-1)
popt_lower, _ = curve_fit(exponential, x, np.abs(lower_a), p0=[1e-5, 100])
popt_upper, _ = curve_fit(exponential, x, np.abs(upper_a), p0=[1e-5, 100])
print("Lower bounds exponential parameters: ", popt_lower)
print("Upper bounds exponential parameters: ", popt_upper)


# Plots
fig = plt.figure()
plt.plot(np.arange(100), 1e6 * amplitude[1], "*r", label="1: a={0}, b={1}".format(np.round(1e6 * popt[1][0], 2), np.round(popt[1][1], 2)))
plt.plot(np.arange(100), 1e6 * amplitude[1], "--r", linewidth=0.5)
plt.plot(np.arange(100), 1e6 * amplitude[2], "*k", label="2: a={0}, b={1}".format(np.round(1e6 * popt[2][0], 2), np.round(popt[2][1], 2)))
plt.plot(np.arange(100), 1e6 * amplitude[2], "--k", linewidth=0.5)
plt.plot(np.arange(100), 1e6 * amplitude[3], "*g", label="3: a={0}, b={1}".format(np.round(1e6 * popt[3][0], 2), np.round(popt[3][1], 2)))
plt.plot(np.arange(100), 1e6 * amplitude[3], "--g", linewidth=0.5)
plt.plot(np.arange(100), 1e6 * amplitude[4], "*b", label="4: a={0}, b={1}".format(np.round(1e6 * popt[4][0], 2), np.round(popt[4][1], 2)))
plt.plot(np.arange(100), 1e6 * amplitude[4], "--b", linewidth=0.5)
plt.plot(np.arange(100), 1e6 * amplitude[5], "*m", label="5: a={0}, b={1}".format(np.round(1e6 * popt[5][0], 2), np.round(popt[5][1], 2)))
plt.plot(np.arange(100), 1e6 * amplitude[5], "--m", linewidth=0.5)
plt.plot(np.arange(len(lower_a)), exponential(np.arange(len(lower_a)) / (len(lower_a)-1), *popt_lower), "r--", label="Lower bound")
plt.plot(np.arange(len(upper_a)), exponential(np.arange(len(upper_a)) / (len(upper_a)-1), *popt_upper), "k--", label="Upper bound")
plt.xlim([0, 100])
plt.xlabel("MUAP ID")
plt.ylabel("Amplitude [uV]")
plt.title("MUAP Amplitude")
plt.legend()
plt.show()

fig = plt.figure()
plt.plot(np.arange(100), axonalDelay[1], "*r", label="1: {0} ms".format(np.round(mean_axonalDelay[1], 2)))
plt.plot(np.arange(100), axonalDelay[1], "--r", linewidth=0.5)
plt.plot(np.arange(100), axonalDelay[2], "*k", label="2: {0} ms".format(np.round(mean_axonalDelay[2], 2)))
plt.plot(np.arange(100), axonalDelay[2], "--k", linewidth=0.5)
plt.plot(np.arange(100), axonalDelay[3], "*g", label="3: {0} ms".format(np.round(mean_axonalDelay[3], 2)))
plt.plot(np.arange(100), axonalDelay[3], "--g", linewidth=0.5)
plt.plot(np.arange(100), axonalDelay[4], "*b", label="4: {0} ms".format(np.round(mean_axonalDelay[4], 2)))
plt.plot(np.arange(100), axonalDelay[4], "--b", linewidth=0.5)
plt.plot(np.arange(100), axonalDelay[5], "*m", label="5: {0} ms".format(np.round(mean_axonalDelay[5], 2)))
plt.plot(np.arange(100), axonalDelay[5], "--m", linewidth=0.5)
plt.xlim([0, 100])
plt.xlabel("MUAP ID")
plt.ylabel("Time [ms]")
plt.title("Zero-crossing times")
plt.legend()
plt.show()


fig = plt.figure()
plt.plot(np.arange(100), lam[1], "*r", label="1: {0} ms".format(np.round(mean_lam[1], 2)))
plt.plot(np.arange(100), lam[1], "--r", linewidth=0.5)
plt.plot(np.arange(100), lam[2], "*k", label="2: {0} ms".format(np.round(mean_lam[2], 2)))
plt.plot(np.arange(100), lam[2], "--k", linewidth=0.5)
plt.plot(np.arange(100), lam[3], "*g", label="3: {0} ms".format(np.round(mean_lam[3], 2)))
plt.plot(np.arange(100), lam[3], "--g", linewidth=0.5)
plt.plot(np.arange(100), lam[4], "*b", label="4: {0} ms".format(np.round(mean_lam[4], 2)))
plt.plot(np.arange(100), lam[4], "--b", linewidth=0.5)
plt.plot(np.arange(100), lam[5], "*m", label="5: {0} ms".format(np.round(mean_lam[5], 2)))
plt.plot(np.arange(100), lam[5], "--m", linewidth=0.5)
plt.xlim([0, 100])
plt.xlabel("MUAP ID")
plt.ylabel("Lambda [ms]")
plt.title("Width of MUAPs")
plt.legend()
plt.show()


fig = plt.figure()
plt.plot(np.arange(100), R2[1], "*r", label="1: {0}".format(np.round(mean_R2[1], 2)))
plt.plot(np.arange(100), R2[1], "--r", linewidth=0.5)
plt.plot(np.arange(100), R2[2], "*k", label="2: {0}".format(np.round(mean_R2[2], 2)))
plt.plot(np.arange(100), R2[2], "--k", linewidth=0.5)
plt.plot(np.arange(100), R2[3], "*g", label="3: {0}".format(np.round(mean_R2[3], 2)))
plt.plot(np.arange(100), R2[3], "--g", linewidth=0.5)
plt.plot(np.arange(100), R2[4], "*b", label="4: {0}".format(np.round(mean_R2[4], 2)))
plt.plot(np.arange(100), R2[4], "--b", linewidth=0.5)
plt.plot(np.arange(100), R2[5], "*m", label="5: {0}".format(np.round(mean_R2[5], 2)))
plt.plot(np.arange(100), R2[5], "--m", linewidth=0.5)
plt.xlim([0, 100])
plt.xlabel("MUAP ID")
plt.ylabel("R2")
plt.title("Goodness of fit")
plt.legend()
plt.show()


fig = plt.figure()
plt.plot(np.arange(100), RMSE[1], "*r", label="1: {0}".format(np.round(mean_RMSE[1], 3)))
plt.plot(np.arange(100), RMSE[1], "--r", linewidth=0.5)
plt.plot(np.arange(100), RMSE[2], "*k", label="2: {0}".format(np.round(mean_RMSE[2], 3)))
plt.plot(np.arange(100), RMSE[2], "--k", linewidth=0.5)
plt.plot(np.arange(100), RMSE[3], "*g", label="3: {0}".format(np.round(mean_RMSE[3], 3)))
plt.plot(np.arange(100), RMSE[3], "--g", linewidth=0.5)
plt.plot(np.arange(100), RMSE[4], "*b", label="4: {0}".format(np.round(mean_RMSE[4], 3)))
plt.plot(np.arange(100), RMSE[4], "--b", linewidth=0.5)
plt.plot(np.arange(100), RMSE[5], "*m", label="5: {0}".format(np.round(mean_RMSE[5], 3)))
plt.plot(np.arange(100), RMSE[5], "--m", linewidth=0.5)
plt.xlim([0, 100])
plt.xlabel("MUAP ID")
plt.ylabel("Normalised RMSE")
plt.title("Goodness of fit")
plt.legend()
plt.show()


# Calcualte and plot distributions
k = list(lam.keys())
total_lam = np.zeros((len(lam[k[0]]), len(k)))
for i in range(len(k)):
    total_lam[:, i] = lam[k[i]]
total_lam = total_lam.ravel()
lam_mu, lam_std = norm.fit(total_lam)
print("Lambda boudaries: {0} - {1}".format(np.round(min(total_lam), 3), np.round(max(total_lam), 3)))
print("Mean and std of the normal distribution of lambdas: {0}, {1}".format(np.round(lam_mu, 3), np.round(lam_std, 3)))

plt.figure()
plt.hist(total_lam, bins=15, color="blue", edgecolor="blue", density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, lam_mu, lam_std)
plt.plot(x, p, 'r')
plt.xlabel("Lambda")
plt.ylabel("Frequency")
plt.title("Distribution of lambda")
plt.show()

k = list(axonalDelay.keys())
total_axonalDelay = np.zeros((len(axonalDelay[k[0]]), len(k)))
for i in range(len(k)):
    total_axonalDelay[:, i] = axonalDelay[k[i]]
total_axonalDelay = total_axonalDelay.ravel()
axonalDelay_mu, axonalDelay_std = norm.fit(total_axonalDelay)
print("Delay boudaries: {0} - {1}".format(np.round(min(total_axonalDelay), 3), np.round(max(total_axonalDelay), 3)))
print("Mean and std of the normal distribution of delays: {0}, {1}".format(np.round(axonalDelay_mu, 3), np.round(axonalDelay_std, 3)))


plt.figure()
plt.hist(total_axonalDelay, bins=20, color="blue", edgecolor="blue", density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, axonalDelay_mu, axonalDelay_std)
plt.plot(x, p, 'r')
plt.xlabel("Delay")
plt.ylabel("Frequency")
plt.title("Distribution of the delay times")
plt.show()
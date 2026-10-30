"""
MEPmodel.py
===========

Biological spinal peripheral model for TMS-induced MEPs.

All data files (h5) are read from the folder `data` next to this script.

The spinal network is NOT re-simulated during the pygpc run.  The MN firing
times fitted earlier are loaded once from disk; every pygpc sample only
regenerates the MUAPs (b, lam) and superposes them at those firing times.
"""

import os
import h5py
import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


class MEPmodel:

    # Target MEP (ref) + saved MN firing times, shared by all instances
    # (equivalent to the module-level _CACHE of pygpc_main.py)
    _CACHE = {}

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
        self.data_dir = data_dir

    # ======================================================================
    # pygpc interface
    # ======================================================================
    def get_setup(self, subj, withRC, AMPAweight, spike_file):
        """
        Target MEP (ref) + previously saved MN firing times.  Built on the first
        call, cached afterwards.
        """
        spike_file = spike_file or os.path.join(self.data_dir, f'mu_spiketimes_S{subj}.h5')
        key = (subj, withRC, spike_file)

        if key not in MEPmodel._CACHE:
            ref = self.config_model_bio(subj, withRC, AMPAweight)

            if not os.path.isfile(spike_file):
                raise FileNotFoundError(
                    f'Saved spike times not found: {spike_file}\n'
                    'Run the fitted model once (MEPmodel_bio_core) to generate it.'
                )
            with h5py.File(spike_file, 'r') as f:
                spike_times = np.asarray(self.load_h5_to_dict(f)['spike_times'], dtype=float)

            nIntensities = ref['y0'].shape[1]
            if spike_times.shape[2] != nIntensities:
                raise ValueError(
                    f'{spike_file} holds {spike_times.shape[2]} TMS intensities, '
                    f'but subject {subj} has {nIntensities}.'
                )

            MEPmodel._CACHE[key] = (ref, spike_times)

        return MEPmodel._CACHE[key]

    # ----------------------------------------------------------------------
    def run_model(self, b, lam, subj, withRC, AMPAweight, spike_file):
        """
        Call to the biological spinal periferal model.

        b : float
            MUAP amplitude distribution parameter, A = a*b^x. a is hard coded to 6.28
        lam  : float
            MUAP shape parameter (with zero crossing = 2.35 * lam - 0.18)
        subj : int
            the subject to fit
        withRC: int
            0 for the model without Renshaw cells, 1 for the model with Renshaw cells
        AMPAweight: list of floats
            [] to select the best value
        spike_file: str
            the name of the file containing the MN spike times

        Returns
        -------
        R2 : float
            Goodness of fit of the simulated vs. measured MEPs
        """
        ref, spike_times = self.get_setup(subj, withRC, AMPAweight, spike_file)

        # ----- generate MUAPs for this sample -----
        delay = np.ones(100) * (2.35 * lam - 0.18)                                   # fixed parameter
        muaps, tmuap = self.gen_muaps(n_neurons=100,
                                      amplitude=[6.28, b],                           # a fixed to 6.28
                                      axonalDelay=delay,
                                      lam=lam)

        ref['model']['muaps'] = muaps
        ref['model']['tmuap'] = tmuap

        # ----- rebuild MEP from the saved spike times (no network simulation) -----
        sim = self.MEPmodel_bio_core(ref['model'], spike_times=spike_times)

        # ----- output metric -----
        ref = self.cal_error(ref, sim)

        return ref['R2']

    # ======================================================================
    # Model configuration (config_model_bio.py)
    # ======================================================================
    def config_model_bio(self, subj, withRC, AMPAweight=[]):

        model = {}
        model["AMPAweight"] = AMPAweight
        model["withRC"] = withRC
        model["maxES"] = 5  # take first 5 spikes

        # -----subject MEP (target) -----
        ref = {}
        ref["subj"] = subj
        ref["tcrop"] = [20, 50]
        if ref["subj"] == 10:
            intensity_idx = np.arange(6) # remove 50% intensity as no MEP is elicited
        else:
            intensity_idx = None
        y0, t0, _, intensities, _, _ = self.load_MEP(subj, intensity_idx, ref["tcrop"])
        ref["intensities"] = intensities
        intensity_idx = np.arange(len(intensities))
        ref["intensity_idx"] = intensity_idx
        ref["t0"] = t0
        ref["y0"] = y0.T

        # -----subject RMT ---------
        # (by eyeballing the IO curve)
        if ref["subj"] == 1 or ref["subj"] == 8:
            ref["RMT"] = 32
        elif ref["subj"] == 2 or ref["subj"] == 10:
            ref["RMT"] = 35
        elif ref["subj"] == 3 or ref["subj"] == 9:
            ref["RMT"] = 38
        elif ref["subj"] == 4:
            ref["RMT"] = 41
        elif ref["subj"] == 5 or ref["subj"] == 6:
            ref["RMT"] = 47
        else:
            ref["RMT"] = 44

        # ----- simulated DI wave -----
        tlength = t0[-1] # ms
        dt = 0.1 # ms
        t = np.arange(0, tlength, dt)
        model["DIwave0"] = np.zeros((len(ref["intensities"]), len(t)))
        for i in range(len(ref["intensities"])):
            model["DIwave0"][i,:] = self.gen_DIwave(t, ref["intensities"][i] / ref["RMT"])
        model["DIwave"] = self.deconv_DIwave(t, model["DIwave0"], ref)

        # ----- AMPA, NMDA kernels -----
        model["AMPA"], model["NMDA"] = self.gen_kernels(dt, tlength)

        # ----- AchR, GlyR kernels -----
        if "kernel" not in model.keys():
            model["kernel"] = {}
        model["kernel"]["tau"] = [[0.5, 3.6], [1.8, 20.2], [1, 6]]
        model["kernel"]["h"] = [[1.5977], [1.3908], [1.7175]]

        # ----- Motor neuron -----
        model["tauLIF"] = 10 # ms
        model["Eexc"] = 0 # mV
        model["Einh"] = -75 # mV
        model["V_rest"] = -65 # mV
        model["V_thr"] = -55 # mV
        model["T_ref"] = 2 # ms, refractory period
        model ["fastAChRweight"] = 0.5 # AchR (MN->RC)

        # ----- tDIwave (input) -----
        model["DIwaveConv"] = model["DIwave"]
        model["DIwaveConv_AMPA"] = np.zeros((len(ref["intensities"]), len(t)))
        model["DIwaveConv_NMDA"] = np.zeros((len(ref["intensities"]), len(t)))
        fr = 1
        for i in range (len(ref["intensities"])):
            g = np.convolve(fr * model["DIwave"][i,:], model["AMPA"])
            model["DIwaveConv_AMPA"][i,:] = g[:len(t)]
            g = np.convolve(fr * model["DIwave"][i,:], model["NMDA"])
            model["DIwaveConv_NMDA"][i,:] = g[:len(t)]
        model["t"] = t
        model["dt"] = dt

        # ----- search boundary-----
        if withRC == 0:
            nParams = 7
            boundary = np.zeros((nParams, 2)) # [lower, upper]
            boundary[0, :] = np.array([0, 10]) # R1
            boundary[1, :] = np.array([0, 10]) # R2
            boundary[2, :] = np.array([0, 10]) # R3
            boundary[3, :] = np.array([0, 10]) # R4
            boundary[4, :] = np.array([0, 10]) # R5
            boundary[5, :] = np.array([5, 10]) # Tmuap
            boundary[6, :] = np.array([0, 1]) # AMPAweight
            if AMPAweight is not None and len(np.atleast_1d(AMPAweight)) > 0:
                boundary[6, :] = np.array([1, 1]) * np.atleast_1d(AMPAweight) # fixed value
            model["boundarytext"] = ['R1','R2','R3','R4','R5','MU.T','AMPAw']
        else:
            nParams = 12
            boundary = np.zeros((nParams, 2)) # [lower, upper]
            boundary[0, :] = np.array([0, 10]) # R1
            boundary[1, :] = np.array([0, 10]) # R2
            boundary[2, :] = np.array([0, 10]) # R3
            boundary[3, :] = np.array([0, 10]) # R4
            boundary[4, :] = np.array([0, 10]) # R5
            boundary[5, :] = np.array([0, 20]) # E1 (MN1 -> RC)
            boundary[6, :] = np.array([0, 20]) # E2 (MN100 -> RC)
            boundary[7, :] = np.array([0, 10]) # I1 (RC -> MN1)
            boundary[8, :] = np.array([0, 10]) # I2 (RC -> MN100)
            boundary[9, :] = np.array([1, 10]) # RC.th
            boundary[10, :] = np.array([5, 10]) # Tmuap
            boundary[11, :] = np.array([0, 1]) # AMPAweight
            if AMPAweight is not None and len(np.atleast_1d(AMPAweight)) > 0:
                boundary[11, :] = np.array([1, 1]) * np.atleast_1d(AMPAweight) # fixed value
            model["boundarytext"] = ['R1','R2','R3','R4','R5','E1','E2','I1','I2','RC.th','MU.T','AMPAw']

        model["boundary"] = boundary
        ref["model"] = model

        if withRC == 1:
            if AMPAweight is None or len(np.atleast_1d(AMPAweight)) == 0:
                # Free fit — no fixed AMPAweight
                ref["resultname"] = os.path.join(self.data_dir, f"result_bio_s{subj}.h5")
            else:
                # Fixed AMPAweight provided — encode each value in the filename
                ampa_tag = '_'.join(f'{v:g}' for v in np.atleast_1d(AMPAweight))
                ref["resultname"] = os.path.join(self.data_dir, f"result_bio_s{subj}[{ampa_tag}].h5")
        else:
                ref["resultname"] = os.path.join(self.data_dir, f"result_bioNoRC_s{subj}.h5")
        ref["figname"] = ref["resultname"][:-3]+".svg"

        return ref

    # ======================================================================
    # Data loading (load_MEP.py, h5_helpers.py)
    # ======================================================================
    def load_MEP(self, subj, iidx=None, tcrop=[20, 50]):
        TSTIM = 100  # TMS stimulus was at t=100 ms

        # ----- load data -----
        mat_path = os.path.join(self.data_dir, f"S{subj}.h5")

        with h5py.File(mat_path, 'r') as f:
            tmp = self.load_h5_to_dict(f)

        mep = tmp["mep"]                 # shape: [N x time x trials]
        intensities = tmp["intensities"].flatten()
        times = tmp["t"].flatten() - TSTIM

        # ----- select intensities -----
        if iidx is None:
            iidx = np.arange(len(intensities))

        mep = mep[iidx, :, :]
        intensities = intensities[iidx]

        # ----- crop -----
        tidx = np.where((times >= tcrop[0]) & (times < tcrop[1]))[0]
        t = times[tidx]

        yall = mep[:, tidx, :]
        y = np.mean(yall, axis=2)

        # ----- remove baseline [-20, -10] ms -----
        baseline_idx = np.where((times >= -20) & (times < -10))[0]

        baseline = np.mean(
            np.mean(mep[:, baseline_idx, :], axis=2),
            axis=1
        )  # shape: [n]

        # subtract baseline
        y = y - baseline[:, np.newaxis]
        mep = mep - baseline[:, np.newaxis, np.newaxis]

        return y, t, mep, intensities, times, yall

    # ----------------------------------------------------------------------
    @staticmethod
    def load_h5_to_dict(group):
        """Recursively loads an H5 group back into a dictionary."""
        data = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                data[key] = MEPmodel.load_h5_to_dict(item)
            else:
                value = item[()]  # [()] extracts the data as a numpy array/scalar
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()
                data[key] = value
        return data

    # ======================================================================
    # DI-wave generation (gen_DIwave.py)
    # ======================================================================
    def gen_DIwave(self, t, intensity):
        t0 = 5
        T = 1.5
        width = 0.25

        DIwave = np.zeros(np.size(t))

        # D, I1, I2, I3, I4
        x0 = [1.36192637, 1.04127548, 1.16603639, 1.03733872, 1.45405986]
        r =  [18.50774852, 9.26210842, 5.91559859, 17.7805388, 425.51252596]
        a =  [0.34532065, 1.0, 0.80577286, 0.46054753, 0.27828232]

        for i in range(5):
            DIwave = DIwave + np.exp(-(t-t0 -i * T)**2 /2 /width**2) * self.sigmoid(intensity, x0[i], r[i], a[i])
    
        return DIwave

    # ======================================================================
    # DI-wave deconvolution (deconv_DIwave.py, generate_EP.py)
    # ======================================================================
    @staticmethod
    def deconvreg(signal: np.ndarray, psf: np.ndarray, lam: float = 100.0) -> np.ndarray:
        """
        1-D regularised deconvolution (Python equivalent of MATLAB deconvreg).

        Parameters
        ----------
        signal : array_like, shape (N,)
            Observed / blurred signal (DIwave2 in the original code).
        psf : array_like, shape (M,)
            Point-spread function / kernel (EP).  If M < N it is zero-padded
            to length N before the FFT; if M > N it is truncated.
        lam : float, optional
            Regularisation parameter (lambda in MATLAB).  Default is 100.

        Returns
        -------
        rate : ndarray, shape (N,)
            Deconvolved (restored) signal, real-valued.

        Notes
        -----
        The formula applied is Tikhonov regularisation with a second-difference
        (Laplacian) constraint matrix C:

            F = conj(H) * G / (|H|² + lambda * C²)

        where G = FFT(signal), H = FFT(psf_padded), and
        C(k) = 2 − 2·cos(2π k / N)  (eigenvalues of the 1-D discrete Laplacian).

        This matches MATLAB's default deconvreg behaviour to floating-point
        precision (relative error < 1e-12 on test data).
        """
        signal = np.asarray(signal, dtype=float)
        psf    = np.asarray(psf,    dtype=float)

        N = len(signal)

        # Zero-pad (or truncate) PSF to the length of the signal
        psf_padded = np.zeros(N)
        m = min(len(psf), N)
        psf_padded[:m] = psf[:m]

        # Optionally apply psf2otf-style centering: pad FIRST, then roll.
        # This replicates MATLAB's psf2otf(), which shifts the already-padded
        # array so that the PSF centre lands at index 0.
        if True:
            center = len(psf) // 2
            psf_padded = np.roll(psf_padded, -center)

        # Frequency-domain representations
        G = fft(signal)
        H = fft(psf_padded)

        # Eigenvalues of the 1-D discrete Laplacian (constraint matrix)
        k = np.arange(N)
        C = 2.0 - 2.0 * np.cos(2.0 * np.pi * k / N)

        # Tikhonov regularised inverse filter
        F = (np.conj(H) * G) / (np.abs(H) ** 2 + lam * C ** 2)

        # Back to time domain -- result should be real
        rate = np.real(ifft(F))
        return rate

    # ----------------------------------------------------------------------
    def deconv_DIwave(self, times: np.ndarray, DIwave: np.ndarray, ref: dict) -> np.ndarray:
        """
        Python translation of the MATLAB function deconv_DIwave.

        Parameters
        ----------
        times   : 1-D array, shape (501,)      — time axis in ms (0–50 ms).
        DIwave  : 2-D array, shape (10, 501)   — input waveforms.
        ref     : dict with optional keys:
                    'EP' — pre-computed EP waveform (1-D array)
                    'dt' — time step of EP (float, ms)

        Returns
        -------
        DIwave_deconv : 2-D array, shape (10, 501) — deconvolved, normalised waveforms.
        """

        DIwave_deconv = np.zeros(DIwave.shape)

        # --------- EP ------------
        if "EP" not in ref:
            d = 0.1
            EP, t, _ = self.generate_EP(d)
            dt = t[1] - t[0]
            EP = -EP
        else:
            EP = ref["EP"]
            dt = ref["dt"]

        # -------- DIwave -------------
        times2 = np.arange(times[0], times[-1], dt)
        DIwave3 = np.zeros((10, len(times2)))
        rate1 = np.zeros((10, len(times2)))

        for i in range(DIwave.shape[0]):    # number of intensities

            # Resample DIwave[i] onto a uniform grid with step dt.
            # np.arange mirrors MATLAB's times(1):dt:times(end).
            times2 = np.arange(times[0], times[-1], dt)   # ms

            valid = ~np.isnan(DIwave[i])
            f_interp = interp1d(times[valid], DIwave[i][valid],   # ← no NaN knots
                        kind='linear',
                        bounds_error=False,
                        fill_value=np.nan)
            DIwave2 = f_interp(times2)
            DIwave3[i, :] = DIwave2


            # -------- deconv(DIwave, EP) -----------
            lambda_ = 1e9
            # MATLAB: deconvreg(DIwave2', EP, lambda)
            rate = self.deconvreg(DIwave2.T, EP, lambda_)
            rate1[i, :] = rate

            # Interpolate rate back onto the original time axis.
            # NaN outside bounds, matching MATLAB interp1 default.

            f_back = interp1d(
                times2, rate,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            DIwave_deconv[i, :] = f_back(times)

        # Normalise by the global maximum (matches MATLAB's max(DIwave_deconv(:)))
        DIwave_deconv = DIwave_deconv / np.nanmax(DIwave_deconv)
        return DIwave_deconv

    # ----------------------------------------------------------------------
    def generate_EP(self, d=0.01, Axontype=1):
        # ----------load AP (by Arancibia-Carcamo model)
        if Axontype == 1:
            data_path = os.path.join(self.data_dir, 'AP.h5')
        else:
            data_path = os.path.join(self.data_dir, 'AP2.h5')

        # Load the file
        with h5py.File(data_path, 'r') as f:
            mat_contents = self.load_h5_to_dict(f)
        times = mat_contents['TIME_VECTOR']
        times = times.reshape((len(times), 1))
        MEMBRANE_POTENTIAL = mat_contents['MEMBRANE_POTENTIAL']

        # ----------------------------------------------------
        idx = 20
        padding = 1000
        dt = times[1] - times[0]

        # Append padding to times
        padding_times = times[-1] + (np.arange(1, padding + 1) * dt)
        padding_times = padding_times.reshape((len(padding_times), 1))
        times = np.concatenate((times, padding_times))
        # Extract specific AP and add padding
        v_segment = MEMBRANE_POTENTIAL[:, idx]
        v_segment = v_segment.reshape(-1, 1)
        v_padding = np.ones(padding) * MEMBRANE_POTENTIAL[-1, idx]
        v_padding = v_padding.reshape(-1, 1)
        v = np.concatenate((v_segment, v_padding))

        # Calculate derivatives
        if isinstance(idx, int):
            s = 1
        else:
            s = len(idx)
        dv = np.vstack((np.diff(v, axis=0), np.zeros((1, s))))
        ddv = np.vstack((np.diff(dv, axis=0), np.zeros((1, s))))

        # ---------calculate extracellular potential------------
        EP = np.zeros(ddv.shape)
        for i in range(v.shape[1]):
            for t_idx in range(len(times)):
                mask = 1.0 / np.sqrt((times - times[t_idx])**2 + d**2).T
                EP[t_idx, i] = np.sum(ddv[:, i] * mask)

        # ---------------prepare output data---------------
        dt2 = 1e-3
        times2 = np.arange(-0.5, 0.5 + dt2, dt2)

        # Process AP2
        AP2_raw = v.flatten() - np.min(v)
        max_val_ap = np.max(AP2_raw)
        i_max_ap = np.argmax(AP2_raw)
        AP2_norm = AP2_raw / max_val_ap

        f_ap = interp1d((times - times[i_max_ap]).flatten(), AP2_norm, kind='linear')
        AP2 = f_ap(times2).T

        # Process EP2
        min_val_ep = np.min(EP)
        i_min_ep = np.argmin(EP)
        EP2_norm = EP.flatten() / np.abs(min_val_ep)

        f_ep = interp1d((times - times[i_min_ep]).flatten(), EP2_norm, kind='linear')
        EP2 = f_ep(times2).T

        return EP2, times2, AP2

    # ======================================================================
    # Synaptic kernels (gen_kernels.py)
    # ======================================================================
    @staticmethod
    def gen_kernels(dt, tlength):
        """
        Generate AMPA and NMDA kernels (in msec) for convolution.
        Parameters and refs are based on Roese's report.
        """

        t = np.arange(0, tlength, dt)
        kernels = np.zeros((5, t.size))

        # tau = [rise, decay]
        tau = np.array([
            [1.0,   5.0],   # Index 0: AMPA (In->MN)
            [3.0,   50.0],  # Index 1: NMDA (In->MN)
            [0.5,   3.6],   # Index 2: fast AChR
            [1.8,   20.2],  # Index 3: slow AChR
            [1.0,   6.0]    # Index 4: GlyR (RC->MN)
        ])

        # h = normalizing terms
        h = np.array([1.8692, 1.2731, 1.5977, 1.3908, 1.7175]) * 1e-2

        for i in range(5):
            kernels[i, :] = h[i] * (np.exp(-t / tau[i, 1]) - np.exp(-t / tau[i, 0]))

        AMPA = kernels[0, :]
        NMDA = kernels[1, :]

        return AMPA, NMDA

    # ======================================================================
    # Synthetic MUAPs (gen_muaps.py, helper_f_genmuaps.py)
    # ======================================================================
    @staticmethod
    def time_vectors():
        """
        Generates and returns the time vectors of the MUAPs (tmuap) and of the
        motor neurons (t_mn)
        """
        tmuap = np.arange(0, 20, 0.1)
        t_mn = np.linspace(0, 20, 200)
        return tmuap, t_mn

    # ----------------------------------------------------------------------
    def gen_muaps(self, n_neurons, amplitude, axonalDelay, lam, zero_muaps=None):
        """
        Generate MUAP waveforms using the first-order Hermite-Rodriguez
        function described in the paper (Eqs. 4-5)
        (https://journals.physiology.org/doi/full/10.1152/jn.00626.2017).

        Parameters
        ----------
        n_neurons    : int
            Number of motor neurons
        amplitude    : ndarray
            Either amplitude of each MUAP (n_neurons,)
            Or scaling (a) and base (b) of exponent of the amplitude distribution a*b^x (2,)
        axonal delay : ndarray, shape (n_neurons,)
            Zero-crossing time of each anatomical MUAP, or a mean value.
        lam          : float or ndarray, shape (n_neurons,)
            Shape parameter of the Hermite-Rodriguez function [ms]. One value per
            motor unit as returned by fit_lam, or a mean value.
        zero_muaps   : ndarray of bool, shape (n_neurons,), optional
            Motor units whose anatomical MUAP is all zero. The corresponding
            synthetic MUAPs are set to all zero as well. If None, every MUAP is
            generated from the Hermite-Rodriguez function.

        Returns
        -------
        muaps : ndarray, shape (200, n_neurons)
        tmuap : ndarray, shape (200,)
        """
        # Paper parameters
        if isinstance(lam, np.ndarray):
            lam = np.atleast_1d(np.asarray(lam, dtype=float))[None, :]

        # Time vectors
        tmuap, t_mn = self.time_vectors()

        # Amplitude A_i: exponential (log-spaced) distribution over a n_neurons-fold
        # range, from A_1 = a (first-recruited MU) to A_M = b*a
        # (last-recruited MU). Motor units are assumed ordered by recruitment
        # order (column index 0 = first recruited).
        if len(amplitude) == 2:
            a = amplitude[0]
            b = amplitude[1]
            if n_neurons > 1:
                frac = np.arange(n_neurons) / (n_neurons - 1)
            else:
                frac = np.zeros(1)
            A = a * b ** frac
        else:
            A = amplitude

        t_M = t_mn[:, None]      # (200, 1)
        t_D = axonalDelay        # shape (N,)
        t_D = t_D[None, :]       # (1, N)
        t_MUAP = tmuap[:, None]       # (1, N)
        A = A[None, :]           # (1, N)

        z = t_D - t_M - t_MUAP # (200, N)

        # Eq. 4: H_i(t) = A_i * (tau_i - t) * exp(-((tau_i - t)/lambda)^2) * u(tau_i - t)
        normalization_factor = np.max(z * np.exp(-(z / lam) ** 2), axis=0, keepdims=True) # To ensure that Am = b * A1
        muaps = A * (z * np.exp(-(z / lam) ** 2)) / normalization_factor

        # An all-zero anatomical MUAP gives an all-zero synthetic MUAP. Its
        # amplitude and zero-crossing time are undefined (0 and NaN), so the
        # column is overwritten instead of being computed
        if zero_muaps is not None:
            muaps[:, np.asarray(zero_muaps, dtype=bool)] = 0.0

        return muaps, tmuap

    # ======================================================================
    # Spinal model core (MEPmodel_bio_core.py, sigmoid.py)
    # ======================================================================
    @staticmethod
    def sigmoid(x, x0, r, a):
        return a / (1 + np.exp(r * (x0 - x)))

    # ----------------------------------------------------------------------
    def MEPmodel_bio_core(self, model, spike_times=None):
        """
        Spinal model: LIF motor-neuron pool (+ optional Renshaw cells) driven by the
        DI-wave input, followed by superposition of the MUAPs at the MN firing times.

        Parameters
        ----------
        model : dict
            Model configuration (see config_model_bio / update_model_bio).
        spike_times : np.ndarray, optional
            [nMU x maxES x nIntensities] array of previously computed MN firing
            times (NaN-padded).  When supplied, the network simulation is skipped
            entirely and the MEP is rebuilt from these firing times together with
            the MUAPs in `model`.  Only model['t'], model['muaps'] and
            model['tmuap'] are then required.

        Returns
        -------
        sim : dict
        """
        if spike_times is not None:
            return self._core_from_spike_times(model, spike_times)

        # Mapping dictionary keys from the model structure
        DIwave         = model['DIwaveConv']
        t              = model['t']
        muaps          = model['muaps']
        tmuap          = model['tmuap']
        R              = model['R']             # membrane resistance of MNs
        Wexc           = model['Wexc']
        RWinh          = model['RWinh']         # R * Winh
        dt             = model['dt']
        tauLIF         = model['tauLIF']
        Eexc           = model['Eexc']
        Einh           = model['Einh']
        V_rest         = model['V_rest']
        V_thr          = model['V_thr']
        T_ref          = model['T_ref']
        Tmu            = model['Tmu']
        withRC         = model['withRC']
        maxES          = model['maxES']
        fastAChRweight = model['fastAChRweight']
        rc             = model['rc']
        tau            = np.array(model['kernel']['tau']) # Nested structure handled as dict
        h              = np.array(model['kernel']['h']).flatten()

        # ------------record for all TMS intensities--------------
        nIntensities = DIwave.shape[0]
        simMEP = np.zeros((nIntensities, len(t)))
        spike_times = np.full((100, maxES, nIntensities), np.nan)
        spike_times2 = [None] * nIntensities      # all spike times (list of objects)

        gexc_all = np.zeros((nIntensities, len(t)))
        ginh_all = np.zeros((nIntensities, len(t)))
        Rm_all   = np.zeros((nIntensities, len(t)))
        Vm_all   = np.zeros((nIntensities, len(t)))
        mRC_all  = np.zeros((nIntensities, len(t)))
        mMN_all  = np.zeros((nIntensities, len(t)))
        vRC_all  = np.zeros((nIntensities, len(t)))
        Iexc_all = np.zeros((nIntensities, len(t)))
        Iinh_all = np.zeros((nIntensities, len(t)))

        for i in range(nIntensities):

            # ----- Initialization -----
            gexc   = DIwave[i, :]                  # synaptic conductances on MNs
            ginh   = np.zeros(len(t))              # synaptic conductances on MNs
            Iexc   = np.zeros((100, len(t)))       # synaptic current to MNs
            Iinh   = np.zeros((100, len(t)))       # synaptic current to MNs
            Vm     = np.ones((100, len(t))) * V_rest
            Vm_lag = np.ones((100, len(t))) * V_rest
            sMN    = np.zeros((100, len(t)))       # MN spikes (0/1)
            TR     = np.zeros(100)                 # Timer of refractory period
            mMN    = np.zeros(len(t))              # MN firing rate
            mRC    = np.zeros(len(t))              # RC firing rate
            v      = np.zeros((3, len(t)))         # average postsynaptic potential
            v2     = np.zeros((3, len(t)))         # derivative helper

            # ----- simulation -----
            for tt in range(len(t) - 1):

                # -----MN rate-----
                mMN[tt] = np.mean(Wexc * sMN[:, tt])

                # -----RC rate-----
                if withRC:
                    rescale = 100
                    mRC[tt] = self.sigmoid(v[0, tt] * fastAChRweight + v[1, tt] * (1 - fastAChRweight),
                                           rc['v_thr'] / rescale, rc['r'] * rescale, rc['fmax'])

                # -----PSPs (biexponential kernel)-----
                input_vec = np.array([mMN[tt], mMN[tt], mRC[tt]])

                dv  = v2[:, tt]

                dv2 = (h * input_vec - v2[:, tt] * (tau[:, 0] + tau[:, 1]) / (tau[:, 0] * tau[:, 1]) - v[:, tt] / (tau[:, 0] * tau[:, 1]))

                v[:, tt + 1]  = v[:, tt]  + dv * dt
                v2[:, tt + 1] = v2[:, tt] + dv2 * dt

                if withRC:
                    ginh[tt] = v[2, tt]
                    Iexc[:, tt] = (Eexc - Vm[:, tt]) * gexc[tt]
                    Iinh[:, tt] = (Einh - Vm[:, tt]) * ginh[tt]
                    Vm[:, tt + 1] = Vm[:, tt] + (dt / tauLIF) * (V_rest - Vm[:, tt] +
                                    R * Iexc[:, tt] + RWinh * Iinh[:, tt])
                else:
                    Iexc[:, tt] = (Eexc - Vm[:, tt]) * gexc[tt]
                    Vm[:, tt + 1] = Vm[:, tt] + (dt / tauLIF) * (V_rest - Vm[:, tt] +
                                    R * Iexc[:, tt])

                # -----refractory period-----
                idx_tr = TR > 0
                Vm[idx_tr, tt + 1] = V_rest
                TR[idx_tr] -= 1

                # -----spike of 100 motor neurons-----
                idx_spike = Vm[:, tt + 1] > V_thr
                Vm_lag[idx_spike, tt + 1] = Vm[idx_spike, tt + 1]
                sMN[idx_spike, tt + 1] = 1
                Vm[idx_spike, tt + 1] = V_rest
                TR[idx_spike] = round(T_ref / dt)

            # ----- spike times -----
            T_idx = Tmu / dt

            for n in range(100):
                # Find all spikes for neuron n
                idx = np.where(Vm_lag[n, :] > V_thr)[0]

                # -----collect effective spikes-----
                idxES = []
                refractory_idx = 0
                counter = 0

                while len(idx) > 0 and counter < maxES:
                    if idx[0] > refractory_idx:
                        idxES.append(idx[0])
                        refractory_idx = idx[0] + T_idx
                        counter += 1
                    idx = idx[1:]

                # ---------------------------------
                for k in range(len(idxES)):
                    curr_idx = idxES[k]
                    spike_t = (dt / (Vm_lag[n, curr_idx] - Vm[n, curr_idx - 1]) * (V_thr - Vm[n, curr_idx - 1]) + t[curr_idx - 1])

                    spike_times[n, k, i] = spike_t

            # ----- MEP = superposition of MUAPs at the MN firing times -----
            simMEP[i, :] = self.mep_from_spike_times(spike_times[:, :, i], muaps, tmuap, t)

            # ---------for record--------
            gexc_all[i, :] = R[0] * gexc
            Rm_all[i, :]   = sMN[0, :]
            Vm_all[i, :]   = Vm[0, :]

            spike_rows, spike_cols = np.where(sMN == 1)
            spike_times2[i] = np.column_stack((spike_rows + 1, t[spike_cols]))

            mMN_all[i, :]  = mMN
            vRC_all[i, :]  = v[0, :] * fastAChRweight + v[1, :] * (1 - fastAChRweight)
            Iexc_all[i, :] = Iexc[0, :]
            Iinh_all[i, :] = Iinh[0, :]

            if withRC:
                mRC_all[i, :]  = mRC
                ginh_all[i, :] = RWinh[0] * ginh

        # Construct output dictionary
        sim = {
            't': t,
            'simMEP': simMEP,
            'spike_times': spike_times,
            'gexc_all': gexc_all,
            'ginh_all': ginh_all,
            'Rm_all': Rm_all,
            'Vm_all': Vm_all,
            'mRC_all': mRC_all,
            'mMN_all': mMN_all,
            'vRC_all': vRC_all,
            'spike_times2': spike_times2,
            'Iexc_all': Iexc_all,
            'Iinh_all': Iinh_all
        }

        return sim

    # ----------------------------------------------------------------------
    @staticmethod
    def mep_from_spike_times(spike_times_i, muaps, tmuap, t):
        """
        Superpose the MUAPs at the firing times of one TMS intensity.

        spike_times_i : [nMU x maxES] firing times (ms), NaN where there is no spike
        muaps         : [nSamples x nMU] MUAP waveforms
        tmuap         : [nSamples,] time base of the MUAPs (ms, relative to firing)
        t             : [nTime,] simulation time base (ms)
        """
        muaps  = np.asarray(muaps)
        tmuap  = np.asarray(tmuap).ravel()
        t      = np.asarray(t).ravel()
        mep    = np.zeros(len(t))

        for n in range(spike_times_i.shape[0]):
            st = spike_times_i[n, :]
            st = st[~np.isnan(st)]
            if st.size == 0:
                continue
            muap_n = muaps[:, n].ravel()
            for spike_t in st:
                # equivalent to interp1d(..., bounds_error=False, fill_value=0), but faster
                mep += np.interp(t, tmuap + spike_t, muap_n, left=0.0, right=0.0)

        return mep

    # ----------------------------------------------------------------------
    def _core_from_spike_times(self, model, spike_times):
        """
        Rebuild the MEP from previously saved MN firing times — no network
        simulation.  Only the MUAP-dependent part of the model is recomputed, so
        this is the fast path for parameter studies on the MUAPs (e.g. pygpc).
        """
        t           = np.asarray(model['t']).ravel()
        muaps       = np.asarray(model['muaps'])
        tmuap       = np.asarray(model['tmuap'])
        spike_times = np.asarray(spike_times, dtype=float)

        if spike_times.ndim != 3:
            raise ValueError(
                f'spike_times must be [nMU x maxES x nIntensities], got shape {spike_times.shape}'
            )
        nMU, _, nIntensities = spike_times.shape
        if muaps.shape[1] != nMU:
            raise ValueError(
                f'muaps has {muaps.shape[1]} motor units but spike_times has {nMU}'
            )

        simMEP = np.zeros((nIntensities, len(t)))
        for i in range(nIntensities):
            simMEP[i, :] = self.mep_from_spike_times(spike_times[:, :, i], muaps, tmuap, t)

        return {'t': t, 'simMEP': simMEP, 'spike_times': spike_times}

    # ======================================================================
    # Error / goodness of fit (MEPmodel_bio.py)
    # ======================================================================
    @staticmethod
    def _shift_and_scale(t, simMEP, t0, y0, axonalDelay):
        """
        Shift the simulated MEPs by `axonalDelay` ms, resample them on t0 (zero outside
        the simulated range) and scale them to the maximum of the measured MEPs.
        """
        f_interp = interp1d(
            t + axonalDelay,
            simMEP,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=0
        )
        simMEP = f_interp(t0)

        # normalization
        if np.max(simMEP) > 0:
            simMEP2 = simMEP / np.max(simMEP) * np.max(y0)
        else:
            simMEP2 = simMEP

        return simMEP2

    # ----------------------------------------------------------------------
    def fit_axonal_delay(self, t, simMEP, t0, y0, bounds=(0.0, 30.0), grid_step=None):
        """
        Axonal delay [ms] that minimises the sum of squared errors between the
        shifted, normalised simulated MEPs and the measured MEPs, over all TMS
        intensities at once. This is equivalent to maximising R2.

        The error is multimodal in the shift (one local minimum per MEP lobe), so
        a coarse grid over the fixed `bounds` first locates the global minimum,
        and a bounded Brent search then refines it within +/- one grid step.
        `bounds` must not depend on the model parameters, otherwise the result is
        no longer a continuous function of them.
        """
        if grid_step is None:
            grid_step = t[1] - t[0]

        def sse(shift):
            return np.sum((y0 - self._shift_and_scale(t, simMEP, t0, y0, shift)) ** 2)

        grid = np.arange(bounds[0], bounds[1] + 0.5 * grid_step, grid_step)
        i_best = int(np.argmin([sse(s) for s in grid]))

        lo = max(bounds[0], grid[i_best] - grid_step)
        hi = min(bounds[1], grid[i_best] + grid_step)
        res = minimize_scalar(sse, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-4})
        return res.x

    # ----------------------------------------------------------------------
    def cal_error(self, ref, sim, delay_bounds=(0.0, 30.0)):

        y0 = ref["y0"]
        t0 = ref["t0"]

        simMEP = sim["simMEP"].T
        t = sim["t"]

        # ----- axonal delay: shift that minimises the error over all intensities -----
        axonalDelay = self.fit_axonal_delay(t, simMEP, t0, y0, bounds=delay_bounds)

        # ----- shifted, interpolated and normalised simulated MEPs -----
        simMEP2 = self._shift_and_scale(t, simMEP, t0, y0, axonalDelay)

        error = np.ravel(simMEP2, order="F") - np.ravel(y0, order="F")

        NRMSD = (
            np.linalg.norm(y0 - simMEP2)
            / np.sqrt(len(t0))
            / (np.max(y0) - np.min(y0))
        )

        R2 = 1 - np.sum((y0 - simMEP2) ** 2) / np.sum((y0 - np.mean(y0)) ** 2)

        ref["NRMSD"] = NRMSD
        sim["simMEP2"] = simMEP2
        ref["model"]["axonalDelay"] = axonalDelay
        ref["R2"] = R2
        ref["error"] = error
        ref["sim"] = sim

        return ref


# ==========================================================================
if __name__ == '__main__':
    # fixed settings
    SUBJ       = 1
    WITHRC     = 1
    AMPAWEIGHT = None

    ROOT       = os.path.dirname(os.path.realpath(__file__))
    SPIKE_FILE = os.path.join(ROOT, 'data', f'mu_spiketimes_S{SUBJ}.h5')

    # pygpc parameters
    b   = 100
    lam = 3

    # Output metric
    R2 = MEPmodel().run_model(b, lam, SUBJ, WITHRC, AMPAWEIGHT, SPIKE_FILE)
    print(R2)
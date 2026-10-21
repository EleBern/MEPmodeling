import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from plot_summary import plot_summary, get_iocurve
from plot_param_panel import plot_param_panel
from plot_TMScondition import plot_TMScondition
from MEPmodel_bio_core import MEPmodel_bio_core
from gen_resistance_mono import gen_resistance_mono

# --------- Main function ---------
def MEPmodel_bio(p, ref, plotOn=0):

    ref = update_model_bio(ref, p)
    sim = MEPmodel_bio_core(ref["model"])  # simulation 
    ref = cal_error(ref, sim)
    simMEP = ref["sim"]["simMEP2"]
    try:
        _, _, _, _, ref['gof'] = get_iocurve(simMEP, ref)
    except:
        print("Sigmoid curve_fit failed for all initial guesses.")

    if plotOn:
        plot_param_panel(p, ref)
        plot_summary(p, ref)

        idx = len(ref["intensities"]) -1 
        plot_TMScondition(ref, idx)

    return simMEP, ref


# ==========================================================================
def update_model_bio(ref, p):

    if ref["model"]["withRC"] == 0:
        R     = p[0:5]
        E1    = 0
        E2    = 0
        I1    = 0
        I2    = 0
        RCth  = 10
        Tmu   = p[5]
        AMPAw = p[6]

    elif ref["model"]["withRC"] == 1:
        R     = p[0:5]
        E1    = p[5]
        E2    = p[6]
        I1    = p[7]
        I2    = p[8]
        RCth  = p[9]
        Tmu   = p[10]
        AMPAw = p[11]

    # ----- membrane resistances of MNs -----
    ref["model"]["R"] = gen_resistance_mono(R)

    # ----- Wexc (MNs -> RC) -----
    ref["model"]["Wexc"] = np.linspace(E1, E2, 100)

    # ----- R*Winh (RC->MNs) -----
    ref["model"]["RWinh"] = np.linspace(I1, I2, 100)

    # ----- RC sigmoid -----
    ref["model"]["rc"] = RC_setting(RCth)

    # ----- MUAP -----
    ref["model"]["Tmu"] = Tmu

    # ----- DI-waves -----
    ref["model"]["AMPAweight"] = AMPAw
    ref["model"]["DIwaveConv"] = (
        ref["model"]["DIwaveConv_AMPA"] * ref["model"]["AMPAweight"]
        + ref["model"]["DIwaveConv_NMDA"] * (1 - ref["model"]["AMPAweight"])
    )

    # ----- Motor neuron -----
    ref["model"]["tauLIF"] = 10
    ref["model"]["Eexc"] = 0
    ref["model"]["Einh"] = -75
    ref["model"]["V_rest"] = -65
    ref["model"]["V_thr"] = -55
    ref["model"]["T_ref"] = 2
    ref["model"]["fastAChRweight"] = 0.5

    return ref


# ==========================================================================
# Renshaw cell population params
def RC_setting(p):

    rc = {}

    # RC sigmoid function
    rc["r"] = 10
    rc["v_thr"] = p
    rc["fmax"] = 1

    # synaptic kernel (RC->MN, Glycine receptor)
    taur = 1
    tauf = 6
    rc["tau1"] = taur * tauf / (taur + tauf)
    rc["tau2"] = tauf
    rc["h"] = (rc["tau2"] - rc["tau1"]) / rc["tau1"] / rc["tau2"]

    return rc


# ==========================================================================
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

def fit_axonal_delay(t, simMEP, t0, y0, bounds=(0.0, 30.0), grid_step=None):
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
        return np.sum((y0 - _shift_and_scale(t, simMEP, t0, y0, shift)) ** 2)

    grid = np.arange(bounds[0], bounds[1] + 0.5 * grid_step, grid_step)
    i_best = int(np.argmin([sse(s) for s in grid]))

    lo = max(bounds[0], grid[i_best] - grid_step)
    hi = min(bounds[1], grid[i_best] + grid_step)
    res = minimize_scalar(sse, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-4})
    return res.x


def cal_error(ref, sim, delay_bounds=(0.0, 30.0)):

    y0 = ref["y0"]
    t0 = ref["t0"]

    simMEP = sim["simMEP"].T
    t = sim["t"]

    # ----- axonal delay: shift that minimises the error over all intensities -----
    axonalDelay = fit_axonal_delay(t, simMEP, t0, y0, bounds=delay_bounds)

    # ----- shifted, interpolated and normalised simulated MEPs -----
    simMEP2 = _shift_and_scale(t, simMEP, t0, y0, axonalDelay)

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
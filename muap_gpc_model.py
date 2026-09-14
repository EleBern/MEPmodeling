import os
import numpy as np
import inspect
from pygpc_main import run_model
from pygpc.AbstractModel import AbstractModel


class MUAP_gpc(AbstractModel):
    """
    MyModel evaluates something. The parameters of the model (constants and random parameters) are stored in the
    dictionary p. Their type is defined during the problem definition.

    Parameters
    ----------
    p["x1"] : float or ndarray of float [n_grid]
        Parameter 1
    p["x2"] : float or ndarray of float [n_grid]
        Parameter 2
    p["x3"] : float or ndarray of float [n_grid]
        Parameter 3

    Returns
    -------
    y : ndarray of float [n_grid x n_out]
        Results of the n_out quantities of interest the gPC is conducted for
    additional_data : dict or list of dict [n_grid]
        Additional data, will be saved under its keys in the .hdf5 file during gPC simulations.
        If multiple grid-points are evaluated in one function call, return a dict for every grid-point in a list
    """

    def __init__(self, fname_matlab=None, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())
        self.fname_matlab = fname_matlab

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        # fixed settings 
        SUBJ       = 1 # self.p["SUBJ"] -> to pass as a static parameter
        WITHRC     = 1
        AMPAWEIGHT = None

        ROOT       = os.path.dirname(os.path.realpath(__file__))
        SPIKE_FILE = os.path.join(ROOT, 'fitted_results', 'bio', f'mu_spiketimes_S{SUBJ}.h5')

        # setup is expensive, so build it once and reuse it for every pygpc sample
        _CACHE = {}

        # pygpc parameters
        a   = self.p["a"]      # pygpc parameter [2, 14] uniform
        b   = self.p["b"]   # pygpc parameter [45, 425] uniform
        lam = self.p["lam"]      # pygpc parameter [1.111 - 5.706], normal distribution mu=3.619, std=0.774

        R2 = np.zeros(a.shape[0])

        for n in range(a.shape[0]):
            # Output metric
            R2[n] = run_model(a[n], b[n], lam[n], SUBJ, WITHRC, AMPAWEIGHT, SPIKE_FILE)

        R2 = R2[:, np.newaxis]

        return R2
import os
import numpy as np
import inspect
from pygpc_main import run_model
from pygpc.AbstractModel import AbstractModel


class MUAP_gpc(AbstractModel):
    """
    MUAP_gpc runs a spinal periferal model that reproduces TMS invoked MEPs. 
    The parameters of the model (constants and random parameters) are stored in the
    dictionary p. Their type is defined during the problem definition.

    Parameters
    ----------
    p["a"] : float 
        Coefficient of the amplitude distribution
    p["b"] : float 
        Base of the exponent of the amplitude distribution
    p["lam"] : float 
        Width of the MUAPs

    Returns
    -------
    R2 : ndarray of float [n_grid x n_out]
        Goodness of fit of the MEPs in the simulations the gPC is conducted for
    """

    def __init__(self, fname_matlab=None, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())
        self.fname_matlab = fname_matlab

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        # fixed settings 
        SUBJ       = 1  # To pass as a static parameter -> self.p["SUBJ"] 
        WITHRC     = 1
        AMPAWEIGHT = None

        ROOT       = os.path.dirname(os.path.realpath(__file__))
        SPIKE_FILE = os.path.join(ROOT, 'fitted_results', 'bio', f'mu_spiketimes_S{SUBJ}.h5')

        # pygpc parameters
        a   = self.p["a"]      # pygpc parameter [2, 14] uniform
        b   = self.p["b"]      # pygpc parameter [45, 425] uniform
        delay_a   = self.p["delay_a"]      
        delay_b   = self.p["delay_b"]      
        lam = self.p["lam"]    # pygpc parameter [1.111 - 5.706], normal distribution mu=3.619, std=0.774

        R2 = np.zeros(a.shape[0])

        for n in range(a.shape[0]):
            # Output metric
            R2[n] = run_model(a[n], b[n], delay_a[n], delay_b[n], lam[n], SUBJ, WITHRC, AMPAWEIGHT, SPIKE_FILE)

        R2 = R2[:, np.newaxis]

        return R2
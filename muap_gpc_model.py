import os
import numpy as np
import inspect
from MEPmodel import MEPmodel
from pygpc.AbstractModel import AbstractModel


class MUAP_gpc(AbstractModel):
    """
    MUAP_gpc runs a spinal periferal model that reproduces TMS invoked MEPs. 
    The parameters of the model (constants parameters and the parameter distributions tested in the gPC)
    are stored in the dictionary p. Their type is defined during the problem definition (in main.py).

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
        DATA_DIR   = os.path.join(ROOT, 'data')
        SPIKE_FILE = os.path.join(DATA_DIR, f'mu_spiketimes_S{SUBJ}.h5')

        # pygpc parameters
        b   = self.p["b"]      # pygpc parameter [45, 425] uniform  
        lam = self.p["lam"]    # pygpc parameter, normal distribution mu=3.619, std=0.774

        mep_model = MEPmodel(data_dir=DATA_DIR)

        R2 = np.zeros(b.shape[0])

        for n in range(b.shape[0]):
            # Output metric
            R2[n] = mep_model.run_model(b[n], lam[n], SUBJ, WITHRC, AMPAWEIGHT, SPIKE_FILE)

        R2 = R2[:, np.newaxis]

        return R2

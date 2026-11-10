import numpy as np
import inspect
from pygpc_main import run_model
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

        # pygpc parameters
        # b   = self.p["b"]      
        # lam = self.p["lam"]    
        R1 = self.p["R1"] 
        # R2 = self.p["R2"] 
        # R3 = self.p["R3"] 
        # R4 = self.p["R4"] 
        # R5 = self.p["R5"] 
        # E1 = self.p["E1"] 
        # E2 = self.p["E2"] 
        # I1 = self.p["I1"] 
        I2 = self.p["I2"] 
        # RC_th = self.p["RC_th"] 
        # MU_T = self.p["MU_T"] 
        # AMPAw = self.p["AMPAw"] 

        b = np.ones((R1.shape[0])) * 3.7
        lam = np.ones((R1.shape[0])) * 75
        # R1 = np.ones((R1.shape[0]) * 0.86032462
        R2 = np.ones((R1.shape[0])) * 1.05927459
        R3 = np.ones((R1.shape[0])) * 1.57477574
        R4 = np.ones((R1.shape[0])) * 0.76463778
        R5 = np.ones((R1.shape[0])) * 9.80825998
        E1 = np.ones((R1.shape[0])) * 19.96654087
        E2 = np.ones((R1.shape[0])) * 11.40240176
        I1 = np.ones((R1.shape[0])) * 0.14947802
        # I2 = np.ones((R1.shape[0])) * 0.04018722
        RC_th = np.ones((R1.shape[0])) * 3.42343178
        MU_T = np.ones((R1.shape[0])) * 5.
        AMPAw = np.ones((R1.shape[0])) * 0.14387608

        coeff_determination = np.zeros(b.shape[0])

        for n in range(b.shape[0]):
            # Output metric
            coeff_determination[n] = run_model(SUBJ, b[n], lam[n], R1[n], R2[n], R3[n], R4[n], R5[n], E1[n], E2[n], I1[n], I2[n], RC_th[n], MU_T[n], AMPAw[n])

        coeff_determination = coeff_determination[:, np.newaxis]

        return coeff_determination
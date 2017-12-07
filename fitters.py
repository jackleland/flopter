from interfaces import IVFitter
import numpy as np

# Labels
ION_SAT = 'I_sat'       # Ion saturation current label
SHEATH_EXP = 'a'  # Sheath expansion parameter label
FLOAT_POT = 'V_f'       # Floating potential label
ELEC_TEMP = 'T_e'        # Electron temperature label
ELEC_DENS = 'n_e'


class FullFitter(IVFitter):
    """
    IV Fitter implementation utilising the full, 4 parameter IV Curve fitting method.
    """
    def __init__(self):
        super().__init__()
        self.params = {
            ION_SAT: 0,
            SHEATH_EXP: 1,
            FLOAT_POT: 2,
            ELEC_TEMP: 3
        }

    def fit(self):
        super().fit()

    def fit_function(self, v, *parameters):
        I_0 = parameters[0]
        a = parameters[1]
        v_f = parameters[2]
        T_e = parameters[self.params['temperature']]
        V = (v_f - v) / T_e
        return I_0 * (1 - np.exp(-V) + (a * np.float_power(np.absolute(V), 0.75)))

    def get_param_index(self, label):
        return self.params[label]

    def get_temp_index(self):
        return self.params[ELEC_TEMP]

    def get_isat_index(self):
        return self.params[ION_SAT]

    def get_vf_index(self):
        return self.params[FLOAT_POT]

    def get_a_index(self):
        return self.params[SHEATH_EXP]

# class SimpleFitter(IVFitter):
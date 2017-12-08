from abc import ABC, abstractmethod
import numpy as np

# Labels
ION_SAT = 'I_sat'       # Ion saturation current label
SHEATH_EXP = 'a'        # Sheath expansion parameter label
FLOAT_POT = 'V_f'       # Floating potential label
ELEC_TEMP = 'T_e'       # Electron temperature label
ELEC_DENS = 'n_e'       # Electron density label


class IVFitter(ABC):
    """
    Semi-abstract base class for a Fitter object, which describes a model for fitting an IV curve.
    """
    def __init__(self):
        self._params = {}
        self.name = None

    @abstractmethod
    def fit_function(self, v, *parameters):
        pass

    @abstractmethod
    def fit(self):
        pass

    def get_params(self):
        return self._params

    def get_param_index(self, label):
        if label in self._params.keys():
            return self._params[label]
        else:
            print('Param described by \'label\' not found')
            if len(self._params.keys()) > 0:
                print('Params in {} are: \n {}'.format(self.name, self._params.keys()))
            else:
                print('No params, IVFitter class not properly implemented')
            return None


class FullFitter(IVFitter):
    """
    IV Fitter implementation utilising the full, 4 parameter IV Curve fitting method.
    """
    def __init__(self):
        super().__init__()
        self._params = {
            ION_SAT: 0,
            SHEATH_EXP: 1,
            FLOAT_POT: 2,
            ELEC_TEMP: 3
        }
        self.name = '4 Parameter Fit'

    def fit(self):
        super().fit()

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._params[ION_SAT]]
        a = parameters[self._params[SHEATH_EXP]]
        v_f = parameters[self._params[FLOAT_POT]]
        T_e = parameters[self._params[ELEC_TEMP]]
        V = (v_f - v) / T_e
        return I_0 * (1 - np.exp(-V) + (a * np.float_power(np.absolute(V), 0.75)))

    def get_param_index(self, label):
        return self._params[label]

    def get_temp_index(self):
        return self._params[ELEC_TEMP]

    def get_isat_index(self):
        return self._params[ION_SAT]

    def get_vf_index(self):
        return self._params[FLOAT_POT]

    def get_a_index(self):
        return self._params[SHEATH_EXP]


class SimpleFitter(IVFitter):
    def __init__(self):
        super().__init__()
        self._params = {
            ION_SAT: 0,
            FLOAT_POT: 1,
            ELEC_TEMP: 2
        }
        self.name = '3 Parameter Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._params[ION_SAT]]
        v_f = parameters[self._params[FLOAT_POT]]
        T_e = parameters[self._params[ELEC_TEMP]]
        V = (v_f - v) / T_e
        return I_0 * (1 - np.exp(-V))

    def fit(self):
        super().fit()

    def get_temp_index(self):
        return self._params[ELEC_TEMP]

    def get_isat_index(self):
        return self._params[ION_SAT]

    def get_vf_index(self):
        return self._params[FLOAT_POT]


class IonCurrentSEFitter(IVFitter):
    def __init__(self):
        super().__init__()
        self._params = {
            ION_SAT: 0,
            SHEATH_EXP: 1
        }
        self.name = 'Ion Current Sheath Expansion Fit'

    def fit(self):
        super().fit()

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._params[ION_SAT]]
        a = parameters[self._params[SHEATH_EXP]]
        return I_0 * (1 + (a * v))

    def get_isat_index(self):
        return self._params[ION_SAT]

    def get_a_index(self):
        return self._params[SHEATH_EXP]

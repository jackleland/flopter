from abc import ABC, abstractmethod
from constants import ELEC_TEMP, ION_SAT, SHEATH_EXP, FLOAT_POT, ELEC_MASS
from normalisation import _BOLTZMANN, _ELECTRON_MASS
from scipy.optimize import curve_fit
from datatypes import FitData2, IVFitData
import numpy as np


class GenericFitter(ABC):
    """
    Semi-abstract base class for a Fitter object, which describes a model for fitting an IV curve.
    """
    def __init__(self):
        self._param_labels = {}
        self.name = None

    @abstractmethod
    def fit_function(self, v, parameters):
        pass

    def fit(self, x_data, y_data, initial_vals, bounds=(-np.inf, np.inf)):
        # Check params exist and check length of passed arrays
        if len(self._param_labels) == 0:
            print('No params, IVFitter class not properly implemented')
            return None
        if len(initial_vals) != len(self._param_labels):
            raise ValueError('Intial parameter array ({}) must be same length as defined parameter list'
                             '(see get_param_labels()).'.format(len(initial_vals)))
        elif bounds and len(bounds) == 2 * len(self._param_labels):
            raise ValueError('Parameter bounds array ({}) must be exactly double the length of defined parameter list'
                             '(see get_param_labels()).'.format(len(bounds)))

        fit_vals, fit_cov = curve_fit(self.fit_function, x_data, y_data, p0=initial_vals)
        fit_y_data = self.fit_function(x_data, *fit_vals)
        fit_sterrs = np.sqrt(np.diag(fit_cov))
        return FitData2(x_data, y_data, fit_y_data, fit_vals, fit_sterrs, self)

    def get_param_labels(self):
        return list(self._param_labels.keys())

    def get_param_index(self, label):
        if label in self._param_labels.keys():
            return self._param_labels[label]
        else:
            print('Param described by \'label\' not found')
            if len(self._param_labels.keys()) > 0:
                print('Params in {} are: \n {}'.format(self.name, self._param_labels.keys()))
            else:
                print('No params, IVFitter class not properly implemented')
            return None


class IVFitter(GenericFitter, ABC):
    def fit(self, x_data, y_data, initial_vals, bounds=None):
        fit_data = super().fit(x_data, y_data, initial_vals, bounds)
        return IVFitData.from_fit_data(fit_data)


class FullIVFitter(IVFitter):
    """
    IV Fitter implementation utilising the full, 4 parameter IV Curve fitting method.
    """
    def __init__(self):
        super().__init__()
        self._param_labels = {
            ION_SAT: 0,
            SHEATH_EXP: 1,
            FLOAT_POT: 2,
            ELEC_TEMP: 3
        }
        self.name = '4 Parameter Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[ION_SAT]]
        a = parameters[self._param_labels[SHEATH_EXP]]
        v_f = parameters[self._param_labels[FLOAT_POT]]
        T_e = parameters[self._param_labels[ELEC_TEMP]]
        V = (v_f - v) / T_e
        return I_0 * (1 - np.exp(-V) + (a * np.float_power(np.absolute(V), 0.75)))

    def get_param_index(self, label):
        return self._param_labels[label]

    def get_temp_index(self):
        return self._param_labels[ELEC_TEMP]

    def get_isat_index(self):
        return self._param_labels[ION_SAT]

    def get_vf_index(self):
        return self._param_labels[FLOAT_POT]

    def get_a_index(self):
        return self._param_labels[SHEATH_EXP]


class SimpleIVFitter(IVFitter):
    def __init__(self):
        super().__init__()
        self._param_labels = {
            ION_SAT: 0,
            FLOAT_POT: 1,
            ELEC_TEMP: 2
        }
        self.name = '3 Parameter Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[ION_SAT]]
        v_f = parameters[self._param_labels[FLOAT_POT]]
        T_e = parameters[self._param_labels[ELEC_TEMP]]
        V = (v_f - v) / T_e
        return I_0 * (1 - np.exp(-V))

    def get_temp_index(self):
        return self._param_labels[ELEC_TEMP]

    def get_isat_index(self):
        return self._param_labels[ION_SAT]

    def get_vf_index(self):
        return self._param_labels[FLOAT_POT]


class IonCurrentSEFitter(IVFitter):
    def __init__(self):
        super().__init__()
        self._param_labels = {
            ION_SAT: 0,
            SHEATH_EXP: 1
        }
        self.name = 'Ion Current Sheath Expansion Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._params[ION_SAT]]
        a = parameters[self._params[SHEATH_EXP]]
        return I_0 * (1 + (a * v))

    def get_isat_index(self):
        return self._param_labels[ION_SAT]

    def get_a_index(self):
        return self._param_labels[SHEATH_EXP]


class Maxwellian3Fitter(GenericFitter):
    def __init__(self):
        super().__init__()
        self._param_labels = {
            ELEC_TEMP: 0,
            ELEC_MASS: 1,
        }
        self.name = '3D Maxwellian Distribution Fit'

    def fit_function(self, v, *parameters):
        T_e = parameters[self._param_labels[ELEC_TEMP]]
        m = parameters[self._param_labels[ELEC_MASS]]
        a = np.sqrt((T_e * _BOLTZMANN) / (2*m))
        v = np.abs(v)
        return np.power(a/np.sqrt(np.pi), 3) * 4 * np.pi * np.power(v, 2) * np.exp(-np.power(a*v, 2))

    def get_temp_index(self):
        return self._param_labels[ELEC_TEMP]

    def get_mass_index(self):
        return self._param_labels[ELEC_MASS]


class Gaussian1DFitter(GenericFitter):
    def __init__(self):
        super().__init__()
        self._param_labels = {
            ELEC_TEMP: 0,
            ELEC_MASS: 1,
            "x_0": 2,
            "A": 3
        }
        self.name = '1D Gaussian Distribution Fit'

    def fit_function(self, v, *parameters):
        T_e = parameters[self._param_labels[ELEC_TEMP]]
        m = parameters[self._param_labels[ELEC_MASS]]
        x_0 = parameters[self._param_labels["x_0"]]
        A = parameters[self._param_labels["A"]]
        a = m * _ELECTRON_MASS / (2 * _BOLTZMANN * T_e)
        # v = np.abs(v - x_0)
        return A * np.sqrt(a / np.pi) * np.exp(-a * np.power(np.abs(v - x_0), 2))

    def get_temp_index(self):
        return self._param_labels[ELEC_TEMP]

    def get_mass_index(self):
        return self._param_labels[ELEC_MASS]

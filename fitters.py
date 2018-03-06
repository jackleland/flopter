from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from classes.ivdata import IVData
from classes.fitdata import FitData2, IVFitData
import constants as c
from normalisation import _BOLTZMANN, _ELECTRON_MASS, _ELEM_CHARGE, _PROTON_MASS, _P_E_MASS_RATIO
from warnings import warn

# Curve-fit default values and bounds
CF_DEFAULT_VALUES = None
CF_DEFAULT_BOUNDS = (-np.inf, np.inf)


class GenericFitter(ABC):
    """
    Semi-abstract base class for a Fitter object, which describes a model for fitting an IV curve.
    """
    def __init__(self):
        self._param_labels = {}
        self.default_values = None
        self.default_bounds = (-np.inf, np.inf)
        self.name = None

    @abstractmethod
    def fit_function(self, v, parameters):
        pass

    def fit(self, x_data, y_data, initial_vals=None, bounds=None):
        # Check params exist and check length of passed arrays
        if len(self._param_labels) == 0:
            print('No params, IVFitter class not properly implemented')
            return None

        if not initial_vals:
            initial_vals = self.default_values
        if initial_vals and len(initial_vals) != len(self._param_labels):
            warn('Intial parameter array ({}) must be same length as defined parameter list (see get_param_labels()).'
                 .format(len(initial_vals)))
            initial_vals = CF_DEFAULT_VALUES

        if not bounds:
            bounds = self.default_bounds
        if bounds and bounds != CF_DEFAULT_BOUNDS and len(bounds) == 2 * len(self._param_labels):
            warn('Parameter bounds array ({}) must be exactly double the length of defined parameter list '
                 '(see get_param_labels()).'.format(len(bounds)))
            bounds = CF_DEFAULT_VALUES

        fit_vals, fit_cov = curve_fit(self.fit_function, x_data, y_data, p0=initial_vals, bounds=bounds)
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

    def get_default_values(self):
        return self.default_values

    def get_default_bounds(self):
        return self.default_bounds


class IVFitter(GenericFitter, ABC):
    def fit(self, x_data, y_data, initial_vals=None, bounds=None):
        fit_data = super().fit(x_data, y_data, initial_vals=initial_vals, bounds=bounds)
        return IVFitData.from_fit_data(fit_data)

    def fit_iv_data(self, iv_data, initial_vals=None, bounds=None):
        assert isinstance(iv_data, IVData)
        potential = iv_data[c.POTENTIAL]
        current = iv_data[c.CURRENT]
        return self.fit(potential, current, initial_vals, bounds)


class FullIVFitter(IVFitter):
    """
    IV Fitter implementation utilising the full, 4 parameter IV Curve fitting method.
    """
    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.SHEATH_EXP: 1,
            c.FLOAT_POT: 2,
            c.ELEC_TEMP: 3
        }
        self.default_values = (30.0, -1.5, 0.0204, 1)
        self.default_bounds = (
            (-np.inf, -np.inf,      0, -np.inf),
            ( np.inf,  np.inf, np.inf,  np.inf)
        )
        self.name = '4 Parameter Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        a = parameters[self._param_labels[c.SHEATH_EXP]]
        v_f = parameters[self._param_labels[c.FLOAT_POT]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        V = (v_f - v) / T_e
        return I_0 * (1 - np.exp(-V) + (a * np.float_power(np.absolute(V), [0.75])))

    def get_param_index(self, label):
        return self._param_labels[label]

    def get_temp_index(self):
        return self._param_labels[c.ELEC_TEMP]

    def get_isat_index(self):
        return self._param_labels[c.ION_SAT]

    def get_vf_index(self):
        return self._param_labels[c.FLOAT_POT]

    def get_a_index(self):
        return self._param_labels[c.SHEATH_EXP]


class SimpleIVFitter(IVFitter):
    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.FLOAT_POT: 1,
            c.ELEC_TEMP: 2
        }
        self.default_values = (30.0, -1.5,  1)
        self.default_bounds = (
            (-np.inf, -np.inf, -np.inf),
            ( np.inf,  np.inf,  np.inf)
        )
        self.name = '3 Parameter Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        v_f = parameters[self._param_labels[c.FLOAT_POT]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        V = (v_f - v) / T_e
        return I_0 * (1 - np.exp(-V))

    def get_temp_index(self):
        return self._param_labels[c.ELEC_TEMP]

    def get_isat_index(self):
        return self._param_labels[c.ION_SAT]

    def get_vf_index(self):
        return self._param_labels[c.FLOAT_POT]


class IonCurrentSEFitter(IVFitter):
    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.SHEATH_EXP: 1
        }
        self.default_values = (30.0, 0.0204)
        self.default_bounds = (
            (-np.inf,      0),
            ( np.inf, np.inf)
        )
        self.name = 'Ion Current Sheath Expansion Fit'

    def fit_iv_data(self, iv_data, initial_vals=None, bounds=None):
        assert isinstance(iv_data, IVData)
        potential = np.power(np.abs(iv_data[c.POTENTIAL]), 0.75)
        current = iv_data[c.ION_CURRENT]
        return self.fit(potential, current, initial_vals, bounds)

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        a = parameters[self._param_labels[c.SHEATH_EXP]]
        return I_0 * (1 + (a * v))

    def get_isat_index(self):
        return self._param_labels[c.ION_SAT]

    def get_a_index(self):
        return self._param_labels[c.SHEATH_EXP]


class MaxwellianVelFitter(GenericFitter):
    def __init__(self, si_units=False, mu=_P_E_MASS_RATIO):
        super().__init__()
        self._param_labels = {
            c.ELEC_TEMP: 0,
            c.FLOW_VEL: 1
        }
        self.default_values = (1, 1)
        self.default_bounds = (
            (0, -np.inf),
            (np.inf, np.inf)
        )
        if si_units:
            self.temp_conversion = _BOLTZMANN
        else:
            self.temp_conversion = _ELEM_CHARGE

        self.mass = mu * _ELECTRON_MASS
        self.name = '3D Maxwellian Distribution Fit'

    def fit_function(self, v, *parameters):
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        a = self.mass / (2 * self.temp_conversion * T_e)
        return np.power(a / np.pi, 1.5) * 4 * np.pi * np.power(v, 2) * np.exp(-a * np.power(v, 2))

    def get_temp_index(self):
        return self._param_labels[c.ELEC_TEMP]

    def get_flow_velocity(self):
        return self._param_labels[c.FLOW_VEL]


class GaussianFitter(GenericFitter):
    def __init__(self, si_units=False, mu=_P_E_MASS_RATIO, v_scale=1):
        super().__init__()
        self._param_labels = {
            c.ELEC_TEMP: 0,
            c.FLOW_VEL: 1
        }
        self.default_values = (1.0, 0.0)
        self.default_bounds = (
            (   0.0, -np.inf),
            (np.inf,  np.inf)
        )
        self.si_units = si_units
        if si_units:
            self.temp_conversion = _BOLTZMANN
        else:
            self.temp_conversion = _ELEM_CHARGE

        self.v_scale = v_scale
        self.mass = mu * _ELECTRON_MASS
        self.name = 'Gaussian Dist Fit'

    @staticmethod
    def _get_v0_guess(hist, hist_bins):
        # estimate values for the initial guess based on the mid point between the points of greatest +ve and -ve
        # curvature
        grad_fv = np.gradient(hist)
        sm_grad_fv = savgol_filter(grad_fv, 21, 2)
        min = np.argmin(sm_grad_fv)
        max = np.argmax(sm_grad_fv)

        v_0_guess = hist_bins[int((max + min) / 2)]
        return v_0_guess

    def fit(self, x_data, y_data, initial_vals=None, bounds=None, temp=None):
        """
            Override of the fit method with included provision for automatic finding of the flow velocity if one is not
            specified. A separate temperature can be specified with the temp keyword.
        :param x_data:          x-data for fit
        :param y_data:          y-data for fit
        :param initial_vals:    [optional] initial values to be fed to the fitting algorithm. If left None, v_0 will be
                                populated automatically using _get_v0_guess() and Temperature will be taken as a kwarg
                                or taken from the default values.
        :param bounds:          The bounds for the fitting algorithm. If none are specified then the default parameters
                                are
        :param temp:            [optional] temperature to use as an initial value for the fitting algorithm if v_0 is
                                to be estimated automatically. Ignored if initial_vals is not None.
        :return fitdata:        fitdata from fit, scaled back to normal values.
        """
        temp_ind = self._param_labels[c.ELEC_TEMP]
        flow_ind = self._param_labels[c.FLOW_VEL]
        if not initial_vals:
            v_0 = self._get_v0_guess(y_data, x_data) / self.v_scale
            if temp:
                initial_vals = [temp / (self.v_scale ** 2), v_0]
            else:
                initial_vals = [self.default_values[temp_ind] / (self.v_scale ** 2), v_0]

        fitdata = super().fit(x_data, y_data, initial_vals=initial_vals, bounds=bounds)

        fitdata.fit_params[temp_ind].value *= self.v_scale ** 2
        fitdata.fit_params[temp_ind].error *= self.v_scale ** 2
        fitdata.fit_params[flow_ind].value *= self.v_scale
        fitdata.fit_params[flow_ind].error *= self.v_scale
        return fitdata

    def fit_function(self, v, *parameters):
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_0 = parameters[self._param_labels[c.FLOW_VEL]]
        a = self.mass / (2 * self.temp_conversion * T_e)
        return np.sqrt(a / np.pi) * np.exp(-a * np.power(v - v_0, 2))

    def set_mass_scaler(self, scaling_val):
        self.mass = _ELECTRON_MASS * scaling_val

    def get_temp_index(self):
        return self._param_labels[c.ELEC_TEMP]

    def get_vflow_index(self):
        return self._param_labels[c.FLOW_VEL]


class GaussianVelFitter(GaussianFitter):
    """
        Generic Gaussian velocity distribution fitter in SI units
         - V_th in m/s
         - V_0 in m/s
    """
    def __init__(self, v_scale=1):
        super().__init__(v_scale=v_scale, si_units=True)
        self.name = 'Generic Gaussian Vel Dist'

    def fit_function(self, v, *parameters):
        v_th2 = parameters[self._param_labels[c.THERM_VEL]]
        v_0 = parameters[self._param_labels[c.FLOW_VEL]]
        return np.sqrt(1 / v_th2 * np.pi) * np.exp(-np.power(v - v_0, 2) / v_th2)


class GaussianVelElecFitter(GaussianFitter):
    """
        Gaussian velocity distribution fitter in SI units
         - T_e in K
         - V_0 in m/s
         - Electron mass used
    """
    def __init__(self, v_scale=1):
        super().__init__(v_scale=v_scale, si_units=True, mu=1)
        self.name = 'Gauss. Elec V-Dist (SI)'


class GaussianVelIonEvFitter(GaussianFitter):
    """
        Gaussian velocity distribution fitter for ions in alternative units
         - T_e in eV
         - V_0 in m/s
         - Ion mass used
    """
    def __init__(self, mu=_P_E_MASS_RATIO, v_scale=1):
        super().__init__(mu=mu, v_scale=v_scale, si_units=False)
        self.name = 'Gauss. Ion V-Dist (eV)'

    def fit_function(self, v, *parameters):
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_0 = parameters[self._param_labels[c.FLOW_VEL]]
        a = self.mass / (2 * _ELEM_CHARGE * T_e)
        return np.sqrt(a / np.pi) * np.exp(-(a) * np.power(v - v_0, 2))


class GaussianVelElecEvFitter(GaussianFitter):
    """
        Gaussian velocity distribution fitter for ions in alternative units
         - T_e in eV
         - V_0 in m/s
         - Electron mass used
    """
    def __init__(self, v_scale=1):
        super().__init__(mu=1, v_scale=v_scale, si_units=False)
        self.name = 'Gauss. Elec V-Dist (eV)'


class ScalableGaussianFitter(GaussianFitter):
    def __init__(self, mu=_P_E_MASS_RATIO, si_units=False, v_scale=1):
        super().__init__(mu=mu, v_scale=v_scale, si_units=si_units)
        self._param_labels = {
            c.ELEC_TEMP: 0,
            c.FLOW_VEL: 1,
            c.DIST_SCALER: 2
        }
        self.default_values = (1.0, 0.0, 1.0)
        self.default_bounds = (
            (     0, -np.inf, -np.inf),
            (np.inf,  np.inf,  np.inf)
        )
        self.name = 'Scalable Gauss. V-Dist'

    def fit_function(self, v, *parameters):
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        x_0 = parameters[self._param_labels[c.FLOW_VEL]]
        A = parameters[self._param_labels[c.DIST_SCALER]]
        a = self.mass / (2 * self.temp_conversion * T_e)
        return A * np.sqrt(a / np.pi) * np.exp(-a * np.power(np.abs(v - x_0), 2))

    def get_scaler_index(self):
        return self._param_labels[c.DIST_SCALER]

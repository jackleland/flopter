from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import curve_fit
import scipy.signal as sig
from scipy.interpolate import interp1d
import collections as coll

from flopter.core.ivdata import IVData
from flopter.core.fitdata import FitData2, IVFitData
from flopter.core import constants as c
from flopter.core.constants import BOLTZMANN, ELEM_CHARGE, ELECTRON_MASS, P_E_MASS_RATIO
from warnings import warn

# Curve-fit default values and bounds
CF_DEFAULT_VALUES = None
CF_DEFAULT_BOUNDS = (-np.inf, np.inf)


class GenericCurveFitter(ABC):
    """
    Semi-abstract base class for a Fitter object, which describes a model for
    fitting an IV curve using scipy's curve_fit().

    :param: fixed_values
    """
    def __init__(self):
        self._param_labels = {}
        self.default_values = None
        self.default_bounds = (-np.inf, np.inf)
        self.fixed_values = None
        self.name = None

    @abstractmethod
    def fit_function(self, v, parameters):
        pass

    def check_for_fixed_vals(self, parameters):
        if self.has_fixed_values() and len([value for value in self.fixed_values if value is None]) == len(parameters):
            iterator = iter(parameters)
            return [next(iterator) if value is None else value for value in self.fixed_values]
        else:
            return parameters

    def _curve_fit_func(self, v, *parameters):
        parameters = self.check_for_fixed_vals(parameters)
        return self.fit_function(v, *parameters)

    def fit(self, x_data, y_data, initial_vals=None, bounds=None, sigma=None):
        # Check params exist and check length of passed arrays
        if len(self._param_labels) == 0:
            print('No params, IVFitter class not properly implemented')
            return None

        if not initial_vals or np.NaN in initial_vals:
            initial_vals = list(self.default_values)
        if initial_vals and len(initial_vals) != len(self._param_labels):
            warn('Intial parameter array ({}) must be same length as defined parameter list (see get_param_labels()).'
                 .format(len(initial_vals)))
            initial_vals = CF_DEFAULT_VALUES

        if not bounds:
            bounds = [list(bound_tuple) for bound_tuple in self.default_bounds]
        if bounds and bounds != CF_DEFAULT_BOUNDS and len(bounds) == 2 * len(self._param_labels):
            warn('Parameter bounds array ({}) must be exactly double the length of defined parameter list '
                 '(see get_param_labels()).'.format(len(bounds)))
            bounds = list(CF_DEFAULT_BOUNDS)

        if self.has_fixed_values() and len(initial_vals) == len(self.fixed_values):
            pops = 0
            for i, fixed_val in enumerate(self.fixed_values):
                if fixed_val is not None:
                    initial_vals.pop(i - pops)
                    bounds[0].pop(i - pops)
                    bounds[1].pop(i - pops)
                    pops += 1
            assert len(initial_vals) == len([value for value in self.fixed_values if value is None])

        fit_vals, fit_cov = curve_fit(self._curve_fit_func, x_data, y_data, p0=initial_vals, bounds=bounds, sigma=sigma)
        fit_sterrs = np.sqrt(np.diag(fit_cov))

        # Pad fit vals with any fixed values and created model y-data
        if self.has_fixed_values():
            iter_vals = iter(fit_vals)
            iter_ster = iter(fit_sterrs)
            fit_vals = [value if value is not None else next(iter_vals) for value in self.fixed_values]
            fit_sterrs = [0.0 if value is not None else next(iter_ster) for value in self.fixed_values]
        fit_y_data = self.fit_function(x_data, *fit_vals)

        # Calculate chi^2 and reduced chi^2 if errors present in fitting
        if sigma is not None:
            fit_chi2 = (((y_data - fit_y_data) / sigma)**2).sum()
            fit_reduced_chi2 = fit_chi2 / (len(y_data) - len(initial_vals))
        else:
            fit_chi2 = None
            fit_reduced_chi2 = None
        return FitData2(x_data, y_data, fit_y_data, fit_vals, fit_sterrs, self, sigma=sigma, chi2=fit_chi2,
                        reduced_chi2=fit_reduced_chi2)

    def clip_guess(self, guess):
        """
        Filter the values in a given guess (i.e. array of initial parameter
        values) to make them suitable for the default bounds of a CurveFitter
        instantiation. Guess must already be in the correct order as per
        self._param_labels.

        :param guess:   (list-like) List of values desired to be used as
                        initial_vals
        :return:        (list-like) List of values clipped to the default bounds
                        defined.

        """
        assert len(guess) == len(self.default_values)
        output_guess = []
        for i, value in enumerate(guess):
            if value is None or not np.isfinite(value):
                value = self.default_values[i]
            else:
                value = np.clip(value, self.default_bounds[0][i], self.default_bounds[1][i])
            output_guess.append(value)
        return output_guess

    def get_param_labels(self):
        return list(self._param_labels.keys())

    def get_param_index(self, label, silent_fail_fl=True):
        if label in self._param_labels.keys():
            return self._param_labels[label]
        else:
            if silent_fail_fl:
                return None
            else:
                raise ValueError('Param described by \'label\' not found. \n'
                                 'Params in {} are: \n {}'.format(self.name, self._param_labels.keys()))

    def get_default_values(self):
        return self.default_values

    def get_default_bounds(self):
        return self.default_bounds

    def has_fixed_values(self):
        return (isinstance(self.fixed_values, coll.Sequence)
                and any(self.fixed_values)
                and len(self.fixed_values) == len(self._param_labels))

    def get_fixed_value_list(self, value_dict):
        assert isinstance(value_dict, dict)

        if len(value_dict) > len(self._param_labels):
            raise ValueError('Inputted dictionary must be not be longer than the number of parameters in the model.')

        if not set(value_dict.keys()).issubset(set(self._param_labels.keys())):
            raise ValueError('Inputted dictionary contained elements which were not in the parameter list for the '
                             'fitter. Allowed values are: {}'.format(list(self._param_labels.keys())))

        return [value_dict[label] if label in value_dict else None for label in self._param_labels]

    def set_fixed_values(self, value_dict):
        self.fixed_values = self.get_fixed_value_list(value_dict)

    def unset_fixed_values(self):
        self.fixed_values = None

    def __contains__(self, label):
        return label in self._param_labels


# --- IV Fitters --- #

class IVFitter(GenericCurveFitter, ABC):
    _DEFAULT_V_F = -1.0

    def fit(self, x_data, y_data, initial_vals=None, bounds=None, print_fl=False, sigma=None):
        # If no guess has been made and the floating potential is not fixed, set a starting value for the floating
        # potential.
        if self.has_fixed_values() and self.fixed_values[self.get_vf_index()] is None and initial_vals is None:
            initial_vals = list(self.default_values)
            initial_vals[self.get_vf_index()] = self.find_floating_pot(x_data, y_data)

        # Fit the data and return an IVFitData object.
        fit_data = super().fit(x_data, y_data, initial_vals=initial_vals, bounds=bounds, sigma=sigma)
        return IVFitData.from_fit_data(fit_data)

    def fit_iv_data(self, iv_data, initial_vals=None, bounds=None, sigma=None, print_fl=False):
        assert isinstance(iv_data, IVData)
        potential = iv_data[c.POTENTIAL]
        current = iv_data[c.CURRENT]
        return self.fit(potential, current, initial_vals, bounds, sigma=sigma, print_fl=print_fl)

    @classmethod
    def find_floating_pot_iv_data(cls, iv_data):
        return cls.find_floating_pot(iv_data[c.POTENTIAL], iv_data[c.CURRENT])

    @classmethod
    def find_floating_pot(cls, potential, current, print_fl=False):
        try:
            iv_interp = interp1d(current, potential)
            v_f = iv_interp([0.0]).mean()
        except ValueError as e:
            if print_fl:
                print('V_f could not be found effectively, returning approximate value.')
                print('Error: {}'.format(str(e)))
            # Estimate based on the voltage at the current value closest to 0.
            v_f = potential[np.argmin(np.abs(current))]
        return v_f

    def get_temp_index(self):
        return self._param_labels[c.ELEC_TEMP]

    def get_isat_index(self):
        return self._param_labels[c.ION_SAT]

    def get_vf_index(self):
        return self._param_labels[c.FLOAT_POT]

    def generate_guess(self, temperature=None, floating_potential=None, isat=None, sheath_exp_param=None):
        return self.clip_guess([isat, temperature, floating_potential])


class FullIVFitter(IVFitter):
    """
    IV Fitter implementation utilising the full, 4 parameter IV Curve fitting method.
    """
    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.SHEATH_EXP: 1,
            c.ELEC_TEMP: 2,
            c.FLOAT_POT: 3
        }
        self.default_values = (30.0, 0.0204, 1, -1)
        self.default_bounds = (
            (-np.inf,    0,       0, -np.inf),
            ( np.inf,  1.5,  np.inf,  np.inf)
        )
        self.name = '4 Parameter Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        a = parameters[self._param_labels[c.SHEATH_EXP]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_f = parameters[self._param_labels[c.FLOAT_POT]]
        V = (v_f - v) / T_e

        return I_0 * (1 - np.exp(-V) + np.where(v <= v_f, (a * np.float_power(np.absolute(V), [0.75])), 0))

    def get_a_index(self):
        return self._param_labels[c.SHEATH_EXP]

    def generate_guess(self, temperature=None, floating_potential=None, isat=None, sheath_exp_param=None):
        return self.clip_guess([isat, sheath_exp_param, temperature, floating_potential])


DEFAULT_DENS_CALC_OPTIONS = {
    'gamma_i': 1,
    'mass': 1,
    'Z': 1,
}


class ExperimentalIVFitter(IVFitter):
    """
    IV Fitter implementation using an experimental, 4 parameter IV Curve fitting method.
    """
    def __init__(self, probe=None, theta=10.0, dens_calc_opts=None, sheath_exp_form='rotated'):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.ELEC_TEMP: 1,
            c.FLOAT_POT: 2,
            'c_1': 3,
            'c_2': 4,
        }
        self.default_values = (30.0, 1, -1, 2.0, 1.0)
        self.default_bounds = (
            (-np.inf,       0, -np.inf,       0,       0),
            ( np.inf,  np.inf,  np.inf,  np.inf,  np.inf)
        )
        self.probe = probe
        self.theta = np.radians(theta)
        self.sheath_exp_form = sheath_exp_form
        if dens_calc_opts is None:
            self.dens_calc_opts = DEFAULT_DENS_CALC_OPTIONS
        else:
            self.dens_calc_opts = dens_calc_opts
        self.name = '5 Parameter Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_f = parameters[self._param_labels[c.FLOAT_POT]]
        c_1 = parameters[self._param_labels['c_1']]
        c_2 = parameters[self._param_labels['c_2']]
        V = (v_f - v) / T_e

        n_e = self.probe.get_density(np.abs(I_0), T_e, self.theta, **self.dens_calc_opts)
        a = self.probe.get_sheath_exp_param(T_e, n_e, self.theta, c_1=c_1, c_2=c_2, form=self.sheath_exp_form)
        print(a, n_e)
        # return I_0 * (1 - np.exp(-V)) * (1 + (np.where(v <= v_f, (a * np.float_power(np.absolute(V), [0.75])), 0)))
        return I_0 * (1 - np.exp(-V)) * (1 + (a * np.float_power(np.absolute(V), [0.75])))

    def get_a_index(self):
        return self._param_labels[c.SHEATH_EXP]

    def generate_guess(self, temperature=None, floating_potential=None, isat=None, sheath_exp_param=None):
        return self.clip_guess([isat, sheath_exp_param, temperature, floating_potential])


class Experimental3IVFitter(IVFitter):
    """
    IV Fitter implementation using an experimental, 4 parameter IV Curve fitting method.
    """
    def __init__(self, probe=None, theta=10.0, dens_calc_opts=None, sheath_exp_form='rotated', c_1=2.0, c_2=1.0):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.ELEC_TEMP: 1,
            c.FLOAT_POT: 2,
        }
        self.default_values = (30.0, 1, -1)
        self.default_bounds = (
            (-np.inf,       0, -np.inf),
            ( np.inf,  np.inf,  np.inf)
        )
        self.probe = probe
        self.theta = np.radians(theta)
        self.sheath_exp_form = sheath_exp_form
        self.c_1 = c_1
        self.c_2 = c_2
        if dens_calc_opts is None:
            self.dens_calc_opts = DEFAULT_DENS_CALC_OPTIONS
        else:
            self.dens_calc_opts = dens_calc_opts
        self.name = 'Experimental 3 Parameter Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_f = parameters[self._param_labels[c.FLOAT_POT]]
        V = (v_f - v) / T_e

        n_e = self.probe.get_density(np.abs(I_0), T_e, self.theta, **self.dens_calc_opts)
        a = self.probe.get_sheath_exp_param(T_e, n_e, self.theta, c_1=self.c_1, c_2=self.c_2, form=self.sheath_exp_form)

        return I_0 * (1 - np.exp(-V)) * (1 + (a * np.float_power(np.absolute(V), [0.75])))

    def get_a_index(self):
        return self._param_labels[c.SHEATH_EXP]

    def generate_guess(self, temperature=None, floating_potential=None, isat=None, sheath_exp_param=None):
        return self.clip_guess([isat, sheath_exp_param, temperature, floating_potential])


class SimpleIVFitter(IVFitter):
    """
    Fitter object for a 3-parameter IV characteristic fit
    """
    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.ELEC_TEMP: 1,
            c.FLOAT_POT: 2
        }
        self.default_values = (1.0,  5, -1)
        self.default_bounds = (
            (-np.inf,       0, -np.inf),
            ( np.inf,  np.inf,  np.inf)
        )
        self.name = '3 Parameter Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_f = parameters[self._param_labels[c.FLOAT_POT]]
        V = (v_f - v) / T_e

        return I_0 * (1 - np.exp(-V))

    def generate_guess(self, temperature=None, floating_potential=None, isat=None, sheath_exp_param=None):
        return self.clip_guess([isat, temperature, floating_potential])


class StraightIVFitter(IVFitter):
    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.SHEATH_EXP: 1,
            c.ELEC_TEMP: 2,
            c.FLOAT_POT: 3
        }
        self.default_values = (30.0, 0.02, 1, -1)
        self.default_bounds = (
            (-np.inf,      0,      0, -np.inf),
            ( np.inf, np.inf, np.inf,  np.inf)
        )
        self.name = 'Straight IV Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        a = parameters[self._param_labels[c.SHEATH_EXP]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_f = parameters[self._param_labels[c.FLOAT_POT]]
        V = (v_f - v) / T_e
        return I_0 * (1 + (a * np.float_power(np.absolute(V), [0.75])))

    def get_a_index(self):
        return self._param_labels[c.SHEATH_EXP]

    def generate_guess(self, temperature=None, floating_potential=None, isat=None, sheath_exp_param=None):
        return self.clip_guess([isat, sheath_exp_param, temperature, floating_potential])


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

    def fit_iv_data(self, iv_data, initial_vals=None, bounds=None, trim_fl=False, print_fl=False):
        assert isinstance(iv_data, IVData)
        potential = np.power(np.abs(iv_data[c.POTENTIAL]), 0.75)
        current = iv_data[c.ION_CURRENT]
        return self.fit(potential, current, initial_vals, bounds, print_fl=print_fl)

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        a = parameters[self._param_labels[c.SHEATH_EXP]]
        return I_0 * (1 + (a * v))

    def get_isat_index(self):
        return self._param_labels[c.ION_SAT]

    def get_a_index(self):
        return self._param_labels[c.SHEATH_EXP]

    def generate_guess(self, temperature=None, floating_potential=None, isat=None, sheath_exp_param=None):
        return self.clip_guess([isat, sheath_exp_param])


class ElectronCurrentFitter(IVFitter):
    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ELEC_TEMP: 0,
            # c.FLOAT_POT: 1,
            c.ION_SAT: 1,
        }
        self.default_values = (1.0, 500.0)
        self.default_bounds = (
            (      0, -np.inf),
            ( np.inf,  np.inf),
        )
        self.name = 'Electron Current Fit'

    def fit_function(self, v, *parameters):
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        # V_f = parameters[self._param_labels[c.FLOAT_POT]]
        I_0 = parameters[self._param_labels[c.ION_SAT]]

        return I_0 * np.exp(v / T_e)

    def generate_guess(self, temperature=None, floating_potential=None, isat=None, sheath_exp_param=None):
        return self.clip_guess([isat, sheath_exp_param])


class ElectronCurrentSEFitter(IVFitter):
    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ELEC_TEMP: 0,
            c.FLOAT_POT: 1,
            c.ION_SAT: 2,
            c.SHEATH_EXP: 3
        }
        self.default_values = (1.0, 30.0, -1, 0.0204)
        self.default_bounds = (
            (      0, -np.inf, -np.inf,      0),
            ( np.inf,  np.inf,  np.inf, np.inf),
        )
        self.name = 'Electron Current Fit'

    def fit_function(self, v, *parameters):
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_f = parameters[self._param_labels[c.FLOAT_POT]]
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        a = parameters[self._param_labels[c.SHEATH_EXP]]
        V = (v + v_f) / T_e

        return I_0 * np.exp(V) * (1 + (a * np.float_power(np.absolute(V), [0.75])))

    def generate_guess(self, temperature=None, floating_potential=None, isat=None, sheath_exp_param=None):
        return self.clip_guess([temperature, floating_potential, isat, sheath_exp_param])


class PartialIVFitter(IVFitter):
    """
    IV Fitter implementation utilising a partial, 4 parameter IV Curve fitting method.
    """

    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.SHEATH_EXP: 1,
            c.ELEC_TEMP: 2,
        }
        self.default_values = (30.0, 0.0204, 1)
        self.default_bounds = (
            (-np.inf, 0, 0),
            (np.inf, 1.5, np.inf)
        )
        self.name = 'Partial 4 Parameter Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        a = parameters[self._param_labels[c.SHEATH_EXP]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        V = -v / T_e

        return I_0 * (1 - np.exp(-V) + np.where(v <= 0, (a * np.float_power(np.absolute(V), [0.75])), 0))

    def get_a_index(self):
        return self._param_labels[c.SHEATH_EXP]

    def generate_guess(self, temperature=None, floating_potential=None, isat=None, sheath_exp_param=None):
        return self.clip_guess([isat, sheath_exp_param, temperature])


class NormalisedIVFitter(IVFitter):
    """
    IV Fitter implementation utilising a normalised 2 parameter IV Curve fitting
    method. Voltage is already normalised to V = (V_probe - V_wall) / T_e
    """

    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.SHEATH_EXP: 1,
        }
        self.default_values = (30.0, 0.0204)
        self.default_bounds = (
            (-np.inf, 0),
            (np.inf, 1.5)
        )
        self.name = 'Normalised Parameter Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        a = parameters[self._param_labels[c.SHEATH_EXP]]

        return I_0 * (1 - np.exp(v) + np.where(v <= 0, (a * np.float_power(np.absolute(v), [0.75])), 0))

    def get_a_index(self):
        return self._param_labels[c.SHEATH_EXP]

    def generate_guess(self, temperature=None, floating_potential=None, isat=None, sheath_exp_param=None):
        return self.clip_guess([isat, sheath_exp_param])



# --- Two Temperature IV Fitters --- #

class TwoTempIVFitter(IVFitter):
    """
    IV Fitter implementation utilising the full, 4 parameter IV Curve fitting
    method with 2 temperatures

    """

    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.SHEATH_EXP: 1,
            c.ELEC_TEMP: 2,
            c.FLOAT_POT: 3,
            c.TTIV_SCALER: 4,
            c.TTIV_TEMP: 5,
            c.TTIV_VFLOAT: 6
        }
        self.default_values = (30.0, 0.0204, 1, -1, 0.1, 8, -1)
        self.default_bounds = (
            (-np.inf,      0,      0, -np.inf,      0,       0, -np.inf),
            ( np.inf, np.inf, np.inf,  np.inf, np.inf,  np.inf,  np.inf)
        )
        self.name = '2-Temp 4-Param Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        a = parameters[self._param_labels[c.SHEATH_EXP]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_f = parameters[self._param_labels[c.FLOAT_POT]]
        alpha = parameters[self._param_labels[c.TTIV_SCALER]]
        T_e2 = parameters[self._param_labels[c.TTIV_TEMP]]
        v_f2 = parameters[self._param_labels[c.TTIV_VFLOAT]]
        V1 = (v_f - v) / T_e
        V2 = (v_f2 - v) / T_e2
        return I_0 * (1 - np.exp(-V2) - (alpha * np.exp(-V1)) + (a * np.float_power(np.absolute(V1 + V2), [0.75])))


class TwoTempSimpIVFitter(IVFitter):
    """
    IV Fitter implementation utilising the 3-parameter IV Curve fitting method
    with 2 temperatures

    """

    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.ELEC_TEMP: 1,
            c.FLOAT_POT: 2,
            c.TTIV_SCALER: 3,
            c.TTIV_TEMP: 4,
            c.TTIV_VFLOAT: 5
        }
        self.default_values = (30.0, 1, -1, 0.1, 8, -1)
        self.default_bounds = (
            (-np.inf,      0, -np.inf,      0,       0, -np.inf),
            ( np.inf, np.inf,  np.inf, np.inf,  np.inf,  np.inf)
        )
        self.name = '2-Temp 3-Param Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_f = parameters[self._param_labels[c.FLOAT_POT]]
        alpha = parameters[self._param_labels[c.TTIV_SCALER]]
        T_e2 = parameters[self._param_labels[c.TTIV_TEMP]]
        v_f2 = parameters[self._param_labels[c.TTIV_VFLOAT]]
        V1 = (v_f - v) / T_e
        V2 = (v_f2 - v) / T_e2
        return I_0 * (1 - np.exp(-V2) - (alpha * np.exp(-V1)))


class TwoTempImpurityIVFitter(IVFitter):
    """
    IV Fitter implementation utilising two independent 3-parameter IV curves to
    estimate the response of a simple impurity species

    """

    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.ELEC_TEMP: 1,
            c.FLOAT_POT: 2,
            c.TTIV_ISAT: 3,
            c.TTIV_TEMP: 4,
            c.TTIV_VFLOAT: 5
        }
        self.default_values = (30.0, 1, -1, 15, 1.5, -1)
        self.default_bounds = (
            (0.1, 0, -np.inf, 0.1, 0, -np.inf),
            (np.inf, np.inf, 10, np.inf, np.inf, 10)
        )
        self.name = '2 Independent 3-Param Fits'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_f = parameters[self._param_labels[c.FLOAT_POT]]
        I_02 = parameters[self._param_labels[c.TTIV_ISAT]]
        T_e2 = parameters[self._param_labels[c.TTIV_TEMP]]
        v_f2 = parameters[self._param_labels[c.TTIV_VFLOAT]]
        V1 = (v_f - v) / T_e
        V2 = (v_f2 - v) / T_e2
        return (I_0 * (1 - np.exp(-V1))) + (I_02 * (1 - np.exp(-V2)))


class SEEIVFitter(IVFitter):
    """
    IV Fitter implementation utilising the 3=parameter IV Curve fitting method
    with an additional term for a Secondary Electron Emission Coefficient

    """

    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.ION_SAT: 0,
            c.ELEC_TEMP: 1,
            c.FLOAT_POT: 2,
            c.SSE_PARAM: 3
        }
        self.default_values = (30.0, 1, -1, 0)
        self.default_bounds = (
            (0.1, 0, -np.inf, 0.0,),
            (np.inf, np.inf, 10, 1.0,)
        )
        self.name = 'SEE 3-Param Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_f = parameters[self._param_labels[c.FLOAT_POT]]
        delta = parameters[self._param_labels[c.SSE_PARAM]]
        V1 = (v_f - v) / T_e
        return I_0 * (1 - ((1 - delta) * np.exp(-V1)))


# --- Maxwellian Fitters --- #

class MaxwellianVelFitter(GenericCurveFitter):
    def __init__(self, si_units=False, mu=P_E_MASS_RATIO):
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
            self.temp_conversion = BOLTZMANN
        else:
            self.temp_conversion = ELEM_CHARGE

        self.mass = mu * ELECTRON_MASS
        self.name = '3D Maxwellian Distribution Fit'

    def fit_function(self, v, *parameters):
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        a = self.mass / (2 * self.temp_conversion * T_e)
        return np.power(a / np.pi, 1.5) * 4 * np.pi * np.power(v, 2) * np.exp(-a * np.power(v, 2))

    def get_temp_index(self):
        return self._param_labels[c.ELEC_TEMP]

    def get_flow_velocity(self):
        return self._param_labels[c.FLOW_VEL]


class GenericGaussianFitter(GenericCurveFitter):
    """
    More generic gaussian function fitter. Variables are the amplitude (A), width (s - optionally able to be defined as
    the fwhm) and mean (x_0). Function is of the form:

                  ( 1 ( x - x_0 )^2 )
    f(x) = A * exp( - (---------)   )
                  ( 2 (    s    )   )
    """
    def __init__(self, fwhm_fl=False):
        super().__init__()
        self._param_labels = {
            c.AMPLITUDE: 0,
            c.ST_DEV: 1,
            c.OFFSET_X: 2
        }
        self.default_values = (1.0, 1.0, 0.0)
        self.default_bounds = (
            (0.0, -np.inf, -np.inf),
            (np.inf, np.inf, np.inf)
        )
        self.fwhm_fl = fwhm_fl
        self.name = 'Gaussian Function Fit'

    def fit_function(self, v, *parameters):
        amplitude = parameters[self._param_labels[c.AMPLITUDE]]
        if self.fwhm_fl:
            sigma = parameters[self._param_labels[c.ST_DEV]] / 2.35482
        else:
            sigma = parameters[self._param_labels[c.ST_DEV]]
        mu = parameters[self._param_labels[c.OFFSET_X]]
        return amplitude * np.exp(-0.5 * np.power((v - mu) / sigma, 2))


class NormalisedGaussianFitter(GenericCurveFitter):
    """
    Normalised gaussian function fitter class. Implements a generic gaussian function with normalised area, i.e. the
    amplitude is controlled solely by the standard deviation.
    """
    def __init__(self, fwhm_fl=False):
        super().__init__()
        self._param_labels = {
            c.ST_DEV: 0,
            c.OFFSET_X: 1
        }
        self.default_values = (1.0, 0.0)
        self.default_bounds = (
            (0, -np.inf),
            (np.inf, np.inf)
        )
        self.fwhm_fl = fwhm_fl
        self.name = 'Gaussian Function Fit'

    def fit_function(self, v, *parameters):
        sigma = parameters[self._param_labels[c.ST_DEV]]
        if self.fwhm_fl:
            sigma = sigma / 2.35482
        amplitude = 1 / (np.sqrt(2 * np.pi) * sigma)
        mu = parameters[self._param_labels[c.OFFSET_X]]
        return amplitude * np.exp(-0.5 * np.power((v - mu) / sigma, 2))


class NormalisedBimodalGaussianFitter(GenericCurveFitter):
    """
    A bimodal gaussian function made from two normalised gaussians combined.
    Implements 2 generic gaussian function with normalised area, i.e. the
    amplitude is controlled solely by the standard deviation.
    """
    def __init__(self, fwhm_fl=False):
        super().__init__()
        self._param_labels = {
            f'{c.ST_DEV}_0': 0,
            f'{c.OFFSET_X}_0': 1,
            f'{c.ST_DEV}_1': 2,
            f'{c.OFFSET_X}_1': 3,
        }
        self.default_values = (1.0, 0.0, 1.0, 0.0)
        self.default_bounds = (
            (0, -np.inf, 0, -np.inf),
            (np.inf, np.inf, np.inf, np.inf)
        )
        self.fwhm_fl = fwhm_fl
        self.name = 'Bimodal Gaussian Function Fit'

    def fit_function(self, v, *parameters):
        sigma_0 = parameters[self._param_labels[f'{c.ST_DEV}_0']]
        sigma_1 = parameters[self._param_labels[f'{c.ST_DEV}_1']]
        if self.fwhm_fl:
            sigma_0 = sigma_0 / 2.35482
            sigma_1 = sigma_1 / 2.35482
        amplitude_0 = 1 / (np.sqrt(2 * np.pi) * sigma_0)
        amplitude_1 = 1 / (np.sqrt(2 * np.pi) * sigma_1)
        mu_0 = parameters[self._param_labels[f'{c.OFFSET_X}_0']]
        mu_1 = parameters[self._param_labels[f'{c.OFFSET_X}_1']]
        return (amplitude_0 * np.exp(-0.5 * np.power((v - mu_0) / sigma_0, 2))
                + amplitude_1 * np.exp(-0.5 * np.power((v - mu_1) / sigma_1, 2)))


class GaussianFitter(GenericCurveFitter):
    def __init__(self, si_units=False, mu=P_E_MASS_RATIO, v_scale=1):
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
            self.temp_conversion = BOLTZMANN
        else:
            self.temp_conversion = ELEM_CHARGE

        self.v_scale = v_scale
        self.mass = mu * ELECTRON_MASS
        self.name = 'Gaussian Dist Fit'

    @staticmethod
    def _get_v0_guess(hist, hist_bins):
        # estimate values for the initial guess based on the mid point between the points of greatest +ve and -ve
        # curvature
        grad_fv = np.gradient(hist)
        sm_grad_fv = sig.savgol_filter(grad_fv, 21, 2)
        min = np.argmin(sm_grad_fv)
        max = np.argmax(sm_grad_fv)

        v_0_guess = hist_bins[int((max + min) / 2)]
        return v_0_guess

    def fit(self, x_data, y_data, initial_vals=None, bounds=None, temp=None, sigma=None):
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
            v_0 = self._get_v0_guess(y_data, x_data)
            if temp:
                initial_vals = [temp / (self.v_scale ** 2), v_0]
            else:
                initial_vals = [self.default_values[temp_ind] / (self.v_scale ** 2), v_0]

        fitdata = super().fit(x_data, y_data, initial_vals=initial_vals, bounds=bounds, sigma=sigma)

        # Convert values back to pre-scaled magnitude
        fitdata.fit_params[temp_ind].value *= self.v_scale ** 2
        fitdata.fit_params[temp_ind].error *= self.v_scale ** 2
        fitdata.fit_params[flow_ind].value *= self.v_scale
        fitdata.fit_params[flow_ind].error *= self.v_scale
        return fitdata

    def fit_function(self, v, *parameters):
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_0 = parameters[self._param_labels[c.FLOW_VEL]]
        a = self.mass / (2 * self.temp_conversion * T_e)
        return np.squeeze(np.sqrt(a / np.pi) * np.exp(-a * np.power(v - v_0, 2)))

    def set_mass_scaler(self, scaling_val):
        self.mass = ELECTRON_MASS * scaling_val

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
    def __init__(self, mu=P_E_MASS_RATIO, v_scale=1):
        super().__init__(mu=mu, v_scale=v_scale, si_units=False)
        self.name = 'Gauss. Ion V-Dist (eV)'

    def fit_function(self, v, *parameters):
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        v_0 = parameters[self._param_labels[c.FLOW_VEL]]
        a = self.mass / (2 * ELEM_CHARGE * T_e)
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
    def __init__(self, mu=P_E_MASS_RATIO, si_units=False, v_scale=1):
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


class StraightLineFitter(GenericCurveFitter):
    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.GRADIENT: 0,
            c.OFFSET_Y: 1
        }
        self.default_values = [1.0, 0.0]
        self.default_bounds = [
            [-np.inf, -np.inf],
            [ np.inf,  np.inf]
        ]
        self.name = 'Straight Line Fit'

    def fit_function(self, v, *parameters):
        m = parameters[self._param_labels[c.GRADIENT]]
        y_0 = parameters[self._param_labels[c.OFFSET_Y]]
        return (m * v) + y_0


class ExponentialFitter(GenericCurveFitter):
    def __init__(self):
        super().__init__()
        self._param_labels = {
            c.AMPLITUDE: 0,
            c.EXP_SCALER: 1,
            c.OFFSET_Y: 2,
            c.OFFSET_X: 3
        }
        self.default_values = [1.0, 1.0, 0.0, 0.0]
        self.default_bounds = [
            [-np.inf, -np.inf, -np.inf, -np.inf],
            [ np.inf,  np.inf,  np.inf,  np.inf]
        ]
        self.name = 'Exponential Fit'

    def fit_function(self, v, *parameters):
        A = parameters[self._param_labels[c.AMPLITUDE]]
        b = parameters[self._param_labels[c.EXP_SCALER]]
        y_0 = parameters[self._param_labels[c.OFFSET_Y]]
        x_0 = parameters[self._param_labels[c.OFFSET_X]]
        return A * np.exp(b * (v + x_0)) + y_0


class TriangleWaveFitter(GenericCurveFitter):
    def __init__(self, frequency=None):
        super().__init__()
        self._param_labels = {
            c.AMPLITUDE: 0,
            c.OFFSET_Y: 1,
            c.OFFSET_X: 2
        }
        self.freq = frequency
        # These values taken from values used for majority of magnum experimental run
        self.default_values = [53.1, 4.34, 0.0]
        self.default_bounds = [
            [   0, -np.inf,      0],
            [1000,  np.inf, np.inf]
        ]
        self.name = 'Triangle Wave Fit'

    def fit(self, x_data, y_data, freq=None, initial_vals=None, bounds=None, sigma=None, calc_guess_fl=True):
        """
            Override of the fit method with included provision for automatic finding of the amplitude and period of the
            triangular wave if one is not specified.
            :param x_data:          x-data for fit
            :param y_data:          y-data for fit
            :param freq:            specify a known frequency of the wave, will be automatically found if not
            :param initial_vals:    [optional] initial values to be fed to the fitting algorithm. If left None, by
                                    default amplitude and period will be populated automatically using
                                    get_initial_guess(). This behaviour can be turned off with the calc_guess_fl param.
            :param bounds:          [optional] The bounds for the fitting algorithm. If none are specified then the
                                    default parameters are used.
            :param calc_guess_fl:   [optional] Boolean flag to turn on or off the calculation of an initial guess if one
                                    is not given.
            :return fitdata:        fitdata from fit.
        """
        if freq is not None:
            self.freq = freq

        if not self.freq:
            self.freq = self.get_frequency(x_data, y_data)

        period = 1/(2*self.freq)
        if not initial_vals and calc_guess_fl:
            initial_vals = [*self.get_initial_guess(y_data, x_data)]

        # if not bounds and calc_guess_fl:
        #     bounds = self.default_bounds
        #     bounds[0][0] = period * 0.9
        #     bounds[1][0] = period * 1.1

        # length = len(x_data)
        # first_fit = super().fit(x_data[:int(0.01*length)], y_data[:int(0.01*length)],
        #                         initial_vals=initial_vals, bounds=bounds)
        # first_fit.print_fit_params()
        # second_fit = super().fit(x_data[:int(0.05*length)], y_data[:int(0.05*length)],
        #                          initial_vals=first_fit.fit_params.get_values(), bounds=bounds)
        # second_fit.print_fit_params()
        # third_fit = super().fit(x_data[:int(0.2 * length)], y_data[:int(0.2 * length)],
        #                         initial_vals=first_fit.fit_params.get_values(), bounds=bounds)
        # third_fit.print_fit_params()
        # fit_y_data = self.fit_function(x_data, *third_fit.fit_params.get_values())
        # return FitData2(x_data, y_data, fit_y_data, third_fit.fit_params.get_values(), third_fit.fit_params.get_errors(),
        #                 self)
        return super().fit(x_data, y_data, initial_vals=initial_vals, bounds=bounds, sigma=sigma)

    @staticmethod
    def get_initial_guess(voltage, time):
        # Get a guess for the period and amplitude by smoothing and getting the locations of peaks and troughs with
        # argrelmax/min. Doesn't need to be exact as it's only a starting point for the fitting function.
        smoothed_voltage = sig.savgol_filter(voltage, 21, 2)
        top = sig.argrelmax(smoothed_voltage, order=20)
        bottom = sig.argrelmin(smoothed_voltage, order=20)
        peaks = time[np.sort(np.concatenate([top, bottom], 1))[0]]
        print(peaks)

        ampl_guess = (np.mean(voltage[top]) - np.mean(voltage[bottom])) / 2
        period_guess = 2 * np.mean([peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)])
        y_off_guess = ampl_guess + np.mean(voltage[bottom])
        x_off_guess = np.remainder(peaks[0] - time[0], period_guess / 4)
        print(ampl_guess, period_guess, y_off_guess, x_off_guess)
        return ampl_guess, y_off_guess, x_off_guess

    @staticmethod
    def get_frequency(time, voltage, accepted_freqs=None):
        # Take FFT to find frequency of triangle wave
        sp = np.fft.fft(voltage)
        freq = np.fft.fftfreq(len(voltage), time[1] - time[0])

        f = np.abs(freq[np.argmax(np.abs(sp)[1:])])
        if accepted_freqs is not None and isinstance(accepted_freqs, np.ndarray):
            return accepted_freqs[np.abs(accepted_freqs - f).argmin()]
        else:
            return f

    def fit_function(self, v, *parameters):
        a = parameters[self._param_labels[c.AMPLITUDE]]
        p = 1 / self.freq
        y_0 = parameters[self._param_labels[c.OFFSET_Y]]
        x_0 = parameters[self._param_labels[c.OFFSET_X]]
        return (((4 * a) / p) * (np.abs(np.mod(v + (x_0 * p), p) - (p / 2)) - (p / 4))) + y_0

    def get_amplitude_index(self):
        return self._param_labels[c.AMPLITUDE]

    def get_y_offset_index(self):
        return self._param_labels[c.OFFSET_Y]

    def get_x_offset_index(self):
        return self._param_labels[c.OFFSET_X]

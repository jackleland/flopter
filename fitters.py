from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import curve_fit
import scipy.signal as sig
from scipy.interpolate import interp1d

from classes.ivdata import IVData
from classes.fitdata import FitData2, IVFitData
import constants as c
from normalisation import BOLTZMANN, ELECTRON_MASS, ELEM_CHARGE, PROTON_MASS, P_E_MASS_RATIO
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


# --- IV Fitters --- #

class IVFitter(GenericFitter, ABC):
    _DEFAULT_V_F = -1.0

    def __init__(self, floating_potential=None):
        super().__init__()
        self.v_f = floating_potential

    def fit(self, x_data, y_data, initial_vals=None, bounds=None, print_fl=False):
        if not self.v_f:
            if print_fl:
                print('No floating potential specified, using default value ({}).'.format(self._DEFAULT_V_F))
            self.v_f = self._DEFAULT_V_F
        fit_data = super().fit(x_data, y_data, initial_vals=initial_vals, bounds=bounds)
        return IVFitData.from_fit_data(fit_data)

    def fit_iv_data(self, iv_data, initial_vals=None, bounds=None, trim_fl=False, print_fl=False):
        assert isinstance(iv_data, IVData)
        if trim_fl:
            iv_data.trim()
        potential = iv_data[c.POTENTIAL]
        current = iv_data[c.CURRENT]
        return self.fit(potential, current, initial_vals, bounds, print_fl=print_fl)

    def set_floating_pot(self, floating_pot):
        self.v_f = floating_pot

    def autoset_floating_pot(self, iv_data):
        self.v_f = self.find_floating_pot(iv_data)

    @classmethod
    def find_floating_pot(cls, iv_data):
        try:
            iv_interp = interp1d(iv_data[c.CURRENT], iv_data[c.POTENTIAL])
            v_f = iv_interp([0.0])
        except ValueError:
            print('V_f could not be found effectively, returning default value')
            v_f = [cls._DEFAULT_V_F]
        return v_f


class FullIVFitter(IVFitter):
    """
    IV Fitter implementation utilising the full, 4 parameter IV Curve fitting method.
    """
    def __init__(self, floating_potential=None):
        super().__init__(floating_potential=floating_potential)
        self._param_labels = {
            c.ION_SAT: 0,
            c.SHEATH_EXP: 1,
            c.ELEC_TEMP: 2
        }
        self.default_values = (30.0, 0.0204, 1)
        self.default_bounds = (
            (-np.inf,       0,       0),
            ( np.inf,  np.inf,  np.inf)
        )
        self.name = '4 Parameter Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        a = parameters[self._param_labels[c.SHEATH_EXP]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        V = (self.v_f - v) / T_e
        return I_0 * (1 - np.exp(-V) + (a * np.float_power(np.absolute(V), [0.75])))

    def get_param_index(self, label):
        return self._param_labels[label]

    def get_temp_index(self):
        return self._param_labels[c.ELEC_TEMP]

    def get_isat_index(self):
        return self._param_labels[c.ION_SAT]

    def get_a_index(self):
        return self._param_labels[c.SHEATH_EXP]


class SimpleIVFitter(IVFitter):
    def __init__(self, floating_potential=None):
        super().__init__(floating_potential=floating_potential)
        self._param_labels = {
            c.ION_SAT: 0,
            c.ELEC_TEMP: 1
        }
        self.default_values = (30.0,  1)
        self.default_bounds = (
            (-np.inf,       0),
            ( np.inf,  np.inf)
        )
        self.name = '3 Parameter Fit'

    def fit_function(self, v, *parameters):
        I_0 = parameters[self._param_labels[c.ION_SAT]]
        T_e = parameters[self._param_labels[c.ELEC_TEMP]]
        V = (self.v_f - v) / T_e
        return I_0 * (1 - np.exp(-V))

    def get_temp_index(self):
        return self._param_labels[c.ELEC_TEMP]

    def get_isat_index(self):
        return self._param_labels[c.ION_SAT]


class IonCurrentSEFitter(IVFitter):
    def __init__(self):
        super().__init__(floating_potential=None)
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


# --- Maxwellian Fitters --- #

class MaxwellianVelFitter(GenericFitter):
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


class GenericGaussianFitter(GenericFitter):
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

    def fit_function(self, v, parameters):
        amplitude = parameters[self._param_labels[c.AMPLITUDE]]
        if self.fwhm_fl:
            sigma = parameters[self._param_labels[c.ST_DEV]] / 2.35482
        else:
            sigma = parameters[self._param_labels[c.ST_DEV]]
        mu = parameters[self._param_labels[c.OFFSET_X]]
        return amplitude * np.exp(-0.5 * np.power((v - mu) / sigma, 2))


class NormalisedGaussianFitter(GenericFitter):
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

    def fit_function(self, v, parameters):
        sigma = parameters[self._param_labels[c.ST_DEV]]
        if self.fwhm_fl:
            sigma = sigma / 2.35482
        amplitude = 1 / (np.sqrt(2 * np.pi) * sigma)
        mu = parameters[self._param_labels[c.OFFSET_X]]
        return amplitude * np.exp(-0.5 * np.power((v - mu) / sigma, 2))


class GaussianFitter(GenericFitter):
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
            v_0 = self._get_v0_guess(y_data, x_data)
            if temp:
                initial_vals = [temp / (self.v_scale ** 2), v_0]
            else:
                initial_vals = [self.default_values[temp_ind] / (self.v_scale ** 2), v_0]

        fitdata = super().fit(x_data, y_data, initial_vals=initial_vals, bounds=bounds)

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


class TriangleWaveFitter(GenericFitter):
    def __init__(self, frequency=None):
        super().__init__()
        self._param_labels = {
            c.PERIOD: 0,
            c.AMPLITUDE: 1,
            c.OFFSET_Y: 2,
            c.OFFSET_X: 3
        }
        self.freq = frequency
        # These values taken from values used for majority of magnum experimental run
        self.default_values = [0.025, 53.1, 4.34, 0.0]
        self.default_bounds = [
            [     0,    0, -np.inf,      0],
            [np.inf, 1000,  np.inf, np.inf]
        ]
        self.name = 'Triangle Wave'

    def fit(self, x_data, y_data, freq=None, initial_vals=None, bounds=None):
        """
            Override of the fit method with included provision for automatic finding of the amplitude and period of the
            triangular wave if one is not specified.
            :param x_data:          x-data for fit
            :param y_data:          y-data for fit
            :param freq:            specify a known frequency of the wave, will be automatically found if not
            :param initial_vals:    [optional] initial values to be fed to the fitting algorithm. If left None,
                                    amplitude and period will be populated automatically using get_initial_guess().
            :param bounds:          [optional] The bounds for the fitting algorithm. If none are specified then the
                                    default parameters are used.
            :return fitdata:        fitdata from fit.
        """

        if not self.freq:
            self.freq = self.get_frequency(x_data, y_data)

        period = 1/(2*self.freq)
        if not initial_vals:
            initial_vals = [period, *self.get_initial_guess(y_data, x_data)]

        if not bounds:
            bounds = self.default_bounds
            bounds[0][0] = period * 0.9
            bounds[1][0] = period * 1.1

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
        return super().fit(x_data, y_data, initial_vals=initial_vals, bounds=bounds)

    @staticmethod
    def get_initial_guess(voltage, time):
        # Get a guess for the period and amplitude by smoothing and getting the locations of peaks and troughs with
        # argrelmax/min. Doesn't need to be exact as it's only a starting point for the fitting function.
        smoothed_voltage = sig.savgol_filter(voltage, 21, 2)
        top = sig.argrelmax(smoothed_voltage, order=20)
        bottom = sig.argrelmin(smoothed_voltage, order=20)
        peaks = time[np.sort(np.concatenate([top, bottom], 1))[0]]

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

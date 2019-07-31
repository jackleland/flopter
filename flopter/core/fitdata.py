from flopter.core import constants as c
import matplotlib.pyplot as plt


class FitParam(object):
    """
        Fit parameter object containing a value and its associated error, for use in storing fit parameters used in
        fitting.

        Can be retrieved separately or as a pair.
    """
    def __init__(self, value, error):
        self.value = value
        self.error = error

    def get_ve_pair(self):
        return self.value, self.error


class FitParamList(list):
    """
        Overridden list object for storing FitParam objects. Maintains functionality for retrieving separated lists
        simply.
    """
    def __init__(self, values, errors):
        super().__init__()
        if len(values) != len(errors):
            raise ValueError('Value and error lists need to be the same length.')
        for i in range(len(values)):
            param = FitParam(values[i], errors[i])
            self.append(param)

    def get_values(self):
        return [param.value for param in self]

    def get_errors(self):
        return [param.error for param in self]

    def split(self):
        return self.get_values(), self.get_errors()


class FitData2(object):
    def __init__(self, raw_x, raw_y, fit_y, fit_values, fit_errors, fitter, sigma=None, chi2=None, reduced_chi2=None):
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.fit_y = fit_y
        self.fit_params = FitParamList(fit_values, fit_errors)
        self.fitter = fitter
        self.sigma = sigma
        self.chi2 = chi2
        self.reduced_chi2 = reduced_chi2

    def plot(self, fig=None, show_fl=True):
        if not fig:
            plt.figure()
        plt.plot(*self.get_raw_plottables(), 'x')
        plt.plot(*self.get_fit_plottables(), label=self.get_param_str())
        plt.xlabel(c.RAW_X)
        plt.ylabel(self.fitter.name)
        plt.legend()
        if show_fl:
            plt.show()

    def get_raw_plottables(self):
        return self.raw_x, self.raw_y

    def get_fit_plottables(self):
        return self.raw_x, self.fit_y

    def get_fit_params(self):
        return self.fit_params

    def get_param(self, label, errors_fl=False):
        index = self.fitter.get_param_index(label)
        if errors_fl:
            return self.fit_params[index]
        else:
            return self.fit_params[index].value

    def get_param_val(self, label):
        index = self.fitter.get_param_index(label)
        return self.fit_params[index].value

    def get_param_err(self, label):
        index = self.fitter.get_param_index(label)
        return self.fit_params[index].error

    def get_fitter(self):
        return self.fitter

    def fit_function(self, x):
        func = self.fitter.fit_function
        return func(x, *self.fit_params.get_values())

    def print_fit_params(self):
        param_labels = self.fitter.get_param_labels()
        param_values = self.fit_params.get_values()
        param_errors = self.fit_params.get_errors()
        print("")
        print("FIT PARAMETERS")
        for i in range(len(param_labels)):
            print("{a} = {b} +/- {c}".format(a=param_labels[i], b=param_values[i], c=param_errors[i]))
        print("")
        if self.reduced_chi2 is not None:
            print('Reduced chi^2 = {:.3f} \n'.format(self.reduced_chi2))

    def get_param_str(self):
        return ''.join(['{}:{:.3f} '.format(label, self.fit_params.get_values()[i], self.fit_params.get_errors()[i])
                        for i, label in enumerate(self.fitter.get_param_labels())])

    def get_residual(self):
        return self.fit_y - self.raw_y

    def to_dict(self):
        dictionary = {
            c.RAW_X: self.raw_x,
            c.RAW_Y: self.raw_y,
            c.FIT_Y: self.fit_y
        }
        if self.sigma is not None:
            dictionary[c.SIGMA] = self.sigma
        if self.chi2:
            dictionary[c.CHI2] = self.chi2
        if self.reduced_chi2:
            dictionary[c.REDUCED_CHI2] = self.reduced_chi2

        param_labels = self.fitter.get_param_labels()
        param_values = self.fit_params.get_values()
        param_errors = self.fit_params.get_errors()
        for i in range(len(param_values)):
            dictionary[param_labels[i]] = param_values[i]
            dictionary['d_{}'.format(param_labels[i])] = param_errors[i]
        return dictionary


class IVFitData(FitData2):
    def __init__(self, raw_voltage, raw_current, fit_current, fit_params, fit_stdevs, fitter, sigma=None, chi2=None,
                 reduced_chi2=None):
        super().__init__(raw_voltage, raw_current, fit_current, fit_params, fit_stdevs, fitter, sigma=sigma, chi2=chi2,
                         reduced_chi2=reduced_chi2)

    def get_temp(self, errors_fl=False):
        return self.get_param(c.ELEC_TEMP, errors_fl)

    def get_temp_err(self):
        return self.get_param(c.ELEC_TEMP, True).error

    def get_isat(self, errors_fl=False):
        return self.get_param(c.ION_SAT, errors_fl)

    def get_isat_err(self):
        return self.get_param(c.ION_SAT, True).error

    def get_sheath_exp(self, errors_fl=False):
        return self.get_param(c.SHEATH_EXP, errors_fl)

    def get_sheath_exp_err(self):
        return self.get_param(c.SHEATH_EXP, True).error

    def get_floating_pot(self):
        return self.fitter.v_f

    @classmethod
    def from_fit_data(cls, fit_data_instance):
        """Create IVFitData object from already instantiated FitData2 object."""
        class_data = [
            fit_data_instance.raw_x,
            fit_data_instance.raw_y,
            fit_data_instance.fit_y,
            fit_data_instance.fit_params.get_values(),
            fit_data_instance.fit_params.get_errors(),
            fit_data_instance.fitter
        ]
        optional_class_data = {
            c.SIGMA: fit_data_instance.sigma,
            c.CHI2: fit_data_instance.chi2,
            c.REDUCED_CHI2: fit_data_instance.reduced_chi2
        }
        return cls(*class_data, **optional_class_data)

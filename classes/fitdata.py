from constants import ELEC_TEMP, ION_SAT, SHEATH_EXP, FLOAT_POT


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

    def get_value(self):
        return self.value

    def get_error(self):
        return self.error


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
        return [param.get_value() for param in self]

    def get_errors(self):
        return [param.get_error() for param in self]


class FitData2(object):
    def __init__(self, raw_x, raw_y, fit_y, fit_values, fit_errors, fitter):
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.fit_y = fit_y
        self.fit_params = FitParamList(fit_values, fit_errors)
        self.fitter = fitter

    def get_raw_plottables(self):
        return self.raw_x, self.raw_y

    def get_fit_plottables(self):
        return self.raw_x, self.fit_y

    def get_fit_params(self):
        return self.fit_params

    def get_param(self, label, errors_fl=True):
        index = self.fitter.get_param_index(label)
        if errors_fl:
            return self.fit_params[index]
        else:
            return self.fit_params[index].get_value()

    def get_fitter(self):
        return self.fitter

    def print_fit_params(self):
        param_labels = self.fitter.get_param_labels()
        param_values = self.fit_params.get_values()
        param_errors = self.fit_params.get_errors()
        print("")
        print("FIT PARAMETERS")
        for i in range(len(param_labels)):
            print("{a} = {b} +/- {c}".format(a=param_labels[i], b=param_values[i], c=param_errors[i]))
        print("")


class IVFitData(FitData2):
    def __init__(self, raw_voltage, raw_current, fit_current, fit_params, fit_stdevs, fitter):
        super().__init__(raw_voltage, raw_current, fit_current, fit_params, fit_stdevs, fitter)

    def get_temp(self, errors_fl=True):
        return self.get_param(ELEC_TEMP, errors_fl)

    def get_isat(self, errors_fl=True):
        return self.get_param(ION_SAT, errors_fl)

    def get_sheath_exp(self, errors_fl=True):
        return self.get_param(SHEATH_EXP, errors_fl)

    def get_floating_pot(self, errors_fl=True):
        return self.get_param(FLOAT_POT, errors_fl)

    @classmethod
    def from_fit_data(cls, fit_data_instance):
        """Create IVFitData object from already instantiated FitData2 object."""
        class_data = [
            fit_data_instance.raw_x,
            fit_data_instance.raw_y,
            fit_data_instance.fit_y,
            fit_data_instance.fit_params.get_values,
            fit_data_instance.fit_params.get_errors,
            fit_data_instance.fitter
        ]
        return cls(*class_data)
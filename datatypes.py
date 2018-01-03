import fitters as f
from constants import POTENTIAL, CURRENT, TIME, ELEC_CURRENT, ION_CURRENT


class IVData(dict):
    """
        A dictionary which holds IV specific data, namely
            - Voltage
            - Current
            - Time
            - [Electron Current] (for simulations)
            - [Ion Current] (for simulations)
    """
    def __init__(self, voltage, total_current, time, e_current=None, i_current=None):
        super().__init__([
            (CURRENT, total_current),
            (POTENTIAL, voltage),
            (TIME, time)
        ])
        if e_current is not None:
            self[ELEC_CURRENT] = e_current
        if i_current is not None:
            self[ION_CURRENT] = i_current

    def split(self):
        if ION_CURRENT in self.keys():
            return self[POTENTIAL], self[CURRENT], self[ION_CURRENT]
        else:
            return self[POTENTIAL], self[CURRENT]


class IVFitData(object):
    def __init__(self, raw_voltage, raw_current, fit_current, fit_params, fit_stdevs, fitter):
        self.voltage = raw_voltage
        self.current = raw_current
        self.f_current = fit_current
        self.f_params = fit_params
        self.f_stdevs = fit_stdevs
        self.fitter = fitter

    def get_raw_plottables(self):
        return self.voltage, self.current

    def get_fit_plottables(self):
        return self.voltage, self.f_current

    def get_fit_params(self):
        # TODO: Make a value+error object and return a list of them
        return self.f_params, self.f_stdevs

    def get_param(self, label, errors_fl=True):
        index = self.fitter.get_param_index(label)
        if errors_fl:
            return self.f_params[index], self.f_stdevs[index]
        else:
            return self.f_params[index]

    def get_temp(self, errors_fl=True):
        return self.get_param(f.ELEC_TEMP, errors_fl)

    def get_isat(self, errors_fl=True):
        return self.get_param(f.ION_SAT, errors_fl)

    def get_sheath_exp(self, errors_fl=True):
        return self.get_param(f.SHEATH_EXP, errors_fl)

    def get_floating_pot(self, errors_fl=True):
        return self.get_param(f.FLOAT_POT, errors_fl)

    def get_fitter(self):
        return self.fitter

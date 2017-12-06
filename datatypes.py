
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
            ("I", total_current),
            ("V", voltage),
            ("t", time)
        ])
        if e_current is not None:
            self['I_e'] = e_current
        if i_current is not None:
            self['I_i'] = i_current

    def split(self):
        if 'I_i' in self.keys():
            return self['V'], self['I'], self['I_i']
        else:
            return self['V'], self['I']


class IVFitData(object):
    def __init__(self, raw_voltage, raw_current, fit_current, fit_params, fit_stdevs, fitting_function):
        self.voltage = raw_voltage
        self.current = raw_current
        self.f_current = fit_current
        self.f_params = fit_params
        self.f_stdevs = fit_stdevs
        self.f_function = fitting_function

    def get_raw_plottables(self):
        return self.voltage, self.current

    def get_fit_plottables(self):
        return self.voltage, self.f_current

    def get_fit_params(self):
        return self.f_params, self.f_stdevs

    # def get_temp(self):
    #     return self.f_params[]

    def get_fit_function(self):
        return self.f_function

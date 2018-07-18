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
    _DEFAULT_TRIM_BEG = 0.0
    _DEFAULT_TRIM_END = 1.0

    def __init__(self, voltage, total_current, time, e_current=None, i_current=None,
                 trim_beg=_DEFAULT_TRIM_BEG, trim_end=_DEFAULT_TRIM_END):
        super().__init__([
            (CURRENT, total_current),
            (POTENTIAL, voltage),
            (TIME, time)
        ])
        if e_current is not None:
            self[ELEC_CURRENT] = e_current
        if i_current is not None:
            self[ION_CURRENT] = i_current
        self.trim_beg = trim_beg
        self.trim_end = trim_end

    def split(self):
        if ION_CURRENT in self.keys():
            return self[POTENTIAL], self[CURRENT], self[ION_CURRENT]
        else:
            return self[POTENTIAL], self[CURRENT]

    def trim(self, trim_beg=None, trim_end=None):
        if not trim_beg and not trim_end:
            if self.trim_beg == self._DEFAULT_TRIM_BEG and self.trim_end == self._DEFAULT_TRIM_END:
                print('WARNING: trim values unchanged from default, no trimming will take place')
                return
            else:
                trim_beg = self.trim_beg
                trim_end = self.trim_end

        full_length = len(self[CURRENT])
        start = int(full_length * trim_beg)
        stop = int(full_length * trim_end)
        for key, value in self.items():
            self[key] = value[start:stop]

    def copy(self):
        e_current = None
        i_current = None
        if ELEC_CURRENT in self:
            e_current = self[ELEC_CURRENT]
        if ION_CURRENT in self:
            i_current = self[ION_CURRENT]
        return IVData(self[POTENTIAL], self[CURRENT], self[TIME], e_current=e_current, i_current=i_current,
                      trim_beg=self.trim_beg, trim_end=self.trim_end)

    def multi_fit(self, sat_region=-30):
        """
        Multi-stage fitting method using an initial straight line fit to the saturation region of the IV curve (decided
        by the sat_region kwarg). The fitted I_sat is then left fixed while T_e and a are found with a 2 parameter fit,
        which gives the guess parameters for an unconstrained full IV fit.
        :param sat_region:  Threshold voltage value to start fitting the saturation current to
        :return:            Full 4-param IV fit data
        """
        import numpy as np
        import fitters as f

        # find floating potential
        v_f = f.IVFitter.find_floating_pot(self)
        v_f_pos = np.abs(self[POTENTIAL] - v_f).argmin()

        # find and fit straight section
        str_sec = np.where(self[POTENTIAL] <= sat_region)
        v_ss = self[POTENTIAL][str_sec]
        i_ss = self[CURRENT][str_sec]
        siv_f = f.StraightIVFitter(floating_potential=v_f)
        siv_f_data = siv_f.fit(v_ss, i_ss)

        # Use I_sat value to fit a reduced parameter IV fit
        I_sat_guess = siv_f_data.get_isat().value
        fis_f = f.FullIVFixedISatFitter(I_sat_guess, floating_potential=v_f)
        iv_data_trim = trim_pos(self, 0, v_f_pos)
        first_fit_data = fis_f.fit_iv_data(iv_data_trim)

        # Do a full 4 parameter fit with initial guess params taken from previous fit
        params = [I_sat_guess, *first_fit_data.fit_params.get_values()]
        fitter = f.FullIVFitter(floating_potential=v_f)
        ff_data = fitter.fit_iv_data(iv_data_trim, initial_vals=params)

        return ff_data


def trim(iv_data, trim_beg=0.0, trim_end=1.0):
    if not iv_data or not isinstance(iv_data, IVData):
        raise ValueError('Invalid iv_data given.')
    full_length = len(iv_data[CURRENT])
    start = int(full_length * trim_beg)
    stop = int(full_length * trim_end)
    return trim_pos(iv_data, start, stop)


def trim_pos(iv_data, start, stop):
    """
    Function for trimming an IVData object by array index.
    :param iv_data: IVData object to trim
    :param start:   Start trimming from this index - must be integer
    :param stop:    Stop index - must be integer
    :return: trimmed IVData object with arrays of length (stop - start)
    """
    if not iv_data or not isinstance(iv_data, IVData):
        raise ValueError('Invalid iv_data given.')
    new_iv_data = iv_data.copy()
    for key, value in iv_data.items():
        new_iv_data[key] = value[start:stop]
    return new_iv_data

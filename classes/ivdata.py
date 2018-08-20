import constants as c
import numpy as np
import collections as coll
import pandas as pd


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
    _DEFAULT_STRAIGHT_CUTOFF = -30
    _DEFAULT_STD_SCALER = 1.0

    def __init__(self, voltage, total_current, time, e_current=None, i_current=None, sigma=None, estimate_error_fl=True,
                 sat_region=_DEFAULT_STRAIGHT_CUTOFF, std_err_scaler=_DEFAULT_STD_SCALER,
                 trim_beg=_DEFAULT_TRIM_BEG, trim_end=_DEFAULT_TRIM_END):
        super().__init__([
            (c.CURRENT, total_current),
            (c.POTENTIAL, voltage),
            (c.TIME, time)
        ])
        if e_current is not None:
            self[c.ELEC_CURRENT] = e_current
        if i_current is not None:
            self[c.ION_CURRENT] = i_current
        self.trim_beg = trim_beg
        self.trim_end = trim_end

        if isinstance(sigma, coll.Sized) and len(sigma) == len(voltage):
            self[c.SIGMA] = sigma

        if estimate_error_fl and c.SIGMA not in self:
            str_sec = np.where(self[c.POTENTIAL] <= sat_region)
            i_ss = self[c.CURRENT][str_sec]
            self[c.SIGMA] = std_err_scaler * np.std(i_ss) * np.ones_like(self[c.CURRENT])

        self.untrimmed_items = {}
        for k, v in self.items():
            self.untrimmed_items[k] = v

    def split(self):
        if c.ION_CURRENT in self.keys():
            return self[c.POTENTIAL], self[c.CURRENT], self[c.ION_CURRENT]
        else:
            return self[c.POTENTIAL], self[c.CURRENT]

    def set_trim(self, trim_beg=_DEFAULT_TRIM_BEG, trim_end=_DEFAULT_TRIM_END):
        assert self._DEFAULT_TRIM_BEG <= trim_beg < trim_end <= self._DEFAULT_TRIM_END
        if trim_beg != self._DEFAULT_TRIM_BEG:
            self.trim_beg = trim_beg
        if trim_end != self._DEFAULT_TRIM_END:
            self.trim_end = trim_end

    def trim(self, trim_beg=None, trim_end=None):
        if not trim_beg and not trim_end:
            if self.trim_beg == self._DEFAULT_TRIM_BEG and self.trim_end == self._DEFAULT_TRIM_END:
                print('WARNING: trim values unchanged from default, no trimming will take place')
                return
            else:
                print('Continuing with pre-set trim values.')
                trim_beg = self.trim_beg
                trim_end = self.trim_end

        full_length = len(self[c.CURRENT])
        start = int(full_length * trim_beg)
        stop = int(full_length * trim_end)
        for key, value in self.items():
            self[key] = value[start:stop]

    def save(self, filename, columns=(c.TIME, c.POTENTIAL, c.CURRENT, c.SIGMA)):
        df = pd.DataFrame(data={column: self[column] for column in columns})
        df.to_csv(filename)

    def copy(self):
        e_current = None
        i_current = None
        sigma = None
        error_fl = True
        if c.ELEC_CURRENT in self:
            e_current = self[c.ELEC_CURRENT]
        if c.ION_CURRENT in self:
            i_current = self[c.ION_CURRENT]
        if c.SIGMA in self:
            sigma = self[c.SIGMA]
            error_fl = False
        copied_iv_data = IVData(self[c.POTENTIAL], self[c.CURRENT], self[c.TIME], e_current=e_current, i_current=i_current,
                                trim_beg=self.trim_beg, trim_end=self.trim_end, sigma=sigma, estimate_error_fl=error_fl)
        copied_iv_data.untrimmed_items = self.untrimmed_items
        return copied_iv_data

    def multi_fit(self, sat_region=_DEFAULT_STRAIGHT_CUTOFF, fitter=None, fix_vf_fl=False, plot_fl=False):
        """
        Multi-stage fitting method using an initial straight line fit to the saturation region of the IV curve (decided
        by the sat_region kwarg). The fitted I_sat is then left fixed while T_e and a are found with a 2 parameter fit,
        which gives the guess parameters for an unconstrained full IV fit.
        :param sat_region:  Threshold voltage value below which the 'Straight section' is defined. The straight section
                            is fitted to get an initial value of saturation current for subsequent fits.
        :param fitter:      Fitter object to be used for the fixed-I_sat fit and the final, full-free fit.
        :param plot_fl:     (Boolean) If true, plots the output of all 3 stages of fitting. Default is False.
        :param fix_vf_fl:   (Boolean) If true, fixes the floating potential for all 3 stages of fitting. The value used
                            is the interpolated value of Voltage where Current = 0. Default is False.
        :return:            Full 4-param IVFit data
        """
        import fitters as f
        import matplotlib.pyplot as plt

        if fitter is None:
            fitter = f.FullIVFitter()

        print('Running fit with {}'.format(fitter.name))

        # find floating potential and max potential
        v_f = f.IVFitter.find_floating_pot_iv_data(self)
        v_min = np.min(self[c.POTENTIAL])
        L = v_min - v_f

        # Define lower and upper bounds depending on set trim parameters.
        lower_offset = v_f + (self.trim_beg * L)
        upper_offset = v_f + (self.trim_end * L)
        fit_sec = np.where((self[c.POTENTIAL] <= lower_offset) & (self[c.POTENTIAL] >= upper_offset))
        iv_data_trim = IVData.non_contiguous_trim(self, fit_sec)

        # Find and fit straight section
        str_sec = np.where(iv_data_trim[c.POTENTIAL] <= sat_region)
        iv_data_ss = IVData.non_contiguous_trim(iv_data_trim, str_sec)
        siv_f = f.StraightIVFitter()
        if fix_vf_fl:
            siv_f.set_fixed_values({c.FLOAT_POT: v_f})
        siv_f_data = siv_f.fit_iv_data(iv_data_ss, sigma=iv_data_ss[c.SIGMA])

        # Use I_sat value to fit a fixed_value 4-parameter IV fit
        I_sat_guess = siv_f_data.get_isat().value
        fitter_type = f.FullIVFitter()
        if fix_vf_fl:
            fitter_type.set_fixed_values({c.FLOAT_POT: v_f, c.ION_SAT: I_sat_guess})
        else:
            fitter_type.set_fixed_values({c.ION_SAT: I_sat_guess})
        first_fit_data = fitter_type.fit_iv_data(iv_data_trim, sigma=iv_data_trim[c.SIGMA])

        # Do a full 4 parameter fit with initial guess params taken from previous fit
        params = first_fit_data.fit_params.get_values()
        fitter_type.unset_fixed_values()
        if fix_vf_fl:
            fitter_type.set_fixed_values({c.FLOAT_POT: v_f})
        ff_data = fitter_type.fit_iv_data(iv_data_trim, initial_vals=params, sigma=iv_data_trim[c.SIGMA])

        if plot_fl:
            fig = plt.figure()
            plt.subplot(311)
            siv_f_data.plot(fig=fig, show_fl=False)

            plt.subplot(312)
            first_fit_data.plot(fig=fig, show_fl=False)

            plt.subplot(313)
            ff_data.plot(fig=fig, show_fl=False)
            plt.xlabel('Volt')
            fig.suptitle('lower_offset = {}, upper_offset = {}'.format(self.trim_beg, self.trim_end))

        return ff_data

    @staticmethod
    def percentage_trim(iv_data, trim_beg=0.0, trim_end=1.0):
        if not iv_data or not isinstance(iv_data, IVData):
            raise ValueError('Invalid iv_data given.')
        full_length = len(iv_data[c.CURRENT])
        start = int(full_length * trim_beg)
        stop = int(full_length * trim_end)
        return IVData.positional_trim(iv_data, start, stop)

    @staticmethod
    def positional_trim(iv_data, start, stop):
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

    @staticmethod
    def non_contiguous_trim(iv_data, selection):
        assert isinstance(iv_data, IVData)
        assert isinstance(selection, coll.Iterable)
        new_iv_data = iv_data.copy()
        for label, data in new_iv_data.items():
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            new_iv_data[label] = data[selection]
        return new_iv_data

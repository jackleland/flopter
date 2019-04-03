from flopter.core import constants as c
import numpy as np
import collections as coll
import pandas as pd


class IVData(dict):
    """
    A dictionary which holds IV specific data, namely
        * Voltage
        * Current
        * Time
        * [Electron Current] (for simulations)
        * [Ion Current] (for simulations)

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
        """
        Returns the potential, current (and optionally ion current) arrays from the IVData object

        :return: Tuple of [potential, current, ion_current (if present)]

        """
        if c.ION_CURRENT in self.keys():
            return self[c.POTENTIAL], self[c.CURRENT], self[c.ION_CURRENT]
        else:
            return self[c.POTENTIAL], self[c.CURRENT]

    def set_trim(self, trim_beg=_DEFAULT_TRIM_BEG, trim_end=_DEFAULT_TRIM_END):
        # TODO: Find alternative to this, it is not clear how it should work.
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
        """
        Converts the IV data object to a pandas dataframe and then stores is at the location specified by filename in
        csv format. The columns which are saved to file can be altered with the columns kwarg.

        :param filename:    String (or Path-like) to destination file.
        :param columns:     Which columns from the IVData object to store to file in the form of a tuple of
                            strings. Default is (c.TIME, c.POTENTIAL, c.CURRENT, c.SIGMA)

        """
        df = pd.DataFrame(data={column: self[column] for column in columns})
        df.to_csv(filename)

    def copy(self):
        """
        Deep copy of IVData object, including conditional copying for electron and ion current, set trim values, sigma
        values and the estimation of error flag.

        :return: Deep copy of IVData object

        """
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

    def multi_fit(self, sat_region=_DEFAULT_STRAIGHT_CUTOFF, fitter=None, fix_vf_fl=False, plot_fl=False,
                  print_fl=False):
        """
        Multi-stage fitting method using an initial straight line fit to the saturation region of the IV curve (decided
        by the sat_region kwarg). The fitted I_sat is then left fixed while T_e and a are found with a 2 parameter fit,
        which gives the guess parameters for an unconstrained full IV fit.

        :param sat_region:  (Integer) Threshold voltage value below which the 'Straight section' is defined. The straight section
                            is fitted to get an initial value of saturation current for subsequent fits.
        :param fitter:      Fitter object to be used for the fixed-I_sat fit and the final, full-free fit.
        :param plot_fl:     (Boolean) If true, plots the output of all 3 stages of fitting. Default is False.
        :param fix_vf_fl:   (Boolean) If true, fixes the floating potential for all 3 stages of fitting. The value used
                            is the interpolated value of Voltage where Current = 0. Default is False.
        :param print_fl:    (Boolean) If true, prints fitter information to console.
        :return:            (IVFitData) Full 4-param IVFit data

        """
        import matplotlib.pyplot as plt
        import flopter.core.fitters as f

        if fitter is None or not isinstance(fitter, f.IVFitter):
            fitter = f.FullIVFitter()

        if print_fl:
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
        if fix_vf_fl:
            fitter.set_fixed_values({c.FLOAT_POT: v_f, c.ION_SAT: I_sat_guess})
        else:
            fitter.set_fixed_values({c.ION_SAT: I_sat_guess})
        first_fit_data = fitter.fit_iv_data(iv_data_trim, sigma=iv_data_trim[c.SIGMA])

        # Do a full 4 parameter fit with initial guess params taken from previous fit
        params = first_fit_data.fit_params.get_values()
        fitter.unset_fixed_values()
        if fix_vf_fl:
            fitter.set_fixed_values({c.FLOAT_POT: v_f})
        ff_data = fitter.fit_iv_data(iv_data_trim, initial_vals=params, sigma=iv_data_trim[c.SIGMA])

        if plot_fl:
            fig = plt.figure()
            plt.subplot(311)
            siv_f_data.plot(fig=fig, show_fl=False)
            plt.xlabel('')

            plt.subplot(312)
            first_fit_data.plot(fig=fig, show_fl=False)
            plt.xlabel('')

            plt.subplot(313)
            ff_data.plot(fig=fig, show_fl=False)
            plt.xlabel('Voltage (V)')
            fig.suptitle('lower_offset = {}, upper_offset = {}'.format(self.trim_beg, self.trim_end))
            plt.show()

        return ff_data

    @staticmethod
    def fractional_trim(iv_data, trim_beg=0.0, trim_end=1.0):
        """
        Static method for trimming all the arrays in an IVData object from and to indices calculated from the fraction
        (0.0-1.0) of the arrays total length. e.g. trim_beg=0.2 would have a trim start at the nearest index to 20%
        through the array.

        :param iv_data:     IVData object to trim
        :param trim_beg:    Lower fraction of trim, default is the start of the array.
        :param trim_end:    Upper fraction of trim, default is the end of the array
        :return: Trimmed IVData object

        """
        if not iv_data or not isinstance(iv_data, IVData):
            raise ValueError('Invalid iv_data given.')
        full_length = len(iv_data[c.CURRENT])
        start = int(full_length * trim_beg)
        stop = int(full_length * trim_end)
        return IVData.positional_trim(iv_data, start, stop)

    @staticmethod
    def positional_trim(iv_data, start, stop):
        """
        Function for trimming an IVData's underlying numpy arrays, using start and stop indices.

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
        """
        Static method for trimming a non-contiguous section of the IVData arrays. Non-contiguous area is trimmed using
        an array of indices, usually found with numpy's where() function.

        :param iv_data:     IVData object to be trimmed
        :param selection:   List of indices referring to non-contiguous area to be trimmed.
        :return: trimmed IVData object with all arrays of length (len(selection))

        """
        assert isinstance(iv_data, IVData)
        assert isinstance(selection, coll.Iterable)
        new_iv_data = iv_data.copy()
        for label, data in new_iv_data.items():
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            new_iv_data[label] = data[selection]
        return new_iv_data

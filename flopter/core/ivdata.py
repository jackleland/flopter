from flopter.core import constants as c
import numpy as np
import matplotlib.pyplot as plt
import collections as coll
import pandas as pd
import xarray as xr
from scipy import interpolate


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
    _DEFAULT_PANDAS_COLUMNS = (c.TIME, c.POTENTIAL, c.CURRENT, c.SIGMA)
    _DEFAULT_XARRAY_DIMS = (c.TIME, 'time')
    _DEFAULT_XARRAY_VARIABLES = (
        (c.POTENTIAL, 'voltage'),
        (c.CURRENT, 'current'),
        (c.SIGMA, 'd_current'),
    )

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

    @classmethod
    def from_dataset(cls, ds, sigma=c.XSIGMA):
        """
        Pseudo constructor to create an IVData object from an xarray dataset.
        The data_var to use as the error in the current value can be specified.

        :param ds:      Dataset containing data relating to an IV characteristc
                        to be made into an IVData object
        :param sigma:   Label for the data_var to be used as sigma in the IVData
        :return:        IVData object

        """
        return cls(ds[c.XVOLTAGE].values, ds[c.XCURRENT].values, ds[c.XTIME].values, sigma=ds[sigma].values,
                   estimate_error_fl=False)

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

    def to_dataframe(self, columns=_DEFAULT_PANDAS_COLUMNS):
        """
        Converts the IV data object to a pandas dataframe and returns it.

        :param columns:     Which columns from the IVData object to store to
                            file in the form of a tuple of strings.
                            Default is (c.TIME, c.POTENTIAL, c.CURRENT, c.SIGMA)
        :return: DataFrame of IVData object with specified columns.
        """
        return pd.DataFrame(data={column: self[column] for column in columns})

    def to_dataset(self, dims=_DEFAULT_XARRAY_DIMS, variables=_DEFAULT_XARRAY_VARIABLES, indices=None):
        if indices is None:
            indices = np.where(np.isfinite(self['V']))
        if not isinstance(variables, dict):
            variables = dict(variables)

        dim_new_label = dims[1]
        dim_iv_label = dims[0]

        data_variables_arg = {var_new_label: ([dim_new_label], self[var_iv_label][indices])
                              for var_iv_label, var_new_label in variables.items()}
        coords_kwarg = {dim_new_label: self[dim_iv_label][indices]}
        return xr.Dataset(data_variables_arg, coords=coords_kwarg)

    def save(self, filename, columns=_DEFAULT_PANDAS_COLUMNS):
        """
        Converts the IV data object to a pandas dataframe and then stores is at
        the location specified by filename in csv format. The columns which are
        saved to file can be altered with the columns kwarg.

        :param filename:    String (or Path-like) to destination file.
        :param columns:     Which columns from the IVData object to store to
                            file in the form of a tuple of strings.
                            Default is (c.TIME, c.POTENTIAL, c.CURRENT, c.SIGMA)

        """
        df = self.to_dataframe(columns=columns)
        df.to_csv(filename)

    def get_vf(self):
        iv_interp = interpolate.interp1d(self[c.CURRENT], self[c.POTENTIAL])
        v_float = iv_interp([0.0])
        return v_float

    def get_vf_index(self):
        iv_interp = interpolate.interp1d(self[c.CURRENT], np.arange(len(self[c.CURRENT])))
        return int(iv_interp(0.0).round())

    def get_below_floating(self, v_f=None, print_fl=False):
        if v_f is None:
            import flopter.core.fitters as f
            # find floating potential and max potential
            v_f = f.IVFitter.find_floating_pot_iv_data(self)
        v_min = np.min(self[c.POTENTIAL])
        L = v_min - v_f

        # Define lower and upper bounds depending on set trim parameters.
        lower_offset = v_f + (self.trim_beg * L)
        upper_offset = v_f + (self.trim_end * L)
        fit_sec = np.where((self[c.POTENTIAL] <= lower_offset) & (self[c.POTENTIAL] >= upper_offset))
        if print_fl:
            print('fit_sec is :', fit_sec)

        return IVData.non_contiguous_trim(self, fit_sec)

    def plot(self, ax=None, trim_lines_fl=False, axes_labels=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if c.SIGMA in self.keys():
            plot_func = ax.errorbar
            kwargs['yerr'] = self[c.SIGMA]
        else:
            plot_func = ax.plot

        plot_func(self[c.POTENTIAL], self[c.CURRENT], **kwargs)
        ax.axhline(y=0.0, **c.AX_LINE_DEFAULTS)

        if trim_lines_fl:
            full_length = len(self[c.CURRENT])
            start = int(full_length * self.trim_beg)
            stop = int(full_length * self.trim_end) - 1
            ax.axvline(x=self[c.POTENTIAL][start], label='trim start', **c.AX_LINE_DEFAULTS)
            ax.axvline(x=self[c.POTENTIAL][stop], label='trim stop', **c.AX_LINE_DEFAULTS)

        if axes_labels is True:
            ax.set_xlabel(r'$I_P$ (A)')
            ax.set_ylabel(r'$V_P$ (V)')
        elif axes_labels is not None:
            xlabel, ylabel = axes_labels
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

    def copy(self):
        """
        Deep copy of IVData object, including conditional copying for electron
        and ion current, set trim values, sigma values and the estimation of
        error flag.

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
        copied_iv_data = IVData(self[c.POTENTIAL], self[c.CURRENT], self[c.TIME], e_current=e_current,
                                i_current=i_current, trim_beg=self.trim_beg, trim_end=self.trim_end, sigma=sigma,
                                estimate_error_fl=error_fl)
        copied_iv_data.untrimmed_items = self.untrimmed_items
        return copied_iv_data

    def multi_fit(self, sat_region=_DEFAULT_STRAIGHT_CUTOFF, fitter=None, fix_vf_fl=False, plot_fl=False,
                  print_fl=False, minimise_temp_fl=True):
        """
        Multi-stage fitting method using an initial straight line fit to the
        saturation region of the IV curve (decided by the sat_region kwarg). The
        fitted I_sat is then left fixed while T_e and a are found with a
        2-parameter fit, which gives the guess parameters for an unconstrained
        full IV fit.

        :param sat_region:  (Integer) Threshold voltage value below which the
                            'Straight section' is defined. The straight section
                            is fitted to get an initial value of saturation
                            current for subsequent fits.
        :param fitter:      Fitter object to be used for the fixed-I_sat fit and
                            the final, full, free fit.
        :param plot_fl:     (Boolean) If true, plots the output of all 3 stages
                            of fitting. Default is False.
        :param fix_vf_fl:   (Boolean) If true, fixes the floating potential for
                            all 3 stages of fitting. The value used is the
                            interpolated value of Voltage where Current = 0.
                            Default is False.
        :param print_fl:    (Boolean) If true, prints fitter information to
                            console.
        :param minimise_temp_fl:
                            (Boolean) Flag to control whether multiple final
                            fits are performed at a range of upper indices past
                            the floating potential, with the fit producing the
                            lowest temperature returned.
        :return:            (IVFitData) Full 4-parameter IVFitData object

        """
        import flopter.core.fitters as f

        if fitter is None or not isinstance(fitter, f.IVFitter):
            fitter = f.FullIVFitter()

        if print_fl:
            print('Running fit with {}'.format(fitter.name))

        # find floating potential and max potential
        v_f = f.IVFitter.find_floating_pot_iv_data(self)

        iv_data_trim = self.get_below_floating(v_f=v_f, print_fl=print_fl)

        # Find and fit straight section
        str_sec = np.where(iv_data_trim[c.POTENTIAL] <= sat_region)
        iv_data_ss = IVData.non_contiguous_trim(iv_data_trim, str_sec)
        siv_f = f.StraightIVFitter()
        if fix_vf_fl:
            siv_f.set_fixed_values({c.FLOAT_POT: v_f})
        siv_f_data = siv_f.fit_iv_data(iv_data_ss, sigma=iv_data_ss[c.SIGMA])

        # Use I_sat value to fit a fixed_value 4-parameter IV fit
        I_sat_guess = siv_f_data.get_isat()
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

        # Option to fit to multiple values past the floating potential to minimise the temperature of the fit.
        if minimise_temp_fl:
            ff_data = self.fit_to_min_temperature(initial_vals=params, fitter=fitter, plot_fl=plot_fl)
        else:
            ff_data = fitter.fit_iv_data(iv_data_trim, initial_vals=params, sigma=iv_data_trim[c.SIGMA])

        if plot_fl:
            fig, ax = plt.subplots(3, sharex=True, sharey=True)
            siv_f_data.plot(ax=ax[0])
            ax[0].set_xlabel('')
            ax[0].set_ylabel('Current (A)')

            first_fit_data.plot(ax=ax[1])
            ax[1].set_xlabel('')
            ax[1].set_ylabel('Current (A)')

            ff_data.plot(ax=ax[2])
            ax[2].set_xlabel('Voltage (V)')
            ax[2].set_ylabel('Current (A)')

            fig.suptitle('lower_offset = {}, upper_offset = {}'.format(self.trim_beg, self.trim_end))

        return ff_data

    def gunn_fit(self, sat_region=_DEFAULT_STRAIGHT_CUTOFF, plot_fl=False):
        """
        A version of Jamie Gunn's 4-parameter fit by subtracting the straight
        line of the ion saturation region from the whole IV curve. This is was
        adapted from a jupyter notebook made for Magnum data, and generates four
        plots: averaged IV, straight line overlay, corrected IV, 3-param fit to
        corrected IV.

        :return: A tuple of (corrected_iv_data, corrected_iv_fit_data)

        """
        import matplotlib.pyplot as plt
        import flopter.core.fitters as fts

        iv_data = self

        # define a straight section and trim the iv data to it
        str_sec = np.where(iv_data['V'] <= sat_region)
        iv_data_ss = IVData.non_contiguous_trim(iv_data, str_sec)

        # needed to define the area of the straight section on a graph with a vertical line
        str_sec_end = np.argmax(iv_data['V'][str_sec])

        # fit & plot a straight line to the 'straight section'
        sl_fitter = fts.StraightLineFitter()
        fit_data_ss = sl_fitter.fit(iv_data_ss['V'], iv_data_ss['I'], sigma=iv_data_ss['sigma'])

        # Extrapolate the straight line over a wider voltage range for illustrative purposes
        sl_range = np.linspace(-120, 100, 100)
        sl_function = fit_data_ss.fit_function(sl_range)

        # Subtract the gradient of the straight section from the whole IV curve.
        iv_data_corrected = iv_data.copy()
        iv_data_corrected['I'] = iv_data_corrected['I'] - (fit_data_ss.get_param('m') * iv_data_corrected['V'])

        simple_iv_fitter = fts.SimpleIVFitter()
        fit_data_corrected = iv_data_corrected.multi_fit(sat_region=sat_region, fitter=simple_iv_fitter,
                                                         plot_fl=plot_fl)

        if plot_fl:
            plt.figure()
            plt.errorbar(iv_data['V'], iv_data['I'], yerr=iv_data['sigma'], label='Full IV', color='darkgrey',
                         ecolor='lightgray')
            plt.legend()
            plt.xlabel(r'$V_p$ / V')
            plt.ylabel(r'$I$ / A')
            plt.ylim(-0.5, 1.3)
            plt.xlim(-102, 5)

            plt.figure()
            plt.errorbar(iv_data['V'], iv_data['I'], yerr=iv_data['sigma'], label='Full IV', color='darkgrey',
                         ecolor='lightgray')
            plt.plot(sl_range, sl_function, label='SE Line', color='blue', linewidth=0.5, zorder=10)
            plt.legend()
            plt.xlabel(r'$V_p$ / V')
            plt.ylabel(r'$I$ / A')
            plt.ylim(-0.5, 1.3)
            plt.xlim(-102, 5)

            plt.figure()
            plt.plot(sl_range, sl_function, label='SE Line', color='blue', linewidth=0.5, zorder=10)
            plt.errorbar(iv_data_corrected['V'], iv_data_corrected['I'], label='Corrected IV',
                         yerr=iv_data_corrected[c.SIGMA], color='darkgrey', ecolor='lightgray')
            plt.legend()
            plt.xlabel(r'$V_p$ / V')
            plt.ylabel(r'$I$ / A')
            plt.ylim(-0.5, 1.3)
            plt.xlim(-102, 5)

            plt.figure()
            plt.plot(sl_range, sl_function, label='SE Line', color='blue', linewidth=0.5)
            plt.errorbar(iv_data_corrected['V'], iv_data_corrected['I'], label='Corrected IV',
                         yerr=iv_data_corrected[c.SIGMA], color='darkgrey', ecolor='lightgray')
            plt.plot(*fit_data_corrected.get_fit_plottables(), label='3 Param-Fit', zorder=10, color='r')
            plt.legend()
            plt.xlabel(r'$V_p$ / V')
            plt.ylabel(r'$I$ / A')
            plt.ylim(-0.5, 1.3)
            plt.xlim(-102, 5)

            plt.show()
        return iv_data_corrected, fit_data_corrected

    def fit_to_min_temperature(self, initial_vals=None, fitter=None, plot_fl=False):
        """
        A fitting function which minimises the fitted temperature of the IV
        characteristic by varying how many values around the floating potential
        are included in the fit. The default values for distance around the
        floating potential are 0.2 and 0.1 towards the plasma potential and ion
        saturation region respectively. As of now these are hard coded, but will
        be configurable in future updates.

        :param fitter:          (IVFitter) Fitter object used to perform the
                                fit. The default is FullIVFitter
        :param initial_vals:    (tuple) The initial parameters for the fit.
                                Default's to the fitter's preset starting params
                                if left as None.
        :param plot_fl:         (Boolean) Flag to control whether the process is
                                plotted while running. Current configuration is
                                to plot the full IV, overlaid with each of the
                                trimmed IVs, and below that a separate plot of
                                temperature as a function of upper trim voltage.
        :return:                (IVFitData) FitData object for lowest
                                temperature fit performed.

        """
        import flopter.core.fitters as f

        if fitter is None or not isinstance(fitter, f.IVFitter):
            fitter = f.FullIVFitter()

        if plot_fl:
            fig, ax = plt.subplots(2, sharex=True)
            self.plot(ax=ax[0])

        vf_index = self.get_vf_index()

        # Define how far from the floating potential we want to trim. These values are taken from a test case for a low
        # field shot.
        # TODO: (2020-05-19) Make this configurable
        upper_dist_frac = 0.2  # 10/50
        lower_dist_frac = 0.1  # 5/50
        step_frac = 0.02  # 1/50

        trim_range_updist = int(upper_dist_frac * len(self['t']))
        trim_range_dndist = int(lower_dist_frac * len(self['t']))
        trim_range_step = max(int(step_frac * len(self['t'])), 1)

        # Set trim range depending on what direction our data is in.
        if self['V'][0] < self['V'][-1]:
            trim_range = np.arange(max(vf_index - trim_range_dndist, 0),
                                   min(vf_index + trim_range_updist, len(self['t'])),
                                   trim_range_step)
        else:
            trim_range = np.arange(max(vf_index - trim_range_updist, 0),
                                   min(vf_index + trim_range_dndist, len(self['t'])),
                                   trim_range_step)

        # Fit for each value of trim around the floating potential
        trim_fits = []
        temps = []
        volts = []
        for i in trim_range:
            # Trim is defined as: from the 0th element to the ith element, or from the ith element to the last element
            # depending on the direction of sweep.
            if self['V'][0] < self['V'][-1]:
                trim_iv = self.upper_trim(i)
            else:
                trim_iv = self.lower_trim(i)

            if plot_fl:
                trim_iv.plot(ax=ax[0])

            trim_fit = fitter.fit_iv_data(trim_iv, sigma=trim_iv['sigma'], initial_vals=initial_vals)

            trim_fits.append(trim_fit)
            temps.append(trim_fit.get_temp())

            # The voltages are only extracted for plotting purposes
            volts.append(np.max(trim_fit.raw_x))

        if plot_fl:
            ax[1].plot(volts, temps)

        # Get the index for the minimum value of temperature
        min_temp_index = int(np.argmin(temps, axis=None))

        return trim_fits[min_temp_index]

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
        return IVData.bi_positional_trim(iv_data, start, stop)

    def upper_trim(self, upper_index):
        """
        Function for trimming an IVData's underlying numpy arrays up to a
        maximum upper index.

        :param upper_index:     Index at which to trim to, measured from the
                                beginning of the array
        :return:                trimmed IVData object with arrays of length
                                upper_index - 1

        """
        new_iv_data = self.copy()
        for key, value in self.items():
            new_iv_data[key] = value[:upper_index]
        return new_iv_data

    def lower_trim(self, lower_index):
        """
        Function for trimming an IVData's underlying numpy arrays from some
        index to the end of the array

        :param lower_index:     Index at which to trim from, measured from the
                                beginning of the array. Trims from this value to
                                the end of the array.
        :return:                Trimmed IVData object with arrays of length
                                len(arr) - lower_index

        """
        new_iv_data = self.copy()
        for key, value in self.items():
            new_iv_data[key] = value[lower_index:]
        return new_iv_data

    @staticmethod
    def bi_positional_trim(iv_data, start, stop):
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

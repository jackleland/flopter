import flopter as fl
import fitters as f
import numpy as np
import pathlib as pth
import matplotlib.pyplot as plt
import classes.magnumadcdata as md
import classes.ivdata as iv
import pandas as pd
import scipy.signal as sig
import databases.magnum as mag
import external.readfastadc as adc
from codac.datastore import client
import Ice
import lputils as lp
import normalisation as nrm


class Magopter(fl.IVAnalyser):
    # Default values
    _FOLDER_STRUCTURE = '/Data/Magnum/'
    _TAR_VOLTAGE_CHANNEL = 0
    _SRC_VOLTAGE_CHANNEL = 2
    _VOLTAGE_CHANNEL = 5
    _PROBE_CHANNEL_3 = 6
    _PROBE_CHANNEL_4 = 7
    _COAX_CONVERSION = {
        3: 0,
        4: 1
    }
    _ADC_TIMER_OFFSET = 1.0
    _ARCING_THRESHOLD = 15
    _ACCEPTED_FREQS = np.array([10.0, 20.0, 100.0, 1000.0])

    def __init__(self, directory, filename, ts_filename=None):
        super().__init__()
        # Check for leading/trailing forward slashes?
        self.directory = directory
        self.file = filename
        self.ts_file = ts_filename
        self.full_path = '{}{}{}{}'.format(pth.Path.home(), self._FOLDER_STRUCTURE, directory, filename)

        self.m_data = md.MagnumAdcData(self.full_path, filename)

        self.adc_duration = max(self.m_data.time)

        self.peaks = None
        self.iv_arr_coax_0 = None
        self.iv_arr_coax_1 = None
        self.max_voltage = []
        self.arcs = []
        self.iv_arrs = []
        self.fit_arrs = None
        self.trim_beg = 0.0
        self.trim_end = 1.0

        self.timestamp = int(adc.get_magnumdb_timestamp(filename))
        self.ts_timestamp = None
        self.ts_temp = None
        self.ts_temp_d = None
        self.ts_dens = None
        self.ts_dens_d = None
        self.ts_coords = None
        try:
            self.offline = False
            self.magnum_db = mag.MagnumDB(time_stamp=self.timestamp)
            beam_down = self.magnum_db.get_data(mag.BEAM_DUMP_DOWN)
            self.beam_down_timestamp = [beam_down[mag.TIMES][i] for i in range(len(beam_down[mag.TIMES]))
                                        if beam_down[mag.DATA][i]][0]
            print('Beam Down Timestamp: ', self.beam_down_timestamp, client.timetoposix(self.beam_down_timestamp))
            print('Regular Timestamp: ', self.timestamp, client.timetoposix(self.timestamp))
            self.magnum_data = self.magnum_db.get_data_dict(ref_time=self.timestamp)
            if ts_filename:
                self.ts_timestamp = int(adc.get_magnumdb_timestamp(ts_filename))
                ts_time_range = self.magnum_db.get_approx_time_range(filename=self.ts_file)
                ts_data = self.magnum_db.get_ts_data(time_range=ts_time_range, ref_time=self.ts_timestamp)
                self.ts_temp = ts_data[mag.TS_TEMP_PROF]
                self.ts_temp_d = ts_data[mag.TS_TEMP_PROF_D]
                self.ts_dens = ts_data[mag.TS_DENS_PROF]
                self.ts_dens_d = ts_data[mag.TS_DENS_PROF_D]
                self.ts_coords = ts_data[mag.TS_RAD_COORDS]

            elif set(mag.TS_VARS).issubset(self.magnum_data):
                self.ts_temp = self.magnum_data[mag.TS_TEMP_PROF]
                self.ts_temp_d = self.magnum_data[mag.TS_TEMP_PROF_D]
                self.ts_dens = self.magnum_data[mag.TS_DENS_PROF]
                self.ts_dens_d = self.magnum_data[mag.TS_DENS_PROF_D]
                self.ts_coords = self.magnum_data[mag.TS_RAD_COORDS]

        except Ice.LocalException as e:
            print(str(e))
            print('Database could not be connected to, operating in offline mode.')
            self.offline = True
            self.beam_down_timestamp = None
            self.magnum_db = None
            self.magnum_data = None

    def prepare(self, down_sampling_rate=5, plot_fl=False):
        """
        Preparation consists of downsampling (if necessary), choosing the region of interest and putting each sweep
        into a numpy array of iv_datas
        """
        # This whole function should probably be put into a homogeniser implementation

        # Downsample by factor given
        arr_size = len(self.m_data.data[self.m_data.channels[0]])
        downsample = np.arange(0, arr_size, down_sampling_rate, dtype=np.int64)

        for ch, data in self.m_data.data.items():
            self.m_data.data[ch] = data[downsample]
        self.m_data.time = self.m_data.time[downsample] + self._ADC_TIMER_OFFSET

        self.m_data.data[self._VOLTAGE_CHANNEL] = self.m_data.data[self._VOLTAGE_CHANNEL] * 10

        # Find region of interest
        # plt.figure()
        # plt.plot(self.m_data.time, self.m_data.data[self._TAR_VOLTAGE_CHANNEL])
        # plt.plot(self.m_data.time, self.m_data.data[self._PROBE_CHANNEL_3])
        # plt.plot(self.m_data.time, self.m_data.data[self._VOLTAGE_CHANNEL])

        start = 102000
        end = 115500

        # start = 10000
        # end = 360000
        start = 0
        end = len(self.m_data.time)

        self.iv_arr_coax_0 = []
        self.iv_arr_coax_1 = []

        # Find peaks of voltage triangle wave
        raw_time = self.m_data.time
        raw_voltage = self.m_data.data[self._VOLTAGE_CHANNEL]
        raw_current_0 = self.m_data.data[self._PROBE_CHANNEL_3]
        raw_current_1 = self.m_data.data[self._PROBE_CHANNEL_4]

        # Use fourier decomposition from get_frequency method in triangle fitter to get frequency
        triangle = f.TriangleWaveFitter()
        frequency = triangle.get_frequency(raw_time, raw_voltage, accepted_freqs=self._ACCEPTED_FREQS)

        # Smooth the voltage to get a first read of the peaks on the triangle wave
        smoothed_voltage = sig.savgol_filter(raw_voltage, 21, 2)
        top = sig.argrelmax(smoothed_voltage, order=100)[0]
        bottom = sig.argrelmin(smoothed_voltage, order=100)[0]
        _peaks = raw_time[np.concatenate([top, bottom])]
        _peaks.sort()

        # Get distances betweenn the peaks and filter based on the found frequency
        _peak_distances = np.diff(_peaks)
        threshold = (1 / (2 * frequency)) - 0.001
        _peaks_ind = np.where(_peak_distances > threshold)[0]

        # Starting from the first filtered peak, arrange a period-spaced array
        peaks_refined = np.arange(_peaks[_peaks_ind[0]], raw_time[-1], 1 / (2 * frequency))
        self.peaks = peaks_refined

        if plot_fl:
            plt.figure()
            plt.plot(raw_time, raw_voltage)
            # plt.plot(raw_time, triangle_fit.fit_y)
            for peak in self.peaks:
                plt.axvline(x=peak, linestyle='dashed', linewidth=1, color='r')

        straight_line = f.StraightLineFitter()
        for i in range(len(self.peaks) - 1):
            sweep_start = np.abs(raw_time - self.peaks[i]).argmin()
            sweep_stop = np.abs(raw_time - self.peaks[i+1]).argmin()
            # sweep_stop = self.peaks[i + 1]

            sweep_voltage = raw_voltage[sweep_start:sweep_stop]
            # sweep_fit = triangle_fit.fit_y[sweep_start:sweep_stop]
            sweep_time = raw_time[sweep_start:sweep_stop]
            sweep_fit = straight_line.fit(sweep_time, sweep_voltage)
            if i == 0 and plot_fl:
                sweep_fit.plot()
            self.max_voltage.append((np.max(np.abs(sweep_voltage - sweep_fit.fit_y))))

            if np.max(np.abs(sweep_voltage - sweep_fit.fit_y)) > self._ARCING_THRESHOLD:
                self.arcs.append(np.mean(sweep_time))
                continue

            sweep_current_0 = raw_current_0[sweep_start:sweep_stop]
            sweep_current_1 = raw_current_1[sweep_start:sweep_stop]
            if sweep_voltage[0] > sweep_voltage[-1]:
                sweep_voltage = np.array(list(reversed(sweep_voltage)))
                sweep_time = np.array(list(reversed(sweep_time)))
                sweep_current_0 = np.array(list(reversed(sweep_current_0)))
                sweep_current_1 = np.array(list(reversed(sweep_current_1)))
            self.iv_arr_coax_0.append(iv.IVData(np.array(sweep_voltage) - np.array(sweep_current_0),
                                                sweep_current_0, sweep_time))
            self.iv_arr_coax_1.append(iv.IVData(np.array(sweep_voltage) - np.array(sweep_current_1),
                                                sweep_current_1, sweep_time))

        self.iv_arrs = [
            self.iv_arr_coax_0,
            self.iv_arr_coax_1
        ]

    def trim(self, trim_beg=0.0, trim_end=1.0):
        self.trim_beg = trim_beg
        self.trim_end = trim_end
        for iv_arr in self.iv_arrs:
            for _iv_data in iv_arr:
                _iv_data.trim_beg = trim_beg
                _iv_data.trim_end = trim_end

    def denormalise(self):
        pass

    def fit(self, fitter=None, coaxes=(0, 1), initial_vals=None, bounds=None, print_fl=False):
        if not fitter:
            fitter = f.FullIVFitter()
        if all(iv_arr is None or len(iv_arr) == 0 for iv_arr in self.iv_arrs):
            raise ValueError('No iv_data found to fit in self.iv_arrs')

        fit_arrs = [[], []]
        fit_time = [[], []]
        for i in coaxes:
            for iv_data in self.iv_arrs[i]:
                fitter.autoset_floating_pot(iv_data)
                trim_fl = (self.trim_beg is not None and self.trim_end is not None)
                try:
                    fit_data = fitter.fit_iv_data(iv_data, initial_vals=initial_vals, bounds=bounds, trim_fl=trim_fl)
                except RuntimeError:
                    if print_fl:
                        print('Error encountered in fit, skipping timestep {}'.format(np.mean(iv_data.time)))
                    continue
                if any(param.error >= (param.value * 0.5) for param in fit_data.fit_params):
                    if print_fl:
                        print('Fit parameters exceeded good fit threshold, skipping time step {}'
                              .format(np.mean(iv_data.time)))
                    continue
                fit_arrs[i].append(fit_data)
                fit_time[i].append(np.mean(iv_data.time))
        fit_dfs_0 = pd.DataFrame([fit_data.to_dict() for fit_data in fit_arrs[0]], index=fit_time[0])
        fit_dfs_1 = pd.DataFrame([fit_data.to_dict() for fit_data in fit_arrs[1]], index=fit_time[1])
        return fit_dfs_0, fit_dfs_1

    def plot_thomson(self, fig=None, show_fl=False):
        if self.ts_temp is not None:
            if not fig:
                fig = plt.figure()

            plt.subplot(211)
            for i in range(len(self.ts_dens[0])):
                plt.errorbar(self.ts_coords[mag.DATA][i], self.ts_dens[mag.DATA][i], fmt='x-',
                             label='t = {:.1f}'.format(self.ts_dens[mag.TIMES][i]),
                             yerr=self.ts_dens_d[mag.DATA][i])
            plt.xlabel('Radial position (mm)')
            plt.ylabel(r'Density (m$^{-3}$)')
            plt.legend()

            plt.subplot(212)
            for i in range(len(self.ts_temp[0])):
                plt.errorbar(self.ts_coords[mag.DATA][i], self.ts_temp[mag.DATA][i] / nrm.ELEM_CHARGE, fmt='x-',
                             label='t = {:.1f}'.format(self.ts_temp[mag.TIMES][i]),
                             yerr=self.ts_temp_d[mag.DATA][i] / nrm.ELEM_CHARGE)
            plt.xlabel('Radial position (mm)')
            plt.ylabel(r'Temperature (eV)')
            plt.legend()

            if show_fl:
                plt.show()
        else:
            print('No thomson data found, cannot plot.')

    @classmethod
    def get_data_path(cls):
        return "{}{}".format(pth.Path.home(), cls._FOLDER_STRUCTURE)

    @classmethod
    def create_from_file(cls, filename):
        super().create_from_file(filename)


class MagnumProbes(object):
    def __init__(self):
        L_small = 3e-3  # m
        a_small = 2e-3  # m
        b_small = 3e-3  # m
        g_small = 2e-3  # m
        theta_f_small = np.radians(72)

        L_large = 5e-3  # m
        a_large = 4.5e-3  # m
        b_large = 6e-3  # m
        g_large = 1e-3  # m
        theta_f_large = np.radians(73.3)

        L_reg = 5e-3  # m
        a_reg = 2e-3  # m
        b_reg = 3.34e-3  # m
        g_reg = 1e-3  # m
        theta_f_reg = np.radians(75)

        L_cyl = 4e-3  # m
        g_cyl = 5e-4  # m

        d_perp = 3e-4  # m
        theta_p = np.radians(10)

        self.probe_s = lp.AngledTipProbe(a_small, b_small, L_small, g_small, d_perp, theta_f_small, theta_p)
        self.probe_l = lp.AngledTipProbe(a_large, b_large, L_large, g_large, d_perp, theta_f_large, theta_p)
        self.probe_r = lp.AngledTipProbe(a_reg, b_reg, L_reg, g_reg, d_perp, theta_f_reg, theta_p)
        self.probe_c = lp.FlushCylindricalProbe(L_cyl / 2, g_cyl, d_perp)
        self.probes = {
            's': self.probe_s,
            'r': self.probe_r,
            'l': self.probe_l,
            'c': self.probe_c,
        }
        self.position = ['s', 'r', 'l', 'c']

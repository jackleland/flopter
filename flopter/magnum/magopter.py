import flopter.core.constants
from flopter.core.ivanalyser import IVAnalyser
import numpy as np
import pathlib as pth
import matplotlib.pyplot as plt
import flopter.magnum.adcdata as md
import flopter.core.ivdata as iv
import pandas as pd
import scipy.signal as sig
import flopter.magnum.database as mag
import flopter.magnum.readfastadc as adc
from codac.datastore import client
import Ice
from flopter.core import filtering as filt, constants as c, normalise as nrm, fitters as f
import glob
import os


class Magopter(IVAnalyser):
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
    _ACCEPTED_FREQS = np.array([10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 1.0e4, 2.0e4])
    _SHUNT_RESISTANCE = 1
    _CABLING_RESISTANCE = 0
    _FIT_FILE_STRING = 'fit{}_{}.csv'
    _FIT_FILE_GLOBSTR = '*fit*.csv'

    def __init__(self, directory, filename, ts_filename=None, coaxes=2, combine_sweeps_fl=True, shunt_resistor=None,
                 cabling_resistance=None, offline_fl=False):
        super().__init__()
        # Check for leading/trailing forward slashes?
        self.directory = directory
        self.file = filename
        self.ts_file = ts_filename
        self.full_path = '{}{}{}{}'.format(pth.Path.home(), self._FOLDER_STRUCTURE, directory, filename)
        self.coaxes = coaxes
        self.combine_sweeps_fl = combine_sweeps_fl

        self.shunt_resistance = self._SHUNT_RESISTANCE
        self.cabling_resistance = self._CABLING_RESISTANCE
        if shunt_resistor is not None:
            self.shunt_resistance = shunt_resistor
        if cabling_resistance is not None:
            self.cabling_resistance = cabling_resistance

        self.m_data = md.MagnumAdcData(self.full_path, filename)
        self.adc_duration = max(self.m_data.time)

        self.raw_time = None
        self.raw_voltage = None
        # self.raw_current = [[] for i in range(self.coaxes)]
        self.raw_current = []

        self.time = None
        # self.voltage = [[] for i in range(self.coaxes)]
        # self.current = [[] for i in range(self.coaxes)]
        self.voltage = []
        self.current = []

        self.peaks = None
        self.max_voltage = []
        self.arcs = []
        self.iv_arrs = [[] for i in range(self.coaxes)]
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

    def prepare(self, down_sampling_rate=5, plot_fl=False, filter_arcs_fl=False, roi_b_plasma=False, crit_freq=640,
                crit_ampl=1.1e-3):
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

        self.m_data.data[self._VOLTAGE_CHANNEL] = self.m_data.data[self._VOLTAGE_CHANNEL] * 100

        # Find region of interest
        if roi_b_plasma and not self.offline and np.shape(self.magnum_data[mag.PLASMA_STATE])[1] == 2:
            start = np.abs(self.m_data.time - self.magnum_data[mag.PLASMA_STATE][0][0]).argmin()
            end = np.abs(self.m_data.time - self.magnum_data[mag.PLASMA_STATE][0][1]).argmin()
        else:
            start = 0
            end = len(self.m_data.time)

        # Read in raw values from adc file - these are the time and the voltages measured on each channel
        self.raw_time = np.array(self.m_data.time[start:end])
        self.raw_voltage = np.array(self.m_data.data[self._VOLTAGE_CHANNEL][start:end])
        for i, probe_index in enumerate([self._PROBE_CHANNEL_3, self._PROBE_CHANNEL_4]):
            self.raw_current.append(np.array(self.m_data.data[probe_index][start:end]))

        # Convert the adc voltages into the measured values
        for i in range(self.coaxes):
            # Current is ohmicly calculated from the voltage across a shunt resistor
            self.current.append(self.raw_current[i] / self.shunt_resistance)

            # Separate volages are applied to each probe depending on the current they draw
            self.voltage.append(self.raw_voltage - self.raw_current[i] - (self.cabling_resistance * self.current[i]))

        # self.current = np.array(self.current)

        self.filter(crit_ampl=crit_ampl, crit_freq=crit_freq, plot_fl=plot_fl)

        for i in range(self.coaxes):
            # Use fourier decomposition from get_frequency method in triangle fitter to get frequency
            triangle = f.TriangleWaveFitter()
            frequency = triangle.get_frequency(self.raw_time, self.voltage[i], accepted_freqs=self._ACCEPTED_FREQS)

            # Smooth the voltage to get a first read of the peaks on the triangle wave
            smoothed_voltage = sig.savgol_filter(self.voltage[i], 21, 2)
            top = sig.argrelmax(smoothed_voltage, order=100)[0]
            bottom = sig.argrelmin(smoothed_voltage, order=100)[0]
            _peaks = self.raw_time[np.concatenate([top, bottom])]
            _peaks.sort()

            # Get distances between the peaks and filter based on the found frequency
            _peak_distances = np.diff(_peaks)
            threshold = (1 / (2 * frequency)) - 0.001
            _peaks_ind = np.where(_peak_distances > threshold)[0]

            # Starting from the first filtered peak, arrange a period-spaced array
            peaks_refined = np.arange(_peaks[_peaks_ind[0]], self.raw_time[-1], 1 / (2 * frequency))
            self.peaks = peaks_refined

            if plot_fl:
                plt.figure()
                plt.plot(self.raw_time, self.voltage[i])
                plt.plot(self.raw_time, triangle.fit(self.raw_time, self.voltage[i]).fit_y)
                for peak in self.peaks:
                    plt.axvline(x=peak, linestyle='dashed', linewidth=1, color='r')

            if self.combine_sweeps_fl:
                skip = 2
                sweep_fitter = triangle
            else:
                skip = 1
                sweep_fitter = f.StraightLineFitter()

            # print('peaks_len = {}'.format(len(self.peaks) - skip))
            for j in range(len(self.peaks) - skip):
                sweep_start = np.abs(self.raw_time - self.peaks[j]).argmin()
                sweep_stop = np.abs(self.raw_time - self.peaks[j + skip]).argmin()

                sweep_voltage = self.voltage[i][sweep_start:sweep_stop]
                sweep_time = self.raw_time[sweep_start:sweep_stop]

                if filter_arcs_fl:
                    sweep_fit = sweep_fitter.fit(sweep_time, sweep_voltage)
                    self.max_voltage.append((np.max(np.abs(sweep_voltage - sweep_fit.fit_y))))
                    if i == 0 and plot_fl:
                        sweep_fit.plot()
                    if np.max(np.abs(sweep_voltage - sweep_fit.fit_y)) > self._ARCING_THRESHOLD:
                        self.arcs.append(np.mean(sweep_time))
                        continue

                # sweep_current = [current[sweep_start:sweep_stop] for current in self.current]
                sweep_current = self.current[i][sweep_start:sweep_stop]

                # Reverse alternate sweeps if not operating in combined sweeps mode, so
                if not self.combine_sweeps_fl and sweep_voltage[0] > sweep_voltage[-1]:
                    # sweep_voltage = np.array(list(reversed(sweep_voltage)))
                    # sweep_time = np.array(list(reversed(sweep_time)))
                    # sweep_current = np.array(list(reversed(sweep_current)))
                    sweep_time = np.flip(sweep_time)
                    sweep_voltage = np.flip(sweep_voltage)
                    sweep_current = np.flip(sweep_current)

                # Create IVData objects for each sweep (or sweep pair)
                # TODO: What's the std_err_scaler doing here? Look through commits
                self.iv_arrs[i].append(iv.IVData(sweep_voltage, sweep_current, sweep_time, std_err_scaler=0.95))
                # for j in range(self.coaxes):
                #     self.iv_arrs[j].append(iv.IVData(sweep_voltage - sweep_current[j], sweep_current[j], sweep_time,
                #                                      std_err_scaler=0.95))

    def trim(self, trim_beg=0.0, trim_end=1.0):
        self.trim_beg = trim_beg
        self.trim_end = trim_end
        for iv_arr in self.iv_arrs:
            for _iv_data in iv_arr:
                _iv_data.trim_beg = trim_beg
                _iv_data.trim_end = trim_end

    def denormalise(self):
        pass

    def fit(self, fitter=None, initial_vals=None, bounds=None, load_fl=False, save_fl=False, print_fl=False):
        if load_fl and save_fl:
            print('WARNING: Unnecessary to save and load at the same time - loading will be prioritised if successful.')

        # Looks for csv files containing previously fitted data if asked for by the load_fl boolean flag.
        fit_files = [self._FIT_FILE_STRING.format(i, self.timestamp) for i in range(self.coaxes)]
        if load_fl:
            start_dir = os.getcwd()
            os.chdir('{}{}{}'.format(pth.Path.home(), self._FOLDER_STRUCTURE, self.directory))
            directory_fit_files = glob.glob(self._FIT_FILE_GLOBSTR)
            if set(fit_files).issubset(directory_fit_files):
                return [pd.read_csv(filepath_or_buffer=ff) for ff in fit_files]
            else:
                print('Could not find fit files - they should be in {} with names of the format {}'
                      .format(self.directory, fit_files[0]))
                print('Continuing with regular fit...')
            os.chdir(start_dir)

        # Fitting routine
        if not fitter:
            fitter = f.FullIVFitter()
        if all(iv_arr is None or len(iv_arr) == 0 for iv_arr in self.iv_arrs):
            raise ValueError('No iv_data found to fit in self.iv_arrs')
        # pool = mp.Pool()
        fit_arrs = [[] for dummy in range(self.coaxes)]
        fit_time = [[] for dummy in range(self.coaxes)]
        for i in range(self.coaxes):
            for iv_data in self.iv_arrs[i]:
                try:
                    # Parallelised using multiprocessing.pool
                    # TODO: Not currently working according to system monitor.
                    fit_data = iv_data.multi_fit(plot_fl=False)
                    # result = pool.apply_async(iv_data.multi_fit)
                    # fit_data = result.get(timeout=10)
                except RuntimeError:
                    if print_fl:
                        print('Error encountered in fit, skipping timestep {}'.format(np.mean(iv_data.time)))
                    continue
                if all(param.error >= (param.value * 0.5) for param in fit_data.fit_params):
                    if print_fl:
                        print('All fit parameters exceeded good fit threshold, skipping time step {}'
                              .format(np.mean(iv_data[c.TIME])))
                    continue
                fit_arrs[i].append(fit_data)
                fit_time[i].append(np.mean(iv_data[c.TIME]))
        fit_dfs = [pd.DataFrame([fit_data.to_dict() for fit_data in fit_arrs[i]], index=fit_time[i]) for i in range(self.coaxes)]
        if save_fl:
            for i in range(self.coaxes):
                fit_dfs[i].to_csv(path_or_buf='{}{}{}{}'.format(pth.Path.home(), self._FOLDER_STRUCTURE,
                                                                self.directory, fit_files[i]))
        return fit_dfs

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
                plt.errorbar(self.ts_coords[mag.DATA][i], self.ts_temp[mag.DATA][i] / flopter.core.constants.ELEM_CHARGE, fmt='x-',
                             label='t = {:.1f}'.format(self.ts_temp[mag.TIMES][i]),
                             yerr=self.ts_temp_d[mag.DATA][i] / flopter.core.constants.ELEM_CHARGE)
            plt.xlabel('Radial position (mm)')
            plt.ylabel(r'Temperature (eV)')
            plt.legend()

            if show_fl:
                plt.show()
        else:
            print('No thomson data found, cannot plot.')

    def quick_plot(self, index=None, coax=0, fig=None, show_fl=True):
        # TODO: (06/08/2018) Make plottingmethod into a decorator.
        if not fig:
            fig = plt.figure()

        if not index:
            index = int(0.5 * len(self.iv_arrs[coax]))
        extracted_range = slice(index, index + 3)

        # Plot the first time and current value for each iv_data in iv_arrs[coax]
        plt.plot(*zip(*[[iv_data[c.TIME][0], iv_data[c.CURRENT][0]] for iv_data in self.iv_arrs[coax]]))
        for i, iv_data in enumerate(self.iv_arrs[coax][extracted_range]):
            plt.axvline(x=iv_data[c.TIME][0], linestyle='dashed', color='g', label='iv_data - {}'.format(i))
        plt.legend()

        if show_fl:
            plt.show()

    def filter(self, crit_freq=None, plot_fl=None, crit_ampl=None):
        """
        The filtering applied during preparation has it's own function to tidy
        up the prepare() method. Takes in the same arguments passed to prepare
        and modifies the member variables raw_current and raw_voltage.

        :param crit_freq:   The critical frequency of the desired butterworth
                            filter.
        :param crit_ampl:   The critical amplitude of the desired gate filter.
        :param plot_fl:     Boolean flag for conrolling plotting functionality

        Note: if crit_freq and crit_ampl are both none and plotting is on, the
        function will exit. <- this may be changed in an upcoming release.

        """

        # Filter out high frequency noise (if present) with a butterworth filter set to a crit_freq defined by the user
        if crit_freq:
            low_pass = filt.LowPassFilter(crit_freq)
            nyq = 1 / (2 * (self.raw_time[1] - self.raw_time[0]))
            crit_freq_norm = crit_freq / nyq

            if plot_fl:
                plt.figure()
                plt.plot(self.raw_time, self.raw_current[0], color='silver', label='Raw')
                for cf in [0.5, 1, 2]:
                    bb, aa = sig.butter(6, cf * crit_freq_norm, analog=False)
                    ff_current = sig.filtfilt(bb, aa, self.raw_current[0])
                    plt.plot(self.raw_time, ff_current,
                             label='crit_freq = {:.0f}'.format(cf * crit_freq_norm * nyq))
                plt.legend()
                plt.xlim(12, 12.5)
                plt.ylim(-2.25, 1.35)
                plt.xlabel('Time (s)')
                plt.ylabel('Current (A)')

            # Apply filter to raw_current signals.
            for i in range(self.coaxes):
                self.voltage[i] = low_pass.apply(self.raw_time, self.voltage[i])

            if plot_fl:
                fig = plt.figure()
                low_pass.plot(self.raw_time, self.raw_voltage[0], apply_plot_fl=False,
                              range=[[12.22, 12.25], [-100, 11]],
                              fig=plt.subplot(211), show_fl=False)
                low_pass.plot(self.raw_time, self.raw_current[0], apply_plot_fl=False,
                              range=[[12.22, 12.25], [-2.25, 1.35]],
                              fig=plt.subplot(212))
            # self.current[0] = low_pass.apply(self.raw_time, self.raw_current[0])
            # self.current[1] = low_pass.apply(self.raw_time, self.raw_current[1])
            # self.current = np.array(self.current)
            # self.current = np.array(self.raw_current)

        if crit_ampl:
            gate = filt.GatedFilter(crit_ampl)

            # if crit_freq:
            #     gated_voltage_0 = gate.apply(self.raw_time, self.voltage[0], plot_fl=plot_fl)
            #     gated_voltage_1 = gate.apply(self.raw_time, self.voltage[1], plot_fl=plot_fl)
            #     gated_current_0 = gate.apply(self.raw_time, self.current[0])
            #     gated_current_1 = gate.apply(self.raw_time, self.current[1])
            # else:
            #     gated_voltage_0 = gate.apply(self.raw_time, self.raw_voltage[0], plot_fl=plot_fl)
            #     gated_voltage_1 = gate.apply(self.raw_time, self.raw_voltage[1], plot_fl=plot_fl)
            #     gated_current_0 = gate.apply(self.raw_time, self.raw_current[0])
            #     gated_current_1 = gate.apply(self.raw_time, self.raw_current[1])
            #
            # self.voltage[0] = gated_voltage_0.astype(np.float64)
            # self.voltage[1] = gated_voltage_1.astype(np.float64)
            # self.current = np.array([gated_current_0, gated_current_1], dtype=np.float64)
            for i in range(self.coaxes):
                self.voltage[i] = gate.apply(self.raw_time, self.voltage[i], plot_fl=plot_fl).astype(np.float64)
                self.current[i] = gate.apply(self.raw_time, self.current[i]).astype(np.float64)

        if not crit_freq and not crit_ampl and plot_fl:
            # self.voltage = self.raw_voltage
            # self.current = np.array(self.raw_current)

            sp_I = np.fft.fft(self.raw_current[0])
            sp_V = np.fft.fft(self.raw_voltage)
            sp_I_norm = np.abs(sp_I) / np.max(np.abs(sp_I))
            sp_V_norm = np.abs(sp_V) / np.max(np.abs(sp_V))
            freq = np.fft.fftfreq(len(self.raw_current[0]), self.raw_time[1] - self.raw_time[0])
            nyq = 1 / (2 * (self.raw_time[1] - self.raw_time[0]))

            # plot the two fourier transforms overlaid
            plt.figure()
            plt.semilogy(freq, np.abs(sp_I), '--', label='FFT Spectrum - I')
            plt.semilogy(freq, np.abs(sp_V), '--', label='FFT Spectrum - V')
            plt.ylabel('Amplitude')
            plt.xlabel('Frequency (Hz)')
            plt.grid(which='both', axis='both')
            plt.xlim(0, 5000)
            # plt.xlim(0, 50000)
            plt.ylim(1e-1, 1e8)
            # plt.ylim(1e-6, 1)
            plt.legend()

            # plot the two separately with peaks found.
            peaks_I = sig.argrelmax(sp_I_norm, order=120)
            peaks_V = sig.argrelmax(sp_V_norm, order=200)

            plt.figure()
            plt.subplot(211)
            plt.semilogy(freq, sp_I_norm, 'x', label='FFT Spectrum - I')
            # plt.semilogy(freq[peaks_I], sp_I_norm[peaks_I], 'x', label='FFT Peaks - I')
            plt.ylabel('Amplitude')
            plt.grid(which='both', axis='both')
            plt.xlim(0, 5000)
            # plt.xlim(0, 50000)
            plt.ylim(1e-6, 1)
            plt.legend()

            plt.subplot(212)
            plt.semilogy(freq, sp_V_norm, 'x', label='FFT Spectrum - V')
            # plt.semilogy(freq[peaks_V], sp_V_norm[peaks_V], 'x', label='FFT Peaks - V')
            plt.ylabel('Amplitude')
            plt.xlabel('Frequency (Hz)')
            plt.grid(which='both', axis='both')
            plt.xlim(0, 5000)
            # plt.xlim(0, 50000)
            plt.ylim(1e-6, 1)
            plt.legend()

            # Plot spectrogram of voltage trace
            plt.figure()
            sample_freq = 1 / (self.raw_time[1] - self.raw_time[0])
            print(sample_freq)
            ff, tt, Sxx = sig.spectrogram(self.raw_voltage, sample_freq)
            Sxx_log = np.log(Sxx)
            plt.pcolormesh(tt, ff, Sxx_log)
            cbar = plt.colorbar()
            cbar.set_label('Log(Amplitude)')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')

            filtered_voltage = filt.LowPassFilter(4000).apply(self.raw_time, self.raw_voltage, plot_fl=True)
            plt.figure()
            plt.plot(self.raw_time, self.raw_voltage, label='Raw Voltage')
            plt.plot(self.raw_time, filtered_voltage, label='Filtered Voltage')
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.xlim(8, 8.4)

            plt.show()
            exit()

    @classmethod
    def get_data_path(cls):
        return "{}{}".format(pth.Path.home(), cls._FOLDER_STRUCTURE)

    @classmethod
    def create_from_file(cls, filename):
        super().create_from_file(filename)



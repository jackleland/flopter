import flopter as fl
import fitters as f
import numpy as np
import pathlib as pth
import matplotlib.pyplot as plt
import classes.magnumadcdata as md
import classes.ivdata as iv
import glob
import os
import pandas as pd
import scipy.signal as sig
import scipy.interpolate as itp
import constants as c
import normalisation as nrm
import lputils as lp
import databases.magnum as mag
import external.readfastadc as adc
import external.magnumdbutils as ut
import Ice

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
    _ARCING_THRESHOLD = 15
    _ACCEPTED_FREQS = np.array([10.0, 20.0, 100.0, 1000.0])

    def __init__(self, directory, filename, ts_filename=None):
        super().__init__()
        # Check for leading/trailing forward slashes?
        self.directory = directory
        self.file = filename
        self.ts_file = ts_filename
        self.full_path = ''.join([pth.Path.home(), self._FOLDER_STRUCTURE, directory, filename])

        self.m_data = md.MagnumAdcData(self.full_path, filename)

        self.adc_duration = max(self.m_data.time)
        print(self.adc_duration)

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
        self.ts_temp = None
        self.ts_temp_d = None
        self.ts_dens = None
        self.ts_dens_d = None
        self.ts_coords = None
        try:
            self.offline = False
            self.magnum_db = mag.MagnumDB(time_stamp=self.timestamp)
            self.magnum_data = self.magnum_db.get_data_dict(ref_time=self.timestamp)
            if ts_filename:
                ts_time_range = self.magnum_db.get_approx_time_range(filename=self.ts_file)
                ts_data = self.magnum_db.get_ts_data(time_range=ts_time_range)
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

        except Ice.ConnectionRefusedException__str__:
            print('Database could not be connected to, operating in offline mode.')
            self.offline = True
            self.magnum_db = None
            self.magnum_data = None

    def prepare(self, down_sampling_rate=5):
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
        self.m_data.time = self.m_data.time[downsample]

        self.m_data.data[self._VOLTAGE_CHANNEL] = self.m_data.data[self._VOLTAGE_CHANNEL] * 10

        # Find region of interest
        # plt.figure()
        # plt.plot(self.m_data.time, self.m_data.data[self._TAR_VOLTAGE_CHANNEL])
        # plt.plot(self.m_data.time, self.m_data.data[self._PROBE_CHANNEL_3])
        # plt.plot(self.m_data.time, self.m_data.data[self._VOLTAGE_CHANNEL])

        start = 102000
        end = 115500

        start = 10000
        # start = 0
        end = 360000

        segmentation = np.linspace(start, end, 2, dtype=np.int64)
        print(len(segmentation))
        self.iv_arr_coax_0 = []
        self.iv_arr_coax_1 = []

        for j in range(len(segmentation) - 1):
            seg_start = segmentation[j]
            seg_end = segmentation[j+1]

            # Fit and find peaks of voltage triangle wave
            raw_time = self.m_data.time[seg_start:seg_end]
            raw_voltage = self.m_data.data[self._VOLTAGE_CHANNEL][seg_start:seg_end]
            raw_current_0 = self.m_data.data[self._PROBE_CHANNEL_3][seg_start:seg_end]
            raw_current_1 = self.m_data.data[self._PROBE_CHANNEL_4][seg_start:seg_end]

            triangle = f.TriangleWaveFitter()
            frequency = triangle.get_frequency(raw_time, raw_voltage, accepted_freqs=self._ACCEPTED_FREQS)
            triangle_fit = triangle.fit(raw_time, raw_voltage, freq=frequency)
            triangle_fit.print_fit_params()

            # smoothed_voltage = sig.savgol_filter(raw_voltage, 21, 2)
            top = sig.argrelmax(raw_voltage, order=20)[0]
            bottom = sig.argrelmin(raw_voltage, order=20)[0]
            # top = sig.argrelmax(triangle_fit.fit_y, order=20)[0]
            # bottom = sig.argrelmin(triangle_fit.fit_y, order=20)[0]

            print(raw_time[np.min([bottom[0], top[0]])], raw_time[-1], 1/(2 * frequency))
            self.peaks = np.arange(raw_time[top[0]], raw_time[-1], 1/(2 * frequency))
            # self.peaks = raw_time[np.sort(np.concatenate([top, bottom], 0))]

            plt.figure()
            plt.plot(raw_time, raw_voltage)
            plt.plot(raw_time, triangle_fit.fit_y)
            for peak in self.peaks:
                plt.axvline(x=peak, linestyle='dashed', linewidth=1, color='r')
            # plt.show()

            for i in range(len(self.peaks) - 1):
                sweep_start = np.abs(raw_time - self.peaks[i]).argmin()
                sweep_stop = np.abs(raw_time - self.peaks[i+1]).argmin()
                # sweep_stop = self.peaks[i + 1]

                sweep_voltage = raw_voltage[sweep_start:sweep_stop]
                # sweep_fit = triangle_fit.fit_y[sweep_start:sweep_stop]
                # self.max_voltage.append((np.max(np.abs(sweep_voltage - sweep_fit))))
                sweep_time = raw_time[sweep_start:sweep_stop]

                # if np.max(np.abs(sweep_voltage - sweep_fit)) > self._ARCING_THRESHOLD:
                #     self.arcs.append(np.mean(sweep_time))
                #     continue

                sweep_current_0 = raw_current_0[sweep_start:sweep_stop]
                sweep_current_1 = raw_current_1[sweep_start:sweep_stop]
                if sweep_voltage[0] > sweep_voltage[-1]:
                    sweep_voltage = list(reversed(sweep_voltage))
                    sweep_time = list(reversed(sweep_time))
                    sweep_current_0 = list(reversed(sweep_current_0))
                    sweep_current_1 = list(reversed(sweep_current_1))
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
        # fit_arrs = np.zeros([np.shape(self.iv_arrs)[1], 2])
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

    @classmethod
    def get_data_path(cls):
        return "{}{}".format(pth.Path.home(), cls._FOLDER_STRUCTURE)

    @classmethod
    def create_from_file(cls, filename):
        super().create_from_file(filename)


if __name__ == '__main__':
    folders = ['2018-05-01_Leland/', '2018-05-02_Leland/', '2018-05-03_Leland/',
               '2018-06-05_Leland/', '2018-06-06_Leland/', '2018-06-07_Leland/']
    files = []
    file_folders = []
    for folder1 in folders:
        os.chdir(Magopter.get_data_path() + folder1)
        files.extend(glob.glob('*.adc'))
        file_folders.extend([folder1] * len(glob.glob('*.adc')))

    # files = [f.replace(' ', '_') for f in files]
    files.sort()

    # file = '2018-05-01_12h_55m_47s_TT_06550564404491814477.adc'  # 8
    # file = '2018-05-03_11h_31m_41s_TT_06551284859908422561.adc'  # 82

    files_of_interest = {
        8: "First analysed",
        82: "Higher Temp",
        97: "Angular Sweep with different probes"
    }
    file_index = 82
    # file = files[file_index]
    file = files[-2]
    ts_file = files[-1]
    folder = file_folders[-2]
    print(folder, file)
    print(ut.human_time_str(adc.get_magnumdb_timestamp(ts_file)))
    print(ts_file)

    magopter = Magopter(folder, file)
    # print(file, magopter.magnum_db.get_time_range(filename=file))
    # plt.figure()
    # plt.errorbar(magopter.ts_coords, magopter.ts_temp, yerr=magopter.ts_temp_d, label='Temperature')

    # exit()
    # length = len(magopter.t_file)
    # for i in range(1, 20):
    #     split = int(length / i)
    # plt.figure()
    # plt.title('i = {}'.format(i))
    # plt.log
    # for j in range(i):
    #     plt.semilogy(magopter.t_file[j*split:j+1*split], label='j = {}'.format(j))

    # plt.show()
    dsr = 10
    magopter.prepare(down_sampling_rate=dsr)
    # magopter.trim(trim_end=0.82)
    magopter.trim(trim_end=0.83)
    # exit()
    fit_df_0, fit_df_1 = magopter.fit()

    # plt.figure()
    # # # plt.plot(magopter.m_data.time, triangle_wave)
    # plt.plot(magopter.m_data.time, magopter.m_data.data[5])
    # plt.plot(*triangle_fit.get_fit_plottables())
    # for peak in peaks:
    #     plt.axvline(x=magopter.m_data.time[peak], linestyle='dashed', linewidth=1, color='r')

    iv_data = fit_df_0.iloc[[125]]
    plt.figure()
    for iv_curve in magopter.iv_arr_coax_0:
        plt.plot(iv_curve.time, iv_curve.current)
    plt.axvline(x=iv_data.index)


    # Flush probe measurements
    L_small = 3e-3  # m
    a_small = 2e-3  # m
    b_small = 3e-3  # m
    g_small = 2e-3  # m
    theta_f_small = np.radians(72)

    L_large = 5e-3      # m
    a_large = 4.5e-3    # m
    b_large = 6e-3      # m
    g_large = 1e-3      # m
    theta_f_large = np.radians(73.3)

    L_reg = 5e-3        # m
    a_reg = 2e-3        # m
    b_reg = 3.34e-3     # m
    g_reg = 1e-3        # m
    theta_f_reg = np.radians(75)

    L_cyl = 4e-3    # m
    g_cyl = 5e-4    # m

    # T_e = 1.78      # eV
    # n_e = 5.1e19    # m^-3
    # fwhm = 14.3     # mm
    # T_e = 0.67      # eV
    # n_e = 2.3e19    # m^-3
    # fwhm = 16       # mm
    # T_e = 1.68
    # n_e = 1.93e19
    # fwhm = 16.8
    # T_e = 0.75
    # n_e = 1.3e20
    # fwhm = 16.8
    # T_e = 0.76
    # n_e = 1.0e20
    # fwhm = 16.8
    T_e = 1.61
    n_e = 1.41e20
    fwhm = 12.4
    deg_freedom = 3
    gamma_i = (deg_freedom + 2) / 2
    d_perp = 3e-4  # m
    theta_p = np.radians(10)
    theta_perp = np.radians(10)

    probe_s = lp.AngledTipProbe(a_small, b_small, L_small, g_small, d_perp, theta_f_small, theta_p)
    probe_l = lp.AngledTipProbe(a_large, b_large, L_large, g_large, d_perp, theta_f_large, theta_p)
    probe_r = lp.AngledTipProbe(a_reg, b_reg, L_reg, g_reg, d_perp, theta_f_reg, theta_p)
    probe_c = lp.FlushCylindricalProbe(L_cyl/2, g_cyl, d_perp)

    A_coll_s = lp.calc_probe_collection_area(a_small, b_small, L_small, g_small, d_perp, theta_perp, theta_p,
                                             theta_f_small, print_fl=False)
    A_coll_l = lp.calc_probe_collection_area(a_large, b_large, L_large, g_large, d_perp, theta_perp, theta_p,
                                             theta_f_large, print_fl=False)
    A_coll_r = lp.calc_probe_collection_area(a_reg, b_reg, L_reg, g_reg, d_perp, theta_perp, theta_p, theta_f_reg,
                                             print_fl=False)
    A_coll_c = probe_c.get_collection_area(theta_perp)

    print('Small area: {}, Large area: {}, Regular area: {}, Cylindrical area: {}'.format(A_coll_s, A_coll_l, A_coll_r,
                                                                                          A_coll_c))

    # Plotting analytical IV over the top of the raw IVs

    print(fit_df_0)

    plt.figure()
    # for iv_curve in magopter.iv_arr_coax_0:
    #     plt.plot(iv_curve.voltage, iv_curve.current)

    plt.plot(iv_data[c.RAW_X].tolist()[0], iv_data[c.RAW_Y].tolist()[0], 'x', label='Raw IV')
    plt.plot(iv_data[c.RAW_X].tolist()[0], iv_data[c.FIT_Y].tolist()[0], label='Fit IV')
    iv_interp = itp.interp1d(iv_data[c.RAW_X].tolist()[0], iv_data[c.RAW_Y].tolist()[0])
    iv_v_f = -10
    I_s = lp.analytical_iv_curve(iv_data[c.RAW_X].tolist()[0], iv_v_f, T_e, n_e, theta_perp, A_coll_s, L=L_small, g=g_small)
    I_c = lp.analytical_iv_curve(iv_data[c.RAW_X].tolist()[0], iv_v_f, T_e, n_e, theta_perp, A_coll_c, L=L_small, g=g_small)

    plt.plot(iv_data[c.RAW_X].tolist()[0], I_s, label='Analytical', linestyle='dashed', linewidth=1, color='r')
    # plt.plot(iv_data[c.RAW_X].tolist()[0], I_c, label='Analytical (c)', linestyle='dashed', linewidth=1, color='g')
    plt.legend()
    plt.title('Comparison of analytical to measured IV curves for the small area probe')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    # A_coll_s = calc_probe_collection_A_alt(a_small, b_small, L_small, theta_perp, theta_p)
    # A_coll_l = calc_probe_collection_A_alt(a_large, b_large, L_large, theta_perp, theta_p)
    # A_coll_l = (26.25 * 1e-6) * np.sin(theta_perp + theta_p)
    # print('Small area: {}, Large area: {}'.format(A_coll_s, A_coll_l))

    c_s = np.sqrt((nrm.ELEM_CHARGE * (T_e + gamma_i * T_e)) / nrm.PROTON_MASS)
    n_e_0 = fit_df_0[c.ION_SAT] / (nrm.ELEM_CHARGE * c_s * A_coll_s)
    n_e_1 = fit_df_1[c.ION_SAT] / (nrm.ELEM_CHARGE * c_s * A_coll_c)
    I_sat_0 = c_s * n_e * nrm.ELEM_CHARGE * A_coll_s
    I_sat_1 = c_s * n_e * nrm.ELEM_CHARGE * A_coll_c

    J_sat_0 = fit_df_0[c.ION_SAT] / A_coll_s
    J_sat_1 = fit_df_1[c.ION_SAT] / A_coll_c

    plt.figure()
    plt.subplot(221)
    plt.title('Electron Temperature Measurements')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$T_e$ (eV)')
    plt.errorbar(fit_df_0.index, c.ELEC_TEMP, yerr=c.ERROR_STRING.format(c.ELEC_TEMP), data=fit_df_0, fmt='x',
                 label='Half area')
    plt.errorbar(fit_df_1.index, c.ELEC_TEMP, yerr=c.ERROR_STRING.format(c.ELEC_TEMP), data=fit_df_1, fmt='x',
                 label='Cylinder area')
    plt.axhline(y=T_e, linestyle='dashed', linewidth=1, color='r', label='TS')
    plt.legend()

    plt.subplot(222)
    plt.title('Ion Saturation Current Measurements')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$I^+_{sat}$ (eV)')
    plt.errorbar(fit_df_0.index, c.ION_SAT, yerr=c.ERROR_STRING.format(c.ION_SAT), data=fit_df_0, label='Half area',
                 fmt='x')
    plt.errorbar(fit_df_1.index, c.ION_SAT, yerr=c.ERROR_STRING.format(c.ION_SAT), data=fit_df_1, label='Cylinder area',
                 fmt='x')
    # for arc in magopter.arcs:
    #     plt.axvline(x=arc, linestyle='dashed', linewidth=1, color='r')
    plt.axhline(y=I_sat_0, linestyle='dashed', linewidth=1, color='r', label='Expected I_sat (s)')


    plt.legend()

    # plt.figure()
    # plt.subplot(223)
    # plt.title('Current Density Measurements')
    # plt.xlabel('Time (s)')
    # plt.ylabel(r'$J_{sat}$ (Am$^{-2}$)')
    # plt.plot(fit_df_0.index, J_sat_0, label='Half area')
    # plt.plot(fit_df_1.index, J_sat_1, label='Cylinder area')
    # for arc in magopter.arcs:
    #     plt.axvline(x=arc, linestyle='dashed', linewidth=1, color='r')
    # plt.legend()

    # plt.figure()
    plt.subplot(223)
    plt.title('Electron Density Measurements')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$n_e$ (m$^{-3}$)')
    plt.plot(fit_df_0.index, n_e_0, 'x', label='Half Area')
    plt.plot(fit_df_1.index, n_e_1, 'x', label='Cylinder Area')
    plt.axhline(y=n_e, linestyle='dashed', linewidth=1, color='r', label='TS')
    plt.legend()

    a_s = lp.calc_sheath_expansion_coeff(T_e, n_e, L_small, g_small, theta_perp)
    a_c = lp.calc_sheath_expansion_coeff(T_e, n_e, L_cyl, g_cyl, theta_perp)
    print(a_s, a_c)

    plt.subplot(224)
    plt.title('Sheath Expansion Coefficient Measurements')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$a$')
    plt.errorbar(fit_df_0.index, c.SHEATH_EXP, yerr=c.ERROR_STRING.format(c.SHEATH_EXP), data=fit_df_0, fmt='x',
                 label='Half Area')
    plt.errorbar(fit_df_1.index, c.SHEATH_EXP, yerr=c.ERROR_STRING.format(c.SHEATH_EXP), data=fit_df_1, fmt='x',
                 label='Cylinder Area')
    plt.axhline(y=a_s, linestyle='dashed', linewidth=1, color='r', label='Expected - small')
    plt.axhline(y=a_c, linestyle='dashed', linewidth=1, color='b', label='Expected - cyl')

    plt.legend()

    # plt.figure()
    # # plt.plot(fit_dfs[c.ELEC_TEMP])
    # plt.errorbar(fit_dfs.index, c.ION_SAT, yerr='d_{}'.format(c.ION_SAT), data=fit_dfs)

    # plt.figure()
    # plt.plot(*fit_arrs[0][0].get_raw_plottables())
    # plt.plot(*fit_arrs[0][0].get_fit_plottables())

    # plt.figure()
    # plt.plot(raw_voltage)
    # plt.plot(magopter.m_data.data[6][m:n])
    #
    # plt.figure()
    # plt.plot(raw_voltage)
    # print(peaks)
    # print(peaks[0])
    # for peak in list(peaks[0]):
    #     plt.axvline(x=peak, linestyle='dashed', linewidth=1, color='r')

    # plt.figure()
    # plt.plot(raw_voltage - smoothed_voltage)

    # plt.figure()
    # for ch in magopter.m_data.channels:
    #     plt.plot(magopter.m_data.data[ch])
    # plt.plot(magopter.m_data.data[5][m:n], magopter.m_data.data[6][m:n])
    # plt.plot(np.linspace(m, n, n - m), np.log(np.abs(magopter.t_file[m:n])))


    # plt.figure()
    # plt.plot(magopter.m_data.data[5][i:j])
    # plt.plot(magopter.m_data.data[6][i:j])
    # plt.plot(sig.savgol_filter(magopter.m_data.data[5][i:j], 15, 2))
    #
    # plt.figure()
    # plt.plot(magopter.m_data.data[5][i:j], magopter.m_data.data[6][i:j])

    plt.show()

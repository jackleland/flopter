import lputils as lp
import matplotlib.pyplot as plt
import numpy as np
import os
from magopter import Magopter, MagnumProbes
import glob
import external.magnumdbutils as ut
import external.readfastadc as adc
import constants as c
import normalisation as nrm
import databases.magnum as mag
from scipy.interpolate import interp1d
import scipy.signal as sig
import fitters as f
from tkinter.filedialog import askopenfilename


def main_magopter_analysis():
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

    magopter = Magopter(folder, ts_file)
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
    magopter.prepare(down_sampling_rate=dsr, plot_fl=True)
    # magopter.simple_relative_trim(trim_end=0.82)
    magopter.trim(trim_end=0.83)
    fit_df_0, fit_df_1 = magopter.fit()

    iv_data = fit_df_0.iloc[[125]]
    plt.figure()
    for iv_curve in magopter.iv_arrs[0]:
        plt.plot(iv_curve.time, iv_curve.current)
    plt.axvline(x=iv_data.index)

    # Flush probe measurements
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
    probe_c = lp.FlushCylindricalProbe(L_cyl / 2, g_cyl, d_perp)

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
    iv_v_f = -10
    I_s = lp.analytical_iv_curve(iv_data[c.RAW_X].tolist()[0], iv_v_f, T_e, n_e, theta_perp, A_coll_s, L=L_small,
                                 g=g_small)
    I_c = lp.analytical_iv_curve(iv_data[c.RAW_X].tolist()[0], iv_v_f, T_e, n_e, theta_perp, A_coll_c, L=L_small,
                                 g=g_small)

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

    plt.show()


def integrated_analysis(probe_coax_0, probe_coax_1, folder, file, ts_file=None):
    magopter = Magopter(folder, file, ts_filename=ts_file)
    dsr = 10
    magopter.prepare(down_sampling_rate=dsr, roi_b_plasma=True)
    magopter.trim(trim_end=0.83)
    fit_df_0, fit_df_1 = magopter.fit()

    theta_perp = np.radians(10)
    A_coll_0 = probe_coax_0.get_collection_area(theta_perp)
    A_coll_1 = probe_coax_1.get_collection_area(theta_perp)

    if magopter.ts_temp is not None:
        temps = [np.max(temp) / nrm.ELEM_CHARGE for temp in magopter.ts_temp[mag.DATA]]
        denss = [np.max(dens) for dens in magopter.ts_dens[mag.DATA]]
        T_e = np.mean(temps)
        d_T_e = np.std(temps) / np.sqrt(len(temps))
        n_e = np.mean(denss)
        d_n_e = np.std(denss) / np.sqrt(len(denss))
        print('T = {}+-{}, n = {}+-{}'.format(T_e, d_T_e, n_e, d_n_e))
    else:
        T_e = 1.61
        d_T_e = 0.01
        n_e = 1.41e20
        d_n_e = 0.01e20
        fwhm = 12.4

    # t_0 = -0.35
    t_0 = 0
    target_pos_t, target_pos_x = magopter.magnum_data[mag.TARGET_POS]
    # target_pos_t, target_pos_x = magopter.magnum_db.pad_continuous_variable(magopter.magnum_data[mag.TARGET_POS])
    target_pos_t = np.array(target_pos_t)

    target_voltage_t = np.array(magopter.magnum_data[mag.TARGET_VOLTAGE][0])
    target_voltage_x = np.array(magopter.magnum_data[mag.TARGET_VOLTAGE][1])

    deg_freedom = 2
    # gamma_i = (deg_freedom + 2) / 2
    gamma_i = 1
    c_s_0 = np.sqrt((nrm.ELEM_CHARGE * (fit_df_0[c.ELEC_TEMP] + gamma_i * fit_df_0[c.ELEC_TEMP])) / nrm.PROTON_MASS)
    c_s_1 = np.sqrt((nrm.ELEM_CHARGE * (fit_df_1[c.ELEC_TEMP] + gamma_i * fit_df_1[c.ELEC_TEMP])) / nrm.PROTON_MASS)
    n_e_0 = fit_df_0[c.ION_SAT] / (nrm.ELEM_CHARGE * c_s_0 * A_coll_0)
    n_e_1 = fit_df_1[c.ION_SAT] / (nrm.ELEM_CHARGE * c_s_1 * A_coll_1)

    I_sat_0 = c_s_0 * n_e * nrm.ELEM_CHARGE * A_coll_0
    I_sat_1 = c_s_1 * n_e * nrm.ELEM_CHARGE * A_coll_1

    J_sat_0 = fit_df_0[c.ION_SAT] / A_coll_0
    J_sat_1 = fit_df_1[c.ION_SAT] / A_coll_1

    plt.figure()
    ax1 = plt.subplot(211)
    plt.title('Electron Temperature Measurements')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$T_e$ (eV)')
    plt.errorbar(fit_df_0.index, c.ELEC_TEMP, yerr=c.ERROR_STRING.format(c.ELEC_TEMP), data=fit_df_0, fmt='x',
                 label='Half area')
    plt.errorbar(fit_df_1.index, c.ELEC_TEMP, yerr=c.ERROR_STRING.format(c.ELEC_TEMP), data=fit_df_1, fmt='x',
                 label='Cylinder area')
    plt.axhline(y=T_e, linestyle='dashed', linewidth=1, color='gray', label='TS')
    plt.axhline(y=T_e + d_T_e, linestyle='dotted', linewidth=0.5, color='gray')
    plt.axhline(y=T_e - d_T_e, linestyle='dotted', linewidth=0.5, color='gray')
    # plt.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = plt.subplot(212, sharex=ax1)
    plt.title('Electron Density Measurements')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$n_e$ (m$^{-3}$)')
    plt.plot(fit_df_0.index, n_e_0, 'x', label='Half Area')
    plt.plot(fit_df_1.index, n_e_1, 'x', label='Cylinder Area')
    plt.axhline(y=n_e, linestyle='dashed', linewidth=1, color='gray', label='TS')
    # plt.legend()
    plt.setp(ax2.get_xticklabels(), visible=False)

    for ax in [ax1, ax2]:
        ax3 = ax.twinx()
        # for i in range(np.shape(magopter.magnum_data[mag.BEAM_DUMP_DOWN])[1]):
        #     if magopter.magnum_data[mag.BEAM_DUMP_DOWN][1][i]:
        #         plt.axvline(x=magopter.magnum_data[mag.BEAM_DUMP_DOWN][0][i], color='r', linestyle='--', linewidth=1)
        #     if magopter.magnum_data[mag.BEAM_DUMP_UP][1][i]:
        #         plt.axvline(x=magopter.magnum_data[mag.BEAM_DUMP_UP][0][i], color='b', linestyle='--', linewidth=1)

        for j in range(np.shape(magopter.magnum_data[mag.PLASMA_STATE])[1]):
            plt.axvline(x=magopter.magnum_data[mag.PLASMA_STATE][0][j], color='g', linestyle='--', linewidth=1)

        # for k in range(np.shape(magopter.magnum_data[mag.TRIGGER_START])[1]):
        #     plt.axvline(x=magopter.magnum_data[mag.TRIGGER_START][0][k], color='b', linestyle='--', linewidth=1)
        #     print(magopter.magnum_data[mag.TRIGGER_START][0][k])

        if magopter.ts_temp is not None:
            for k in range(len(magopter.ts_temp[0])):
                if k == 0:
                    plt.axvline(x=magopter.ts_temp[0][k], color='m', linestyle='--', linewidth=1, label='TS')
                else:
                    plt.axvline(x=magopter.ts_temp[0][k], color='m', linestyle='--', linewidth=1)

        plt.plot(target_pos_t, target_pos_x, color='k', label='Target Position')
        # plt.axvline(x=0, color='k', linewidth=1, linestyle='-.')
        plt.xlabel('Time (s)')
        plt.legend()

    #########################################
    #            Whole IV plot              #
    #########################################
    fig, ax = plt.subplots()
    max_currents = [[], []]
    for iv_curve in magopter.iv_arrs[0]:
        plt.plot(iv_curve[c.TIME], -iv_curve[c.CURRENT])
        # plt.plot(iv_curve.time, iv_curve.voltage)
        max_current = np.max(iv_curve[c.CURRENT])
        max_currents[1].append(np.max(iv_curve[c.CURRENT]))
        max_currents[0].append(iv_curve[c.TIME][list(iv_curve[c.CURRENT]).index(max_current)])

    ax1 = ax.twinx()
    plt.plot(target_pos_t, target_pos_x, color='k', label='Target Position')
    # plt.plot(target_voltage_t,target_voltage_x, color='m', label='Target Voltage')
    for arc in magopter.arcs:
        plt.axvline(x=arc, color='r', linewidth=1, linestyle='-.')

    iv_data = fit_df_0.iloc[[10]]
    plt.axvline(x=iv_data.index, color='gray', linestyle='--')

    #########################################
    #        Analytical IV Comparison       #
    #########################################
    plt.figure()

    plt.plot(iv_data[c.RAW_X].tolist()[0], iv_data[c.RAW_Y].tolist()[0], 'x', label='Raw IV')
    plt.plot(iv_data[c.RAW_X].tolist()[0], iv_data[c.FIT_Y].tolist()[0], label='Fit IV')
    v_f_fitted = iv_data[c.FLOAT_POT].values[0]
    n_e_fitted = n_e_0.iloc[[10]].values[0]
    I_s = probe_coax_0.get_analytical_iv(iv_data[c.RAW_X].tolist()[0], v_f_fitted, theta_perp, T_e, n_e)
    I_s_shifted = probe_coax_0.get_analytical_iv(iv_data[c.RAW_X].tolist()[0], v_f_fitted, theta_perp, T_e, n_e_fitted)

    plt.plot(iv_data[c.RAW_X].tolist()[0], I_s, label='Analytical 1', linestyle='dashed', linewidth=1, color='r')
    plt.plot(iv_data[c.RAW_X].tolist()[0], I_s_shifted, label='Analytical 2', linestyle='dashed', linewidth=1, color='g')
    plt.legend()
    plt.title('Comparison of analytical to measured IV curves for the small area probe')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')

    #########################################
    #           target_pos sweep            #
    #########################################
    plt.figure()
    ax1 = plt.subplot(211)
    plt.title('Smoothed Max Current')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$I^+_{sat}$ (eV)')
    plt.plot(max_currents[0], sig.savgol_filter(max_currents[1], 51, 2), 'xk', label='Max current')
    # plt.axhline(y=I_sat_0, linestyle='dashed', linewidth=1, color='r', label='Expected I_sat (s)')

    # ax2 = ax1.twinx()
    # for t_0 in np.linspace(-3, 1, 11):
    plt.plot(target_pos_t, target_pos_x + 1, label='Target Position')
    plt.axvline(x=0, color='k', linewidth=1, linestyle='-.')
    plt.xlabel('Time (s)')
    plt.legend()

    # plt.subplot(312)
    # plt.plot(max_currents[0], np.gradient(sig.savgol_filter(max_currents[1], 51, 2)), 'xk', label='Max current')
    # plt.plot(target_pos_t, np.gradient(target_pos_x + 1), label='t_0 = {:.1f}'.format(t_0))

    plt.subplot(212)
    # matching_times = []
    # for t in (target_pos_t):
    #     matching_times.append(min(abs(t - fit_df_0.index)))
    target_pos_func = interp1d(target_pos_t, target_pos_x)
    lo = min(target_pos_t)
    hi = max(target_pos_t)
    # t_range = np.linspace(lo, hi, len(fit_df_0[c.ION_SAT]))

    plt.plot(target_pos_func(max_currents[0]), max_currents[1], 'kx')
    plt.xlabel('Target position (m)')
    plt.ylabel('Ion saturation current (A)')
    plt.legend()

    ##############################
    #   density vs target pos.

    plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(fit_df_0.index, n_e_0, 'kx')
    plt.ylabel(r'Density (m$^{-3}$)')
    plt.axhline(y=n_e, color='gray', linestyle='--', linewidth=2)
    ax2 = ax1.twinx()
    ax2.plot(target_pos_t, target_pos_x, color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_ylabel('Target position', color='r')
    plt.xlabel('Time (s)')

    plt.subplot(212)
    # densities = np.array(max_currents[1]) / (nrm.ELEM_CHARGE * c_s * A_coll_0)
    # plt.plot(target_pos_func(max_currents[0]), densities, 'kx')
    plt.plot(target_pos_func(fit_df_0.index), n_e_0, 'kx', label='Measured')
    plt.plot(2.93e-3, n_e, 'bx', label='TS Result')
    plt.xlabel('Target position (m)')
    plt.ylabel(r'Density (m$^{-3}$)')
    plt.legend()


def ts_ir_comparison(probe_0, probe_1, folder, file, ts_file):
    dsr = 5

    m_ir = Magopter(folder, file)
    m_ir.prepare(down_sampling_rate=dsr)
    m_ir.trim(trim_end=0.83)
    fit_ir_df_0, fit_ir_df_1 = m_ir.fit()

    m_ts = Magopter(folder, ts_file)
    m_ts.prepare(down_sampling_rate=dsr)
    m_ts.trim(trim_end=0.83)
    fit_ts_df_0, fit_ts_df_1 = m_ts.fit()

    tarpos_t_ir = np.array(m_ir.magnum_data[mag.TARGET_POS][0])
    tarpos_x_ir = m_ir.magnum_data[mag.TARGET_POS][1]
    tarpos_func_ir = interp1d(tarpos_t_ir, tarpos_x_ir)

    tarpos_t_ts = np.array(m_ts.magnum_data[mag.TARGET_POS][0])
    tarpos_x_ts = m_ts.magnum_data[mag.TARGET_POS][1]
    tarpos_func_ts = interp1d(tarpos_t_ts, tarpos_x_ts)

    theta_perp = np.radians(10)
    A_coll_0 = probe_0.get_collection_area(theta_perp)
    A_coll_1 = probe_1.get_collection_area(theta_perp)
    if m_ts.ts_temp is not None:
        T_e = np.mean([np.max(temp) for temp in m_ts.ts_temp[mag.DATA]]) / nrm.ELEM_CHARGE
        n_e = np.mean([np.max(dens) for dens in m_ts.ts_dens[mag.DATA]])
        print('T = {}, n = {}'.format(T_e, n_e))
    else:
        T_e = 1.61
        n_e = 1.41e20
        fwhm = 12.4

    plt.figure()
    ax1 = plt.subplot(211)
    plt.title('Small Probe Electron Temperature Measurements')
    plt.ylabel(r'$T_e$ (eV)')
    plt.errorbar(fit_ir_df_0.index, c.ELEC_TEMP, yerr=c.ERROR_STRING.format(c.ELEC_TEMP), data=fit_ir_df_0, fmt='x',
                 label='IR position')
    plt.errorbar(fit_ts_df_0.index, c.ELEC_TEMP, yerr=c.ERROR_STRING.format(c.ELEC_TEMP), data=fit_ts_df_0, fmt='x',
                 label='TS position')
    plt.axhline(y=T_e, linestyle='dashed', linewidth=1, color='gray', label='TS')
    plt.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)

    deg_freedom = 3
    gamma_i = (deg_freedom + 2) / 2
    c_s = np.sqrt((nrm.ELEM_CHARGE * (T_e + gamma_i * T_e)) / nrm.PROTON_MASS)
    n_e_ir = fit_ir_df_0[c.ION_SAT] / (nrm.ELEM_CHARGE * c_s * A_coll_0)
    n_e_ts = fit_ts_df_0[c.ION_SAT] / (nrm.ELEM_CHARGE * c_s * A_coll_1)

    ax2 = plt.subplot(212, sharex=ax1)
    plt.title('Electron Density Measurements')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$n_e$ (m$^{-3}$)')
    plt.plot(fit_ir_df_0.index, n_e_ir, 'x', label='IR position')
    plt.plot(fit_ts_df_0.index, n_e_ts, 'x', label='TS position')
    plt.axhline(y=n_e, linestyle='dashed', linewidth=1, color='gray', label='TS')
    plt.legend()
    # plt.setp(ax2.get_xticklabels(), visible=False)

    fitter = f.ExponentialFitter()
    fitdata_ir = fitter.fit(tarpos_func_ir(fit_ir_df_0.index), n_e_ir)
    fitdata_ts = fitter.fit(tarpos_func_ts(fit_ts_df_0.index), n_e_ts)
    plt.figure()
    plt.plot(tarpos_func_ir(fit_ir_df_0.index), n_e_ir, 'kx', label='Infrared')
    plt.plot(tarpos_func_ts(fit_ts_df_0.index), n_e_ts, 'mx', label='Thomson')
    plt.plot(2.93e-3, n_e, 'bx', label='TS Result')
    fitdata_ir.plot(show_fl=False)
    fitdata_ts.plot(show_fl=False)
    plt.xlabel('Target position (m)')
    plt.ylabel(r'Density (m$^{-3}$)')
    plt.legend()

    plt.figure()
    plt.semilogy(tarpos_func_ir(fit_ir_df_0.index), n_e_ir, 'kx', label='Infrared')
    plt.semilogy(tarpos_func_ts(fit_ts_df_0.index), n_e_ts, 'mx', label='Thomson')
    plt.plot(2.93e-3, n_e, 'bx', label='TS Result')
    plt.xlabel('Target position (m)')
    plt.ylabel(r'Density (m$^{-3}$)')
    plt.legend()


def deeper_iv_analysis(probe_0, folder, file, plot_comparison_fl=False, plot_timeline_fl=False):
    magopter = Magopter(folder, file, ts_filename=ts_file)
    dsr = 1
    magopter.prepare(down_sampling_rate=dsr, roi_b_plasma=True, plot_fl=False, crit_freq=None, crit_ampl=None)
    print('0: {}, 1: {}'.format(len(magopter.iv_arrs[0]), len(magopter.iv_arrs[1])))

    index = int(0.5 * len(magopter.iv_arrs[0]))
    if plot_timeline_fl:
        magopter.quick_plot(coax=0, index=index)

    magopter.iv_arrs[0] = magopter.iv_arrs[0][index:index + 3]
    magopter.iv_arrs[1] = []
    magopter.trim(trim_beg=0.05, trim_end=0.7)
    fit_df_0, fit_df_1 = magopter.fit(print_fl=True)

    if magopter.ts_temp is not None:
        temps = [np.max(temp) / nrm.ELEM_CHARGE for temp in magopter.ts_temp[mag.DATA]]
        denss = [np.max(dens) for dens in magopter.ts_dens[mag.DATA]]
        T_e_ts = np.mean(temps)
        d_T_e_ts = np.std(temps) / np.sqrt(len(temps))
        n_e_ts = np.mean(denss)
        d_n_e_ts = np.std(denss) / np.sqrt(len(denss))
    else:
        T_e_ts = 1.61
        d_T_e_ts = 0.01
        n_e_ts = 1.4e20
        d_n_e_ts = 0.1e20

    count = fit_df_0[c.ELEC_TEMP].count()
    positions = [0.1, 0.5, 0.7]
    iv_indices = [int(pos * count) for pos in positions]
    iv_datas = [fit_df_0.iloc[[iv_index]] for iv_index in iv_indices]
    print('count={}, 0={}, 1={}, 2={}'.format(count, *iv_indices))

    alpha = 9.95
    theta_perp = np.radians(alpha)
    d_theta_perp = np.radians(0.8)
    A_coll_0 = probe_0.get_collection_area(theta_perp)
    d_A_coll = np.abs(probe_0.get_collection_area(theta_perp + d_theta_perp) - A_coll_0)

    # deg_freedom = 2
    # gamma_i = (deg_freedom + 2) / 2
    gamma_i = 1
    c_s = np.sqrt((nrm.ELEM_CHARGE * (fit_df_0[c.ELEC_TEMP] + gamma_i * fit_df_0[c.ELEC_TEMP])) / nrm.PROTON_MASS)
    d_c_s = np.abs((c_s * fit_df_0[c.ERROR_STRING.format(c.ELEC_TEMP)]) / (2 * fit_df_0[c.ELEC_TEMP]))

    n_e = fit_df_0[c.ION_SAT] / (nrm.ELEM_CHARGE * c_s * A_coll_0)
    d_n_e = np.abs(n_e) * np.sqrt((d_c_s / c_s) ** 2 + (d_A_coll / A_coll_0) ** 2 + (
            fit_df_0[c.ERROR_STRING.format(c.ION_SAT)] / fit_df_0[c.ION_SAT]) ** 2)

    ##################################################
    #            Time line of IVs in shot            #
    ##################################################

    if plot_timeline_fl:
        plt.figure()
        plt.errorbar(fit_df_0.index, n_e, yerr=d_n_e, fmt='x', color='silver', label=r'$\alpha$ = {}'.format(alpha))

        plt.axhline(y=n_e_ts, linestyle='dashed', linewidth=1, color='m', label='TS')
        plt.axhline(y=n_e_ts + d_n_e_ts, linestyle='dotted', linewidth=0.5, color='m')
        plt.axhline(y=n_e_ts - d_n_e_ts, linestyle='dotted', linewidth=0.5, color='m')
        for i, colour in enumerate(['r', 'b', 'g']):
            plt.axvline(x=iv_datas[i].index, color=colour)
        plt.ylabel(r'Density (m$^{-3}$)')
        plt.xlabel('Time (s)')
        plt.legend()

        plt.figure()
        plt.errorbar(fit_df_0.index, fit_df_0[c.ELEC_TEMP], yerr=fit_df_0[c.ERROR_STRING.format(c.ELEC_TEMP)], fmt='x',
                     color='silver', label=r'$\alpha$ = {}'.format(alpha))

        plt.axhline(y=T_e_ts, linestyle='dashed', linewidth=1, color='m', label='TS')
        plt.axhline(y=T_e_ts + d_T_e_ts, linestyle='dotted', linewidth=0.5, color='m')
        plt.axhline(y=T_e_ts - d_T_e_ts, linestyle='dotted', linewidth=0.5, color='m')
        for i, colour in enumerate(['r', 'b', 'g']):
            plt.axvline(x=iv_datas[i].index, color=colour)
        plt.ylabel(r'Temperature (eV)')
        plt.xlabel('Time (s)')
        plt.legend()

    ##################################################
    #         Examination of 3 different IVs         #
    ##################################################

    for i, iv_data in enumerate(iv_datas):
        # Extract individual values from dataframe
        v_f_fitted = iv_data[c.FLOAT_POT].values[0]
        T_e_fitted = iv_data[c.ELEC_TEMP].values[0]
        a_fitted = iv_data[c.SHEATH_EXP].values[0]
        I_sat_fitted = iv_data[c.ION_SAT].values[0]

        d_v_f_fitted = iv_data[c.ERROR_STRING.format(c.FLOAT_POT)].values[0]
        d_T_e_fitted = iv_data[c.ERROR_STRING.format(c.ELEC_TEMP)].values[0]
        d_a_fitted = iv_data[c.ERROR_STRING.format(c.SHEATH_EXP)].values[0]
        d_I_sat_fitted = iv_data[c.ERROR_STRING.format(c.ION_SAT)].values[0]

        c_s_fitted = lp.sound_speed(T_e_fitted, gamma_i=1)
        d_c_s_fitted = lp.d_sound_speed(c_s_fitted, T_e_fitted, d_T_e_fitted)
        n_e_fitted = lp.electron_density(I_sat_fitted, c_s_fitted, A_coll_0)
        d_n_e_fitted = lp.d_electron_density(n_e_fitted, c_s_fitted, d_c_s_fitted, A_coll_0, d_A_coll, I_sat_fitted,
                                             d_I_sat_fitted)

        print('iv = {}: \n'
              '\t v_f = {:.3g} +- {:.1g} \n'
              '\t T_e = {:.3g} +- {:.1g} \n'
              '\t I_sat = {:.3g} +- {:.1g} \n'
              '\t n_e = {:.3g} +- {:.1g} \n'
              '\t a = {:.3g} +- {:.1g} \n'
              '\t c_s = {:.3g} +- {:.1g} \n'
              '\t A_coll = {:.3g} +- {:.1g} \n'
              .format(i, v_f_fitted, d_v_f_fitted, T_e_fitted, d_T_e_fitted, I_sat_fitted, d_I_sat_fitted, n_e_fitted,
                      d_n_e_fitted, a_fitted, d_a_fitted, c_s_fitted, d_c_s_fitted, A_coll_0, d_A_coll))

        I_f = probe_0.get_analytical_iv(iv_data[c.RAW_X].tolist()[0], v_f_fitted, theta_perp, T_e_fitted, n_e_fitted,
                                        print_fl=True)
        I_ts = probe_0.get_analytical_iv(iv_data[c.RAW_X].tolist()[0], v_f_fitted, theta_perp, T_e_ts, n_e_ts,
                                         print_fl=True)
        plt.figure()
        plt.errorbar(iv_data[c.RAW_X].tolist()[0], iv_data[c.RAW_Y].tolist()[0], yerr=iv_data[c.SIGMA].tolist()[0],
                     fmt='x', label='Raw IV', ecolor='silver')
        plt.plot(iv_data[c.RAW_X].tolist()[0], I_f, label='Analytical - measured', linestyle='dashed', linewidth=1, color='r')
        plt.plot(iv_data[c.RAW_X].tolist()[0], I_ts, label='Analytical - TS', linestyle='dashed', linewidth=1, color='m')
        plt.plot(iv_data[c.RAW_X].tolist()[0], iv_data[c.FIT_Y].tolist()[0], color='orange', label='Fit IV')
        plt.legend()
        plt.title('Comparison of analytical to measured IV curves for the small area probe')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')

    ##################################################
    #         Comparison of Parameter Scales         #
    ##################################################

    if plot_comparison_fl:
        plt.figure()
        for alpha in [2, 3, 4, 6, 8, 10]:
            theta_perp = np.radians(alpha)
            d_theta_perp = np.radians(0.1)
            A_coll_0 = probe_0.get_collection_area(theta_perp)
            d_A_coll = np.abs(probe_0.get_collection_area(theta_perp + d_theta_perp) - A_coll_0)

            # deg_freedom = 2
            # gamma_i = (deg_freedom + 2) / 2
            gamma_i = 1
            c_s = lp.sound_speed(fit_df_0[c.ELEC_TEMP])
            d_c_s = lp.d_sound_speed(c_s, fit_df_0[c.ELEC_TEMP], fit_df_0[c.ERROR_STRING.format(c.ELEC_TEMP)])

            # n_e = fit_df_0[c.ION_SAT] / (nrm.ELEM_CHARGE * c_s * A_coll_0)
            n_e = lp.electron_density(fit_df_0[c.ION_SAT], c_s, A_coll_0)
            d_n_e = lp.d_electron_density(n_e, c_s, d_c_s, A_coll_0, d_A_coll, fit_df_0[c.ION_SAT],
                                          fit_df_0[c.ERROR_STRING.format(c.ION_SAT)])
            plt.errorbar(fit_df_0.index, n_e, yerr=d_n_e, fmt='x', label=r'$\alpha$ = {}'.format(alpha))

        plt.axhline(y=n_e_ts, linestyle='dashed', linewidth=1, color='red', label='TS')
        plt.axhline(y=n_e_ts + d_n_e_ts, linestyle='dotted', linewidth=0.5, color='red')
        plt.axhline(y=n_e_ts - d_n_e_ts, linestyle='dotted', linewidth=0.5, color='red')
        plt.ylabel(r'Density (m$^{-3}$)')
        plt.xlabel('Time (s)')
        plt.legend()

        plt.figure()
        for I_sat_scale in [0.5, 1.0, 2.5]:
            theta_perp = np.radians(9.95)
            d_theta_perp = np.radians(0.1)
            A_coll_0 = probe_0.get_collection_area(theta_perp)
            d_A_coll = np.abs(probe_0.get_collection_area(theta_perp + d_theta_perp) - A_coll_0)

            deg_freedom = 2
            gamma_i = (deg_freedom + 2) / 2
            # gamma_i = 1
            c_s = np.sqrt((nrm.ELEM_CHARGE * (fit_df_0[c.ELEC_TEMP] + gamma_i * fit_df_0[c.ELEC_TEMP])) / nrm.PROTON_MASS)
            d_c_s = np.abs((c_s * fit_df_0[c.ERROR_STRING.format(c.ELEC_TEMP)]) / (2 * fit_df_0[c.ELEC_TEMP]))

            I_sat = fit_df_0[c.ION_SAT] * I_sat_scale
            d_I_sat = fit_df_0[c.ERROR_STRING.format(c.ION_SAT)] * I_sat_scale

            n_e = I_sat / (nrm.ELEM_CHARGE * c_s * A_coll_0)
            d_n_e = np.abs(n_e) * np.sqrt((d_c_s / c_s)**2 + (d_A_coll / A_coll_0)**2 + (d_I_sat / I_sat)**2)

            plt.errorbar(fit_df_0.index, n_e, yerr=d_n_e, fmt='x', label=r'$Scale$ = {}'.format(I_sat_scale))

        plt.axhline(y=n_e_ts, linestyle='dashed', linewidth=1, color='red', label='TS')
        plt.axhline(y=n_e_ts + d_n_e_ts, linestyle='dotted', linewidth=0.5, color='red')
        plt.axhline(y=n_e_ts - d_n_e_ts, linestyle='dotted', linewidth=0.5, color='red')
        plt.ylabel(r'Density (m$^{-3}$)')
        plt.xlabel('Time (s)')
        plt.legend()


def multifit_trim_filter_analysis(probe_0, folder, file):
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    fitter = f.FullIVFitter()
    for i, freq in enumerate([None, 4000]):
        magopter = Magopter(folder, file, ts_filename=ts_file)
        magopter.prepare(down_sampling_rate=1, roi_b_plasma=True, crit_freq=freq, crit_ampl=None)

        index = int(0.5 * len(magopter.iv_arrs[0]))
        iv_data = magopter.iv_arrs[0][index]

        fitdata = iv_data.multi_fit(fitter=fitter)

        iv_data.trim_beg = 0.01
        iv_data.trim_end = 0.45
        fitdata_1 = iv_data.multi_fit(fitter=fitter)
        print('{}:{}'.format(iv_data.trim_beg, iv_data.trim_end))
        fitdata_1.print_fit_params()
        # fitdata_1_fvf = iv_data.multi_fit(plot_fl=False, fix_vf_fl=True)

        untrimmed_x = iv_data[c.POTENTIAL]
        untrimmed_y = iv_data[c.CURRENT]

        custom_params = [1.08, 0.006, 4.30, -15.1]

        # Plot the raw signal and the untrimmed fit
        plt.sca(ax[i])
        plt.title('Critical Frequency is {}'.format('{}Hz'.format(freq) if freq is not None else 'not set'))
        plt.errorbar(untrimmed_x, untrimmed_y, fmt='.', yerr=iv_data[c.SIGMA], label='Raw', color='silver', zorder=-1)
        plt.plot(untrimmed_x, fitdata.fit_function(untrimmed_x), label='Fit - No Trim', color='green', zorder=10)

        # Plot the comparion between fixed vf and free vf
        plt.plot(untrimmed_x, fitdata_1.fit_function(untrimmed_x), color='red', linewidth=1,
                 label=r'T_e = {:.3g},  $\chi^2$ = {:.3g}'.format(fitdata_1.get_temp().value, fitdata_1.reduced_chi2))
        plt.axvline(x=np.max(fitdata_1.raw_x), label='Trim Min/Max', color='red', linestyle='dashed', linewidth=1)
        plt.axvline(x=np.min(fitdata_1.raw_x), color='red', linestyle='dashed', linewidth=1)
        # plt.plot(untrimmed_x, fitdata_1_fvf.fit_function(untrimmed_x), label=r'Fixed $V_f$ Fit - {}:{}'
        #          .format(iv_data.trim_end, iv_data.trim_beg), color='red', linestyle='-.')

        # # Plot the raw signal and the untrimmed fit
        # plt.figure()
        # plt.errorbar(untrimmed_x, untrimmed_y, fmt='.', yerr=iv_data[c.SIGMA], label='Raw', color='silver', zorder=-1)
        # plt.plot(untrimmed_x, fitdata.fit_function(untrimmed_x), label='Fit - No Trim', color='green')

        # Trim and plot again
        iv_data.trim_beg = -0.05
        iv_data.trim_end = 0.45
        fitdata_2 = iv_data.multi_fit(fitter=fitter)
        print('{}:{}'.format(iv_data.trim_beg, iv_data.trim_end))
        fitdata_2.print_fit_params()
        # fitdata_2_fvf = iv_data.multi_fit(fix_vf_fl=True)

        plt.plot(untrimmed_x, fitdata_2.fit_function(untrimmed_x), color='blue', linewidth=1,
                 label=r'T_e = {:.3g},  $\chi^2$ = {:.3g}'.format(fitdata_2.get_temp().value, fitdata_2.reduced_chi2))
        plt.axvline(x=np.max(fitdata_2.raw_x), label='Trim Min/Max', color='blue', linestyle='dashed', linewidth=1)
        plt.axvline(x=np.min(fitdata_2.raw_x), color='blue', linestyle='dashed', linewidth=1)
        # plt.plot(untrimmed_x, fitdata_0160_fvf.fit_function(untrimmed_x), label=r'Fixed $V_f$ Fit - {}:{}'
        #          .format(iv_data.trim_end, iv_data.trim_beg), color='blue', linestyle='-.')

        # Trim and plot again
        iv_data.trim_beg = -0.1
        iv_data.trim_end = 0.45
        fitdata_3 = iv_data.multi_fit(fitter=fitter)
        print('{}:{}'.format(iv_data.trim_beg, iv_data.trim_end))
        fitdata_3.print_fit_params()
        # fitdata_3_fvf = iv_data.multi_fit(fix_vf_fl=True)

        plt.plot(untrimmed_x, fitdata_3.fit_function(untrimmed_x), color='orange', linewidth=1,
                 label=r'T_e = {:.3g},  $\chi^2$ = {:.3g}'.format(fitdata_3.get_temp().value, fitdata_3.reduced_chi2))
        plt.axvline(x=np.max(fitdata_3.raw_x), label='Trim Min/Max', color='orange', linestyle='dashed', linewidth=1)
        plt.axvline(x=np.min(fitdata_3.raw_x), color='orange', linestyle='dashed', linewidth=1)

        # Plot the custom fit and an axis line through 0
        # plt.plot(untrimmed_x, f.FullIVFitter().fit_function(untrimmed_x, *custom_params), label='Custom Fit {}'
        #          .format(', '.join([str(i) for i in custom_params])))
        plt.axhline(color='black', linewidth=1)
        plt.ylim(-1.1, 1.6)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.legend()


def multifit_trim_iv_analysis(probe_0, folder, file, trim_upper_fl=False, trim_lower_fl=True):
    magopter = Magopter(folder, file, ts_filename=ts_file)
    magopter.prepare(down_sampling_rate=1, roi_b_plasma=True, plot_fl=False, crit_freq=4000, crit_ampl=None)
    print('0: {}, 1: {}'.format(len(magopter.iv_arrs[0]), len(magopter.iv_arrs[1])))

    index = int(0.5 * len(magopter.iv_arrs[0]))
    magopter.iv_arrs[0] = [magopter.iv_arrs[0][index]]
    magopter.iv_arrs[1] = []

    if magopter.ts_temp is not None:
        temps = [np.max(temp) / nrm.ELEM_CHARGE for temp in magopter.ts_temp[mag.DATA]]
        denss = [np.max(dens) for dens in magopter.ts_dens[mag.DATA]]
        T_e_ts = np.mean(temps)
        d_T_e_ts = np.std(temps) / np.sqrt(len(temps))
        n_e_ts = np.mean(denss)
        d_n_e_ts = np.std(denss) / np.sqrt(len(denss))
    else:
        T_e_ts = 1.61
        d_T_e_ts = 0.01
        n_e_ts = 1.4e20
        d_n_e_ts = 0.1e20

    if not magopter.offline:
        t, data = magopter.magnum_data[mag.TARGET_TILT]
        theta_perp = data.mean()
    else:
        alpha = 9.95
        theta_perp = np.radians(alpha)
    d_theta_perp = np.radians(0.8)
    A_coll_0 = probe_0.get_collection_area(theta_perp)
    d_A_coll = np.abs(probe_0.get_collection_area(theta_perp + d_theta_perp) - A_coll_0)

    if trim_lower_fl:
        trim_lower = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18]
        # trim_upper = [1.0, 0.98, 0.96]
        measured_vals = [[] for dummy in range(len(trim_lower))]
        ivs = [[] for dummy in range(len(trim_lower))]

        for i, tl in enumerate(trim_lower):
            # for tu in trim_upper:
            magopter.trim(trim_beg=tl, trim_end=0.9)
            fit_df_0, fit_df_1 = magopter.fit()
            iv_data = fit_df_0.iloc[[0]]

            ivs[i] = [iv_data[c.RAW_X].tolist()[0], iv_data[c.RAW_Y].tolist()[0]]

            # Extract individual values from dataframe
            v_f = iv_data[c.FLOAT_POT].values[0]
            T_e = iv_data[c.ELEC_TEMP].values[0]
            a = iv_data[c.SHEATH_EXP].values[0]
            I_sat = iv_data[c.ION_SAT].values[0]
            chi2 = iv_data[c.CHI2].values[0]
            red_chi2 = iv_data[c.REDUCED_CHI2].values[0]

            d_v_f = iv_data[c.ERROR_STRING.format(c.FLOAT_POT)].values[0]
            d_T_e = iv_data[c.ERROR_STRING.format(c.ELEC_TEMP)].values[0]
            d_a = iv_data[c.ERROR_STRING.format(c.SHEATH_EXP)].values[0]
            d_I_sat = iv_data[c.ERROR_STRING.format(c.ION_SAT)].values[0]

            c_s = lp.sound_speed(T_e, gamma_i=1)
            d_c_s = lp.d_sound_speed(c_s, T_e, d_T_e)
            n_e = lp.electron_density(I_sat, c_s, A_coll_0)
            d_n_e = lp.d_electron_density(n_e, c_s, d_c_s, A_coll_0, d_A_coll, I_sat, d_I_sat)

            # print('iv = {}: \n'
            #       '\t v_f = {:.3g} +- {:.1g} \n'
            #       '\t T_e = {:.3g} +- {:.1g} \n'
            #       '\t I_sat = {:.3g} +- {:.1g} \n'
            #       '\t n_e = {:.3g} +- {:.1g} \n'
            #       '\t a = {:.3g} +- {:.1g} \n'
            #       '\t c_s = {:.3g} +- {:.1g} \n'
            #       '\t A_coll = {:.3g} +- {:.1g} \n'
            #       .format(i, v_f, d_v_f, T_e, d_T_e, I_sat, d_I_sat, n_e,
            #               d_n_e, a, d_a, c_s, d_c_s, A_coll_0, d_A_coll))

            measured_vals[i] = [v_f, d_v_f, T_e, d_T_e, I_sat, d_I_sat, n_e, d_n_e, a, d_a, c_s, d_c_s, A_coll_0, d_A_coll,
                                chi2, red_chi2]

        measured_vals = np.array(measured_vals)

        plt.figure()
        for i, iv in enumerate(ivs):
            plt.plot(iv[0], iv[1], label=trim_lower[i])
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.legend()

        plt.figure()
        plt.errorbar(trim_lower, measured_vals[:, 2], yerr=measured_vals[:, 3])
        plt.xlabel('Lower_trim percentage')
        plt.ylabel('Measured Temperature (eV)')

        plt.figure()
        plt.errorbar(trim_lower, measured_vals[:, 0], yerr=measured_vals[:, 1])
        plt.xlabel('Lower_trim percentage')
        plt.ylabel('Measured Floating potential (V)')

        plt.figure()
        plt.errorbar(trim_lower, measured_vals[:, 6], yerr=measured_vals[:, 7])
        plt.xlabel('Lower_trim percentage')
        plt.ylabel(r'Measured density (m$^{-3}$)')

        plt.figure()
        # plt.plot(trim_lower, measured_vals[:, 14], label=r'$\chi^2$')
        plt.plot(trim_lower, measured_vals[:, 15], label=r'Reduced $\chi^2$')
        plt.axhline(y=1, linestyle='dashed', color='red')
        plt.xlabel('Lower_trim percentage')
        plt.legend()

    if trim_upper_fl:
        trim_upper = [1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76, 0.72, 0.68, 0.64]
        measured_vals = [[] for dummy in range(len(trim_upper))]
        ivs = [[] for dummy in range(len(trim_upper))]
        for i, tu in enumerate(trim_upper):
            # for tu in trim_upper:
            magopter.trim(trim_beg=0.0, trim_end=tu)
            fit_df_0, fit_df_1 = magopter.fit()
            iv_data = fit_df_0.iloc[[0]]

            ivs[i] = [iv_data[c.RAW_X].tolist()[0], iv_data[c.RAW_Y].tolist()[0]]

            # Extract individual values from dataframe
            v_f = iv_data[c.FLOAT_POT].values[0]
            T_e = iv_data[c.ELEC_TEMP].values[0]
            a = iv_data[c.SHEATH_EXP].values[0]
            I_sat = iv_data[c.ION_SAT].values[0]

            d_v_f = iv_data[c.ERROR_STRING.format(c.FLOAT_POT)].values[0]
            d_T_e = iv_data[c.ERROR_STRING.format(c.ELEC_TEMP)].values[0]
            d_a = iv_data[c.ERROR_STRING.format(c.SHEATH_EXP)].values[0]
            d_I_sat = iv_data[c.ERROR_STRING.format(c.ION_SAT)].values[0]

            c_s = lp.sound_speed(T_e, gamma_i=1)
            d_c_s = lp.d_sound_speed(c_s, T_e, d_T_e)
            n_e = lp.electron_density(I_sat, c_s, A_coll_0)
            d_n_e = lp.d_electron_density(n_e, c_s, d_c_s, A_coll_0, d_A_coll, I_sat, d_I_sat)

            # print('iv = {}: \n'
            #       '\t v_f = {:.3g} +- {:.1g} \n'
            #       '\t T_e = {:.3g} +- {:.1g} \n'
            #       '\t I_sat = {:.3g} +- {:.1g} \n'
            #       '\t n_e = {:.3g} +- {:.1g} \n'
            #       '\t a = {:.3g} +- {:.1g} \n'
            #       '\t c_s = {:.3g} +- {:.1g} \n'
            #       '\t A_coll = {:.3g} +- {:.1g} \n'
            #       .format(i, v_f, d_v_f, T_e, d_T_e, I_sat, d_I_sat, n_e,
            #               d_n_e, a, d_a, c_s, d_c_s, A_coll_0, d_A_coll))

            measured_vals[i] = [v_f, d_v_f, T_e, d_T_e, I_sat, d_I_sat, n_e, d_n_e, a, d_a, c_s, d_c_s, A_coll_0, d_A_coll]

        measured_vals = np.array(measured_vals)

        plt.figure()
        for i, iv in enumerate(ivs):
            plt.plot(iv[0], iv[1], label=trim_upper[i])
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.legend()

        plt.figure()
        plt.errorbar(trim_upper, measured_vals[:, 2], yerr=measured_vals[:, 3])
        plt.xlabel('Upper_trim percentage')
        plt.ylabel('Measured Temperature (eV)')

        plt.figure()
        plt.errorbar(trim_upper, measured_vals[:, 0], yerr=measured_vals[:, 1])
        plt.xlabel('Upper_trim percentage')
        plt.ylabel('Measured Floating potential (V)')

        plt.figure()
        plt.errorbar(trim_upper, measured_vals[:, 6], yerr=measured_vals[:, 7])
        plt.xlabel('Upper_trim percentage')
        plt.ylabel(r'Measured density (m$^{-3}$)')

    plt.show()


def multi_file_analysis(probe_0, folder, files):
    params = np.zeros([6, len(files)])
    for i, f in enumerate(files):
        # Run analysis for shot.
        dsr = 1
        m = Magopter(folder, f)
        m.prepare(down_sampling_rate=dsr, roi_b_plasma=True, crit_freq=4000, crit_ampl=1e-3)
        fit_df_0, fit_df_1 = m.fit()

        for j, data_tag in enumerate([mag.TARGET_CHAMBER_PRESSURE, mag.TARGET_TILT]):
            t, data = m.magnum_data[data_tag]
            if isinstance(data, np.ndarray):
                data = data.mean()
            if data_tag is mag.TARGET_TILT:
                data = data * (180/np.pi)
            params[j, i] = data

        T_e = fit_df_0[c.ELEC_TEMP].mean()
        d_T_e = fit_df_0[c.ELEC_TEMP].std() / np.sqrt(fit_df_0[c.ELEC_TEMP].count())
        I_sat = fit_df_0[c.ION_SAT].mean()
        d_I_sat = fit_df_0[c.ION_SAT].std() / np.sqrt(fit_df_0[c.ION_SAT].count())
        params[2:, i] = [T_e, d_T_e, I_sat, d_I_sat]

    fig, ax = plt.subplots()
    plt.errorbar(params[0], params[2], yerr=params[3], label='Temperature')
    plt.ylabel('Temperature')

    ax2 = ax.twinx()
    plt.errorbar(params[0], params[4], yerr=params[5], label=r'I$_{sat}$')
    plt.ylabel(r'I$_{sat}$')
    plt.xlabel('Target Chamber Pressure')
    plt.legend()


if __name__ == '__main__':
    folders = ['2018-05-01_Leland/', '2018-05-02_Leland/', '2018-05-03_Leland/',
               '2018-06-05_Leland/', '2018-06-06_Leland/', '2018-06-07_Leland/']
    files = []
    file_folders = []
    for folder1 in folders:
        os.chdir(Magopter.get_data_path() + folder1)
        files.extend(glob.glob('*.adc'))
        file_folders.extend([folder1] * len(glob.glob('*.adc')))

    files.sort()
    for i, file in enumerate(files):
        print('{}:    {}'.format(i, file))

    # file = files[286]
    file = files[285]
    ts_file = files[284]
    folder = file_folders[-2]
    print(folder, file, ts_file)

    mp = MagnumProbes()

    # file = askopenfilename()

    # main_magopter_analysis()
    # integrated_analysis(mp.probe_s, mp.probe_c, folder, file)
    # ts_ir_comparison(mp.probe_s, mp.probe_c, folder, file, ts_file)
    # multi_file_analysis(mp.probe_s, folder, files[285:297])
    # deeper_iv_analysis(mp.probe_s, folder, file, plot_timeline_fl=False)
    multifit_trim_filter_analysis(mp.probe_s, folder, file)
    # multifit_trim_iv_analysis(mp.probe_s, folder, file)

    plt.show()

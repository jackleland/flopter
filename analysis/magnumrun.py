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
    # magopter.trim(trim_end=0.82)
    magopter.trim(trim_end=0.83)
    fit_df_0, fit_df_1 = magopter.fit()

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
    magopter.prepare(down_sampling_rate=dsr)
    magopter.trim(trim_end=0.83)
    fit_df_0, fit_df_1 = magopter.fit()

    theta_perp = np.radians(10)
    A_coll_0 = probe_coax_0.get_collection_area(theta_perp)
    A_coll_1 = probe_coax_1.get_collection_area(theta_perp)

    if magopter.ts_temp is not None:
        T_e = np.mean([np.max(temp) / nrm.ELEM_CHARGE for temp in magopter.ts_temp[mag.DATA]])
        n_e = np.mean([np.max(dens) for dens in magopter.ts_dens[mag.DATA]])
        print('T = {}, n = {}'.format(T_e, n_e))
    else:
        T_e = 1.61
        n_e = 1.41e20
        fwhm = 12.4

    # t_0 = -0.35
    t_0 = 0
    target_pos_t = np.array(magopter.magnum_data[mag.TARGET_POS][0])
    target_pos_x = magopter.magnum_data[mag.TARGET_POS][1]

    target_voltage_t = np.array(magopter.magnum_data[mag.TARGET_VOLTAGE][0])
    target_voltage_x = np.array(magopter.magnum_data[mag.TARGET_VOLTAGE][1])

    deg_freedom = 3
    gamma_i = (deg_freedom + 2) / 2
    c_s = np.sqrt((nrm.ELEM_CHARGE * (T_e + gamma_i * T_e)) / nrm.PROTON_MASS)
    n_e_0 = fit_df_0[c.ION_SAT] / (nrm.ELEM_CHARGE * c_s * A_coll_0)
    n_e_1 = fit_df_1[c.ION_SAT] / (nrm.ELEM_CHARGE * c_s * A_coll_1)

    I_sat_0 = c_s * n_e * nrm.ELEM_CHARGE * A_coll_0
    I_sat_1 = c_s * n_e * nrm.ELEM_CHARGE * A_coll_1

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
            # if magopter.magnum_data[mag.BEAM_DUMP_UP][1][i]:
            #     plt.axvline(x=magopter.magnum_data[mag.BEAM_DUMP_UP][0][i], color='b', linestyle='--', linewidth=1)

        # for j in range(np.shape(magopter.magnum_data[mag.PLASMA_STATE])[1]):
        #     plt.axvline(x=magopter.magnum_data[mag.PLASMA_STATE][0][j], color='g', linestyle='--', linewidth=1)

        # for k in range(np.shape(magopter.magnum_data[mag.TRIGGER_START])[1]):
        #     plt.axvline(x=magopter.magnum_data[mag.TRIGGER_START][0][k], color='b', linestyle='--', linewidth=1)
        #     print(magopter.magnum_data[mag.TRIGGER_START][0][k])

        if magopter.ts_temp is not None:
            for k in range(len(magopter.ts_temp[0])):
                plt.axvline(x=magopter.ts_temp[0][k], color='m', linestyle='--', linewidth=1, label='TS')

        plt.plot(target_pos_t, target_pos_x, color='k', label='Target Position')
        # plt.axvline(x=0, color='k', linewidth=1, linestyle='-.')
        plt.xlabel('Time (s)')
        plt.legend()

    #########################################
    #            Whole IV plot              #
    #########################################
    iv_data = fit_df_0.iloc[[10]]
    fig, ax = plt.subplots()
    max_currents = [[], []]
    for iv_curve in magopter.iv_arr_coax_0:
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
    plt.axhline(y=I_sat_0, linestyle='dashed', linewidth=1, color='r', label='Expected I_sat (s)')

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

    file = files[-2]
    ts_file = files[-1]
    folder = file_folders[-2]
    print(folder, file, ts_file)

    mp = MagnumProbes()

    # main_magopter_analysis()
    # integrated_analysis(mp.probe_s, mp.probe_c, folder, file, ts_file=ts_file)
    ts_ir_comparison(mp.probe_s, mp.probe_c, folder, file, ts_file)

    plt.show()

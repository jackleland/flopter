import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import json
import os
import glob
# sys.path.append('/home/jleland/Coding/Projects/flopter')
import flopter.magnum.magoptoffline as mg
import flopter.core.lputils as lp
import flopter.core.ivdata as ivd
import flopter.core.fitters as fts
from tkinter.filedialog import askopenfilename


FOLDERS = ('2019-05-28_Leland/',
           '2019-05-29_Leland/',
           '2019-06-03_Leland/',
           '2019-06-04_Leland/',
           '2019-06-05_Leland/',
           '2019-06-06_Leland/',
           '2019-06-07_Leland/',
           )
PROBE_DESIGNATIONS = ('S', 'L')
SWEEP_RANGE = (0, 750)


def averaged_iv_analysis(filename=None, ts_temp=None, ts_dens=None, shunt_resistance=10.0, theta_perp=10.0,
                         probe_designations=PROBE_DESIGNATIONS, sweep_range=SWEEP_RANGE, downsamplnig_factor=1):
    if filename is None:
        # folders = ['2019-05-28_Leland/', '2019-05-29_Leland/']
        mg.Magoptoffline._FOLDER_STRUCTURE = '/Data/external/magnum/'
        files = []
        file_folders = []
        for folder1 in FOLDERS:
            os.chdir(mg.Magoptoffline.get_data_path() + folder1)
            files.extend(glob.glob('*.adc'))
            file_folders.extend([folder1] * len(glob.glob('*.adc')))
        files.sort()

        # for i, f in enumerate(files):
        #     print(i, f)

        # file = files[286]
        # adc_file = files[285]
        # ts_file = files[284]
        adc_file = files[-1]
        folder = FOLDERS[-1]
    else:
        # If using the tkinter file chooser
        adc_file = filename.split('/')[-1]
        folder = filename.split('/')[-2] + '/'
        mg.Magoptoffline._FOLDER_STRUCTURE = '/Data/external/magnum/'

    print('"{}" \t\t "{}"'.format(folder, adc_file))

    mp = lp.MagnumProbes()
    probe_S = mp.probe_s
    probe_B = mp.probe_b

    dsr = downsamplnig_factor

    # Create magopter object
    print('Creating magopter object')
    magopter = mg.Magoptoffline(folder, adc_file, shunt_resistor=shunt_resistance, cabling_resistance=2)
    magopter._VOLTAGE_CHANNEL = 3
    magopter._PROBE_CHANNEL_3 = 4
    magopter._PROBE_CHANNEL_4 = 5
    magopter.prepare(down_sampling_rate=dsr, roi_b_plasma=True, filter_arcs_fl=False, crit_freq=None, crit_ampl=None)

    print('0: {}, 1: {}'.format(len(magopter.iv_arrs[0]), len(magopter.iv_arrs[1])))

    if ts_dens is not None and ts_temp is not None:
        T_e_ts = ts_temp
        d_T_e_ts = ts_temp * 0.01
        n_e_ts = ts_dens
        d_n_e_ts = ts_dens * 0.01
    else:
        T_e_ts = 1.12
        d_T_e_ts = 0.01
        n_e_ts = 4.44e20
        d_n_e_ts = 0.01e20

    # print length of the 0th probes (probe S) number of sweeps
    print(len(magopter.iv_arrs[1]))

    # Create relative t array by subtracting the first timestep value from the first time array
    first_time_arr = magopter.iv_arrs[1][0]['t']
    relative_t = np.zeros(len(first_time_arr))

    sweep_length = np.shape(relative_t)[0] // 2
    print('Sweep length is {}'.format(sweep_length))

    relative_t = first_time_arr - first_time_arr[0]

    # create a list of datasets for each sweep
    ds_probes = []

    for i in range(len(magopter.iv_arrs)):
        ds_list = []
        for j, iv in enumerate(magopter.iv_arrs[i]):
            if j % 2 == 0:
                ds = xr.Dataset({'voltage': (['time'], iv['V'][:sweep_length]),
                                 'current': (['time'], iv['I'][:sweep_length]),
                                 'shot_time': (['time'], iv['t'][:sweep_length]),
                                 'start_time': iv['t'][0]},
                                coords={'time': relative_t[:sweep_length], 'direction': 'up',
                                        'probe': probe_designations[i]})
            else:
                ds = xr.Dataset({'voltage': (['time'], np.flip(iv['V'][:sweep_length])),
                                 'current': (['time'], np.flip(iv['I'][:sweep_length])),
                                 'shot_time': (['time'], np.flip(iv['t'][:sweep_length])),
                                 'start_time': iv['t'][0]},
                                coords={'time': relative_t[:sweep_length], 'direction': 'down',
                                        'probe': probe_designations[i]})
            ds_list.append(ds)

        # # Separate into up and down sweeps then concat along sweep direction as an axis
        print('Before equalisation: ', len(ds_list), len(ds_list[::2]), len(ds_list[1::2]))
        if len(ds_list[::2]) == len(ds_list[1::2]) + 1:
            ds_ups = xr.concat(ds_list[:-2:2], 'sweep')
        else:
            ds_ups = xr.concat(ds_list[::2], 'sweep')
        ds_downs = xr.concat(ds_list[1::2], 'sweep')
        print('After equalisation: ', len(ds_ups['sweep']), len(ds_downs['sweep']))

        direction = xr.DataArray(np.array(['up', 'down']), dims=['direction'], name='direction')
        ds_probes.append(xr.concat([ds_ups, ds_downs], dim=direction))

    probe = xr.DataArray(np.array(probe_designations), dims=['probe'], name='probe')
    ds_full = xr.concat(ds_probes, dim=probe)

    # Select the small probe
    ds_full = ds_full.sel(probe=probe_designations[0])

    manual_start = sweep_range[0]
    manual_end = sweep_range[1]
    plt.figure()
    ds_full.max(dim='time').mean('direction')['current'].plot.line(x='sweep')
    ds_full.max(dim='time').mean('direction').isel(sweep=slice(manual_start, manual_end))['current'].plot.line(
        x='sweep')

    # Choose only the IVs in the static section
    ds_full = ds_full.isel(sweep=slice(manual_start, manual_end))

    # Average across each sweep direction
    sweep_avg_up = ds_full.sel(direction='up').mean('sweep')
    sweep_avg_dn = ds_full.sel(direction='down').mean('sweep')

    # Add in standard deviation of each bin as a new data variable
    sweep_avg_up = sweep_avg_up.assign({'d_current': ds_full.sel(direction='up').std('sweep')['current']})
    sweep_avg_dn = sweep_avg_dn.assign({'d_current': ds_full.sel(direction='down').std('sweep')['current']})

    print(ds_full)

    sweep_avg_updn = ds_full.mean('direction').mean('sweep').assign(
        {'d_current': ds_full.std('direction').std('sweep')['current']})
    sweep_avg_updn = sweep_avg_updn.where(sweep_avg_updn.current <= 0, drop=True)
    print(sweep_avg_updn)

    # sweep_avg_updn['current'].plot.line()

    # concatenate the up and down sweeps together to cancel the (small) capacitance effect
    iv_data = ivd.IVData(sweep_avg_updn['voltage'].data,
                         -sweep_avg_updn['current'].data,
                         sweep_avg_updn['time'].data,
                         sigma=sweep_avg_updn['d_current'].data, estimate_error_fl=False)

    starting_params = [0.69, 0.009, 1.12, +1]

    full_iv_fitter = fts.FullIVFitter()
    fit_data = full_iv_fitter.fit_iv_data(iv_data, initial_vals=starting_params)
    fig = plt.figure()
    fit_data.plot(fig=fig, show_fl=False)
    # plt.errorbar(fit_data.raw_x, fit_data.raw_y, yerr=iv_data['sigma'], ecolor='silver')
    # plt.plot(fit_data.raw_x, fit_data.fit_y, color='orange', label=r'')
    plt.plot(iv_data['V'], full_iv_fitter.fit_function(iv_data['V'], *starting_params), label='Start-param IV')
    plt.legend()
    plt.ylim([-.25, 1])

    # Create new averaged iv figure
    theta_perp = np.radians(theta_perp)

    # probe_selected = probe_L

    A_coll_0 = probe_S.get_collection_area(theta_perp)
    d_A_coll = np.abs(probe_S.get_collection_area(theta_perp + np.radians(0.8)) - A_coll_0)

    v_f_fitted = fit_data.get_param('V_f')
    d_v_f_fitted = fit_data.get_param('V_f', errors_fl=True).error

    v_f_approx = - 3 * fit_data.get_temp()
    d_v_f_approx = 0.05 * v_f_approx

    v_f_approx_ts = - 3 * T_e_ts
    d_v_f_approx_ts = 0.05 * v_f_approx_ts

    c_s_fitted = lp.sound_speed(fit_data.get_temp(), gamma_i=1)
    d_c_s_fitted = lp.d_sound_speed(c_s_fitted, fit_data.get_temp(), fit_data.get_temp(errors_fl=True).error)
    n_e_fitted = lp.electron_density(fit_data.get_isat(), c_s_fitted, A_coll_0)
    d_n_e_fitted = lp.d_electron_density(n_e_fitted, c_s_fitted, d_c_s_fitted, A_coll_0, d_A_coll, fit_data.get_isat(),
                                         fit_data.get_isat(errors_fl=True).error)

    print("iv = averaged: \n"
          "\t v_f = {:.3g} +- {:.1g} \n"
          "\t T_e = {:.3g} +- {:.1g} \n"
          "\t I_sat = {:.3g} +- {:.1g} \n"
          "\t n_e = {:.3g} +- {:.1g} \n"
          "\t a = {:.3g} +- {:.1g} \n"
          "\t c_s = {:.3g} +- {:.1g} \n"
          "\t A_coll = {:.3g} +- {:.1g} \n"
          .format(v_f_fitted, d_v_f_fitted,
                  fit_data.get_temp(), fit_data.get_temp(errors_fl=True).error,
                  fit_data.get_isat(), fit_data.get_isat(errors_fl=True).error,
                  n_e_fitted, d_n_e_fitted,
                  fit_data.get_sheath_exp(), fit_data.get_sheath_exp(errors_fl=True).error,
                  c_s_fitted, d_c_s_fitted,
                  A_coll_0, d_A_coll))

    I_f = probe_S.get_analytical_iv(fit_data.raw_x, v_f_fitted, theta_perp, fit_data.get_temp(), n_e_fitted,
                                    print_fl=True)
    I_ts = probe_S.get_analytical_iv(fit_data.raw_x, v_f_approx_ts, theta_perp, T_e_ts, n_e_ts,
                                     print_fl=True)

    plt.figure()
    plt.errorbar(fit_data.raw_x, fit_data.raw_y, yerr=fit_data.sigma,
                 label='Raw IV', ecolor='silver', color='gray', zorder=-1)
    # plt.plot(iv_data[c.RAW_X].tolist()[0], I_f, label='Analytical - measured', linestyle='dashed', linewidth=1, color='r')
    plt.plot(fit_data.raw_x, fit_data.fit_y, color='blue', linewidth=1.2,
             label='Fit - ({:.2g}eV, {:.2g}m'.format(fit_data.get_temp(), n_e_fitted) + r'$^{-3}$)')
    plt.plot(fit_data.raw_x, I_ts, linestyle='dashed', color='red',
             label='Analytical from TS - ({:.2g}eV, {:.2g}m'.format(T_e_ts, n_e_ts) + '$^{-3}$)')

    plt.legend()
    # plt.title('Comparison of analytical to measured IV curves for the small area probe')
    plt.xlabel(r'$V_p$ / V')
    plt.ylabel(r'$I$ / A')
    # plt.ylim([-0.01, 3.2])
    plt.show()


if __name__ == '__main__':
    with open('config.json', 'r') as fp:
        options = json.load(fp)
    filename = askopenfilename()
    print('ADC File: {} \n'
          'Options: {} \n'
          .format(filename, options))
    averaged_iv_analysis(filename, **options)

import numpy as np
import traceback
import matplotlib.pyplot as plt
import xarray as xr
import os
import concurrent.futures as cf
import flopter.core.magoptoffline as mg
import flopter.classes.ivdata as ivd
import flopter.core.fitters as fts


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


def averaged_iv_analysis(folder, adc_file, output_tag, ts_temp=None, ts_dens=None, probe_designations=PROBE_DESIGNATIONS,
                         shunt_resistance=10, cabling_resistance=2.0, sweep_range=SWEEP_RANGE, downsampling_factor=1):

    # mg.Magoptoffline._FOLDER_STRUCTURE = '/Data/external/magnum/'
    mg.Magoptoffline._FOLDER_STRUCTURE = '/Data/Magnum/adc_files/'
    print('"{}" \t\t "{}"'.format(folder, adc_file))

    dsr = downsampling_factor

    # Create magopter object
    print('Creating magopter object')
    magopter = mg.Magoptoffline(folder, adc_file, shunt_resistor=shunt_resistance, cabling_resistance=cabling_resistance)
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

    # print length of the 0th probe's (probe S) number of sweeps
    print(len(magopter.iv_arrs[1]))

    # Create relative t array by subtracting the first timestep value from the first time array
    first_time_arr = magopter.iv_arrs[1][0]['t']
    second_time_arr = magopter.iv_arrs[0][0]['t']
    if len(first_time_arr) > len(second_time_arr):
        first_time_arr = second_time_arr

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

        # Separate into up and down sweeps then concat along sweep direction as an axis
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
    min_sweep_number = np.min([len(ds_probes[0]['sweep']), len(ds_probes[1]['sweep'])])

    ds_probes[0] = ds_probes[0].sel(sweep=slice(0, min_sweep_number))
    ds_probes[1] = ds_probes[1].sel(sweep=slice(0, min_sweep_number))

    ds_full = xr.concat(ds_probes, dim=probe)

    cwd = os.getcwd()
    os.chdir(mg.Magoptoffline.get_data_path() + 'analysed_1/')
    ds_full.to_netcdf(f'{output_tag}.nc')

    # Select the small probe
    ds_full = ds_full.sel(probe=probe_designations[0])

    manual_start = sweep_range[0]
    manual_end = sweep_range[1]
    plt.figure()
    ds_full.max(dim='time').mean('direction')['current'].plot.line(x='sweep')
    ds_full.max(dim='time').mean('direction').isel(sweep=slice(manual_start, manual_end))['current'].plot.line(
        x='sweep')
    plt.savefig(f'{output_tag}_shot.png', bbox_inches='tight')

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

    starting_params = [0.69, 0.009, 1.12, 1]

    full_iv_fitter = fts.FullIVFitter()
    fit_data = full_iv_fitter.fit_iv_data(iv_data, initial_vals=starting_params)

    fig = plt.figure()
    fit_data.plot(fig=fig, show_fl=False)
    # plt.errorbar(fit_data.raw_x, fit_data.raw_y, yerr=iv_data['sigma'], ecolor='silver')
    # plt.plot(fit_data.raw_x, fit_data.fit_y, color='orange', label=r'')
    plt.plot(iv_data['V'], full_iv_fitter.fit_function(iv_data['V'], *starting_params), label='Start-param IV')
    plt.legend()
    plt.ylim([-.25, 2.0])

    plt.tight_layout()
    plt.savefig(f'{output_tag}_fit.png', bbox_inches='tight')

    os.chdir(cwd)

    del magopter, ds_full, ds_downs, ds_ups, ds_probes, ds_list, sweep_avg_up, sweep_avg_dn, sweep_avg_updn
    import gc
    gc.collect()


# os.chdir('/home/jleland/Data/external/magnum/')
os.chdir('/home/jleland/Data/Magnum/adc_files/')
all_dataset = xr.open_dataset('all_meta_data.nc').max('ts_radial_pos')
shot_numbers = all_dataset.where(np.isfinite(all_dataset['adc_index']), drop=True)['shot_number'].values
shot_dataset = all_dataset.sel(shot_number=shot_numbers)

CABLE_RESISTANCES = [1.7, 1.7, 1.7, 1.6]
PROBE_RESISTANCES = {
    'S': 1.0,
    'L': 1.0,
    'B': 1.0,
    'R': 1.8
}
DESIRED_DATARATE = 10000
COMMONLY_USED_SWEEP_TIME = 0.01


def get_sweep_range(shot_end_time, adc_end_time, acq_length, adc_freq):
    end_index = int(acq_length * adc_freq * COMMONLY_USED_SWEEP_TIME)
    if shot_end_time < adc_end_time:
        end_index = int((acq_length - ((shot_end_time - adc_end_time) / np.timedelta64(1, 's')) - 2)
                        * adc_freq * COMMONLY_USED_SWEEP_TIME)
    return 1, end_index


def aia_mapping_wrapper(shot_number):
    print(f'\n Analysing shot {shot_number}...')
    
    try:
        # print('Try statement')
        shot_dataarray = shot_dataset.sel(shot_number=shot_number)
        
        folder = str(shot_dataarray['adc_folder'].values)
        adc_file = str(shot_dataarray['adc_filename'].values)
        output_tag = 'a{:03d}_{:03d}_{}'.format(shot_number,
                                                int(shot_dataarray['adc_index'].values),
                                                int(shot_dataarray['adc_timestamp'].values))
        ts_temp = shot_dataarray['ts_temp_max'].values
        ts_dens = shot_dataarray['ts_dens_max'].values
        probe_designations = (str(shot_dataarray['adc_4_probe'].values), str(shot_dataarray['adc_5_probe'].values))
        shunt_resistance = shot_dataarray['adc_4_shunt_resistance'].values
        # downsampling_factor = int(shot_dataarray['adc_freqs'].values / DESIRED_DATARATE)
        downsampling_factor = 1
        cabling_resistance = (CABLE_RESISTANCES[int(shot_dataarray['adc_4_coax'].values) - 1] + 1.2
                              + PROBE_RESISTANCES[probe_designations[0]])
        sweep_range = get_sweep_range(shot_dataarray['shot_end_time'].values, shot_dataarray['adc_end_time'].values,
                                      shot_dataarray['acquisition_length'].values,
                                      shot_dataarray['adc_freqs'].values / downsampling_factor)

        print(f'Attempting analysis on shot {shot_number}')
    
        averaged_iv_analysis(folder, adc_file, output_tag, ts_temp=ts_temp, ts_dens=ts_dens,
                             probe_designations=probe_designations, shunt_resistance=shunt_resistance,
                             cabling_resistance=cabling_resistance, sweep_range=sweep_range,
                             downsampling_factor=downsampling_factor)
    except:
        traceback.print_exc()
    print(f'\n ...Finished shot {shot_number}')


def multi_file_analysis(shots):
    print('\nRunning multi-file analysis. Analysing {} shot(s).\n'.format(len(shots)))

    # Execute fitting and saving of files concurrently
    with cf.ProcessPoolExecutor() as executor:
        executor.map(aia_mapping_wrapper, shots)


if __name__ == '__main__':
    multi_file_analysis(shot_numbers)
    # aia_mapping_wrapper(shot_numbers[0])
    # aia_mapping_wrapper(157)

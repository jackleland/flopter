import numpy as np
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import os
import concurrent.futures as cf
import flopter.magnum.magoptoffline as mg
import flopter.core.ivdata as ivd
import flopter.core.fitters as fts


FOLDERS = (
    '2019-05-28_Leland/',
    '2019-05-29_Leland/',
    '2019-06-03_Leland/',
    '2019-06-04_Leland/',
    '2019-06-05_Leland/',
    '2019-06-06_Leland/',
    '2019-06-07_Leland/',
)
PROBE_DESIGNATIONS = ('S', 'L')
SWEEP_RANGE = (0, 750)

# OUTPUT_DIRECTORY = 'analysed_4_downsampled/'
# DOWNSAMPLING_FL = True

OUTPUT_DIRECTORY = 'analysed_4/'
DOWNSAMPLING_FL = False

DATA_DIRECTORY = '/Data/Magnum/adc_files/'
ON_FREIA_FLAG = True
MAX_CPUS = 32
if not os.path.exists(DATA_DIRECTORY):
    MAX_CPUS = 2
    ON_FREIA_FLAG = False
    DATA_DIRECTORY = '/data/external/magnum/'
    OUTPUT_DIRECTORY = 'test'
    DOWNSAMPLING_FL = True


def averaged_iv_analysis(folder, adc_file, output_tag, probe_designations=PROBE_DESIGNATIONS, shunt_resistance=10,
                         cabling_resistance=(2.0, 2.0), downsampling_factor=1, dealloc=True):

    mg.Magoptoffline._FOLDER_STRUCTURE = DATA_DIRECTORY
    print('"{}" \t\t "{}"'.format(folder, adc_file))

    dsr = downsampling_factor

    # Create magopter object
    print('Creating magopter object')
    magopter = mg.Magoptoffline(folder, adc_file, shunt_resistor=shunt_resistance,
                                cabling_resistance=cabling_resistance)
    magopter._VOLTAGE_CHANNEL = 3
    magopter._PROBE_CHANNEL_3 = 4
    magopter._PROBE_CHANNEL_4 = 5
    magopter.prepare(down_sampling_rate=dsr, roi_b_plasma=True, filter_arcs_fl=False, crit_freq=45000, crit_ampl=None)

    print('0: {}, 1: {}'.format(len(magopter.iv_arrs[0]), len(magopter.iv_arrs[1])))

    ds_full = magopter.to_xarray(probe_designations)

    cwd = os.getcwd()
    os.chdir(mg.Magoptoffline.get_data_path() + OUTPUT_DIRECTORY)
    ds_full.to_netcdf(f'{output_tag}.nc')

    os.chdir(cwd)

    if dealloc:
        del magopter, ds_full
        import gc
        gc.collect()
    else:
        return magopter, ds_full


def analyse_small_probe(ds_full, probe_designation, output_tag, sweep_range):
    # Select the small probe
    ds_full = ds_full.sel(probe=probe_designation)

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

    del sweep_avg_up, sweep_avg_dn, sweep_avg_updn
    import gc
    gc.collect()


os.chdir('/home/jleland/' + DATA_DIRECTORY)
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
FEEDTHROUGH_RESISTANCE = 1.25
INTERNAL_RESISTANCE = 6.09
DESIRED_DATARATE = 1000000
COMMONLY_USED_SWEEP_TIME = 0.01


def get_sweep_range(shot_end_time, adc_end_time, acq_length, adc_freq):
    end_index = int(acq_length * adc_freq * COMMONLY_USED_SWEEP_TIME)
    if shot_end_time < adc_end_time:
        end_index = int((acq_length - ((shot_end_time - adc_end_time) / np.timedelta64(1, 's')) - 2)
                        * adc_freq * COMMONLY_USED_SWEEP_TIME)
    return 1, end_index


def get_shot_info_for_analysis(shot_number):
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

    if DOWNSAMPLING_FL:
        downsampling_factor = int(shot_dataarray['adc_freqs'].values / DESIRED_DATARATE)
    else:
        downsampling_factor = 1

    cabling_resistance = (
        (CABLE_RESISTANCES[int(shot_dataarray['adc_4_coax'].values) - 1] + FEEDTHROUGH_RESISTANCE
         + INTERNAL_RESISTANCE + PROBE_RESISTANCES[probe_designations[0]]),
        (CABLE_RESISTANCES[int(shot_dataarray['adc_5_coax'].values) - 1] + FEEDTHROUGH_RESISTANCE
         + INTERNAL_RESISTANCE + PROBE_RESISTANCES[probe_designations[1]])
    )

    sweep_range = get_sweep_range(shot_dataarray['shot_end_time'].values, shot_dataarray['adc_end_time'].values,
                                  shot_dataarray['acquisition_length'].values,
                                  shot_dataarray['adc_freqs'].values / downsampling_factor)
    return (folder, adc_file, output_tag, ts_temp, ts_dens, probe_designations, shunt_resistance, downsampling_factor,
            cabling_resistance, sweep_range)


def aia_mapping_wrapper(shot_number, dsr=None):
    print(f'\n Analysing shot {shot_number}...')
    
    try:
        # print('Try statement')

        print(f'Attempting analysis on shot {shot_number}')

        folder, adc_file, output_tag, ts_temp, ts_dens, probe_designations, shunt_resistance, downsampling_factor, \
        cabling_resistance, sweep_range = get_shot_info_for_analysis(shot_number)

        if dsr is not None:
            downsampling_factor = dsr
    
        averaged_iv_analysis(folder, adc_file, output_tag, probe_designations=probe_designations,
                             shunt_resistance=shunt_resistance, cabling_resistance=cabling_resistance,
                             downsampling_factor=downsampling_factor)
    except:
        print(f' *** SHOT {shot_number} FAILED *** ')
        traceback.print_exc()
    print(f'\n ...Finished shot {shot_number}')


def multi_file_analysis(shots):
    print('\nRunning multi-file analysis. Analysing {} shot(s).\n'.format(len(shots)))

    # Execute fitting and saving of files concurrently
    with cf.ProcessPoolExecutor(max_workers=MAX_CPUS) as executor:
        executor.map(aia_mapping_wrapper, shots)


if __name__ == '__main__':
    test_shots = [215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 395, 399, 400, 401, 402, 404, 405, 406, 407, 409, 410, 411, 412, 413, 414, 415, 416, 423, 424, 432, 433, 434, 435, 436, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451]
    multi_file_analysis(test_shots)
    # multi_file_analysis(shot_numbers)
    # aia_mapping_wrapper(shot_numbers[0])
    # aia_mapping_wrapper(157)

import glob
import re
import xarray as xr
import pandas as pd
import flopter.magnum.database as ut
import numpy as np
import matplotlib.pyplot as plt
import flopter.core.lputils as lpu
import flopter.core.ivdata as iv
import flopter.core.fitters as fts
import flopter.core.constants as c
import matplotlib as mpl
import collections


shot_metadata_ds = xr.open_dataset('/home/jleland/data/external/magnum/all_meta_data.nc')
magnum_probes = lpu.MagnumProbes()


def create_analysed_ds_metadata(path):
    analysed_dataset_fns = glob.glob(path + '/*.nc')
    analysed_dataset_fns.sort()

    analysed_infos = []

    for i, anal_ds in enumerate(analysed_dataset_fns):
        match = re.search("\/a{1}([0-9]{3})_([0-9]{3})_([0-9]{19})\.nc", anal_ds)
        shot_index = int(match.group(1))
        adc_index = int(match.group(2))
        shot_timestamp = int(match.group(3))
        shot_time = ut.human_datetime_str(int(match.group(3)))

        ds = xr.open_dataset(anal_ds)
        time_len = len(ds['time'])
        sweep_len = len(ds['sweep'])

        analysed_infos.append({
            'adc_index': adc_index,
            'shot_number': shot_index,
            'shot_timestamp': shot_timestamp,
            'shot_time': shot_time,
            'filename': anal_ds,
            'time_len': time_len,
            'sweep_len': sweep_len
        })

    return pd.DataFrame(analysed_infos).set_index('adc_index')


def get_dataset_metadata(path_to_analysed_datasets):
    metadata_filename = f'{path_to_analysed_datasets}_xr_metadata.csv'

    # Loading the metadata .csv file into a pandas dataframe
    try:
        analysed_infos_df = pd.read_csv(metadata_filename).set_index('adc_index')
    except FileNotFoundError:
        # If the .csv file doesn't exist, create it and save it to where we would expect it to be
        analysed_infos_df = create_analysed_ds_metadata(path_to_analysed_datasets)
        analysed_infos_df.to_csv(metadata_filename)

    return analysed_infos_df


def preprocess_sel(ds, sweep=slice(0, 997)):
    ds = ds.reset_index('time', drop=True).load()
    return ds.sel(sweep=sweep)


def preprocess_autosel(ds):
    ds = ds.reset_index('time', drop=True).load()
    sweep = slice(*find_sweep_limit(ds))
    return ds.sel(sweep=sweep)


# def preprocess_average(ds):
#     ds = ds.reset_index('time', drop=True).load()
#     sweep_min, sweep_max = find_sweep_limit(ds)
#
#     ds_avg = ds.sel(sweep=slice(sweep_min, sweep_max)).mean('sweep')
#     ds_avg = ds_avg.assign({'d_current': ds.std('sweep')['current']})
#     return ds_avg

def preprocess_average(ds, dims_to_avg=('sweep', 'direction'), sweep_lims_fl=True):
    ds = ds.reset_index('time', drop=True).load()

    if sweep_lims_fl:
        sweep_min, sweep_max = find_sweep_limit(ds)
        ds_trimmed = ds.sel(sweep=slice(sweep_min, sweep_max))
    else:
        ds_trimmed = ds.sel(sweep=find_allowed_sweeps(ds))

    ds_avg = ds_trimmed.mean(dims_to_avg)
    ds_avg = ds_avg.assign({
        'd_current': (ds_trimmed.std(dims_to_avg)['current']
                      / np.sqrt(np.prod([ds_trimmed[dim].size for dim in dims_to_avg]))),
        'std_current': ds_trimmed.std(dims_to_avg)['current']
    })
    return ds_avg


def preprocess_average_downsample(ds, downsample_to=500, dims_to_avg=('sweep', 'direction')):
    ds = preprocess_average(ds, dims_to_avg=dims_to_avg)
    dsf = int(len(ds.time) / downsample_to)
    ds = ds.sel(time=slice(0, len(ds.time), dsf))
    return ds


def preprocess_average_coarsen(ds, coarsen_to=500, dims_to_avg=('sweep', 'direction')):
    ds = preprocess_average(ds, dims_to_avg=dims_to_avg)
    cf = int(len(ds.time) / coarsen_to)
    ds = ds.coarsen(time=cf).mean()
    return ds


def preprocess_average_downsample_old(ds, downsample_to=500):
    ds = ds.reset_index('time', drop=True).load()
    dsf = int(len(ds.time) / downsample_to)
    ds = ds.sel(time=slice(0, len(ds.time), dsf))
    ds_avg = ds.mean('sweep')
    ds_avg = ds_avg.assign({'d_current': ds.std('sweep')['current']})
    return ds_avg


def find_sweep_limit(ds, probe=None):
    if probe is None:
        probe = 'S' if 'S' in ds.probe.values else 'B'
    max_current = ds['current'].mean('direction').max('time').sel(probe=probe)

    max_current_trim = max_current.where(exclude_outliers_cond(max_current, 1.5), drop=True)
    max_current_trim_2 = max_current_trim.where(exclude_outliers_cond(max_current_trim, 2.5), drop=True)

    return max_current_trim_2["sweep"].min().values, max_current_trim_2["sweep"].max().values


def find_allowed_sweeps(ds, min_fl=True, probes=None):
    """
    This is a tweak to 'find_sweep_limit' to filter out arcs as well as the
    end of the shot.
    """
    if probes is None:
        probes = ds.probe.values

    allowed_sweeps = None

    for probe in probes:
        if min_fl:
            sweep_current = ds['current'].mean('direction').min('time').sel(probe=probe)
        else:
            sweep_current = ds['current'].mean('direction').max('time').sel(probe=probe)

        # Create entirely True array of same length = n_sweeps
        if allowed_sweeps is None:
            allowed_sweeps = np.full(sweep_current.size, True)

        current_filter_1 = sweep_current.where(exclude_outliers_cond(sweep_current, 1.5))
        current_filter_2 = current_filter_1.where(exclude_outliers_cond(current_filter_1, 2.5))

        allowed_sweeps = np.logical_and(allowed_sweeps, np.isfinite(current_filter_2))

    return allowed_sweeps


def exclude_outliers_cond(da, n=2.0):
    return np.abs(da - da.median()) < n * da.std()


def get_dataset_from_indices(indices, parallel=True, preprocess='average', load_fl=True, anglescan_fl=True,
                             average_direction_fl=True, path_to_analysed_datasets='analysed_2',
                             check_sweep_trim_fl=False):
    # Create a dataframe of relevant metadata about the analysed datasets
    analysed_metadata_df = get_dataset_metadata(path_to_analysed_datasets)
    analysed_metadata_oi = analysed_metadata_df.loc[indices]

    # Generate a selection of useful arrays and DataArrays from the relevant metadata
    files_oi = analysed_metadata_oi['filename'].values
    shot_numbers = analysed_metadata_oi['shot_number'].values
    shot_numbers_da = xr.DataArray(shot_numbers, dims=['shot_number'], name='shot_number')
    min_sweep_len = analysed_metadata_oi['sweep_len'].min()
    min_time_len = analysed_metadata_oi['time_len'].min()

    if check_sweep_trim_fl:
        print('Plotting sweep trimming for all shots and probes.')
        for filename in files_oi:
            downsampling = 100 if 'downsampled' not in path_to_analysed_datasets else 1
            plot_shot_trimming(filename, downsample=downsampling)

    # Select meta data for shots of interest
    shot_metadata_oi = shot_metadata_ds.sel(shot_number=shot_numbers)

    # Construct an additional DataArray for the probe tilt - required for anglescan
    tilt_da = shot_metadata_oi[['shot_tilt', 'adc_4_probe', 'adc_5_probe']]
    tilt_da['shot_tilt'] = tilt_da.shot_tilt.round()

    if preprocess == 'average':
        combined_ds = xr.open_mfdataset(files_oi, concat_dim=shot_numbers_da, preprocess=preprocess_average,
                                        parallel=parallel, combine='nested')
    elif preprocess == 'average_downsample':
        combined_ds = xr.open_mfdataset(files_oi, concat_dim=shot_numbers_da, preprocess=preprocess_average_downsample,
                                        parallel=parallel, combine='nested')
    elif preprocess == 'average_coarsen':
        combined_ds = xr.open_mfdataset(files_oi, concat_dim=shot_numbers_da, parallel=parallel, combine='nested',
                                        preprocess=lambda x: preprocess_average_coarsen(x, min_time_len))
    elif preprocess == 'autosel':
        combined_ds = xr.open_mfdataset(files_oi, concat_dim=shot_numbers_da, preprocess=preprocess_autosel,
                                        parallel=parallel, combine='nested')
    elif preprocess == 'sel':
        combined_ds = xr.open_mfdataset(files_oi, concat_dim=shot_numbers_da, parallel=parallel, combine='nested',
                                        preprocess=lambda x: preprocess_sel(x, slice(0, min_sweep_len)))
    else:
        raise ValueError('Invalid preprocess function specified.')

    if load_fl:
        combined_ds = combined_ds.load()

    # Then merge it with the metadata for a complete dataset of the desired shots
    combined_ds = xr.merge([combined_ds, shot_metadata_oi.rename({'shot_time': 'sweep_time'})]).assign_coords(
        tilt=tilt_da['shot_tilt'])

    if anglescan_fl:
        anglescan_ds = combined_ds.swap_dims({'shot_number': 'tilt'}).reset_coords()

        # Reorganise to make tilt and probe dimensions
        probes_1 = anglescan_ds.sel(probe=['S', 'L']).groupby(
            'tilt').mean()  # .where(np.isfinite(anglescan_ds.voltage), drop=True)
        probes_2 = anglescan_ds.sel(probe=['R', 'B']).groupby(
            'tilt').mean()  # .where(np.isfinite(anglescan_ds.voltage), drop=True)

        combined_ds = xr.concat([probes_1, probes_2], dim='probe')

    if average_direction_fl:
        combined_ds = combined_ds.mean('direction')

    return combined_ds


def plot_shot_trimming(filename, probes=None, downsample=100):
    analysis_ds = xr.open_dataset(filename)
    swp_lim_dn, swp_lim_up = find_sweep_limit(analysis_ds)

    if probes is None:
        probes = analysis_ds['probe'].values

    fig, ax = plt.subplots(len(probes), sharex=True)

    if len(probes) == 1:
        ax = [ax]

    for i, probe in enumerate(probes):
        semi_coarse_ds = analysis_ds.coarsen(time=downsample).mean().mean('direction')
        semi_coarse_ds.sel(probe=probe)['current'].max('time').plot.line(x='sweep', ax=ax[i], label='raw')
        semi_coarse_ds.sel(probe=probe).isel(sweep=slice(swp_lim_dn, swp_lim_up))['current'].max('time')\
            .plot.line(x='sweep', ax=ax[i], label='trimmed')
        ax[i].legend()

    return fig


###############################################################
#                           Fitting                           #
###############################################################


def fit_magnum_ds(magnum_subset_ds, probes=('L', 'S', 'B', 'R'), ax=True, scan_param='tilt', scan_selection=None,
                  threshold='auto', fitter=None, print_fl=False, temp_from_phi_fl=True, multi_fit_fl=True, mass=1, Z=1,
                  **kwargs):
    if fitter is None:
        fitter = fts.FullIVFitter()

    if scan_selection is not None:
        magnum_subset_ds = magnum_subset_ds.sel(scan_selection)

    metadata_labels = [
        scan_param,
        'probe',
        'B',
        'ts_temp',
        'ts_dens',
        'fit_success_fl',
    ]
    fit_param_labels = [
        'temp',
        'd_temp',
        'isat',
        'd_isat',
        'a',
        'd_a',
        'v_f',
        'd_v_f',
        'dens',
        'd_dens',
        'chi2',
        'reduced_chi2'
    ]
    phi_labels = [
        'phi',
        'temp_phi',
        'temp_fit_phi'
    ]
    if temp_from_phi_fl:
        fit_param_labels += phi_labels

    all_labels = metadata_labels + fit_param_labels

    fit_df = pd.DataFrame(columns=all_labels)

    for scan_param_value in magnum_subset_ds[scan_param].values:
        scan_param_ds = magnum_subset_ds.sel(**{scan_param: scan_param_value})
        if print_fl:
            print(f'\n\n -- ({scan_param_ds.shot_number.values}) Running fit on {scan_param} with {scan_param_value} -- \n')
        for probe in probes:
            probe_paramscan_ds = scan_param_ds.sel(probe=probe)
            probe_paramscan_ds = probe_paramscan_ds.where(np.isfinite(probe_paramscan_ds['voltage']), drop=True)
            probe_paramscan_ds = probe_paramscan_ds.where(np.isfinite(probe_paramscan_ds['current']), drop=True)

            if threshold == 'auto':
                # Auto option finds the first point the other side of the floating potential and then selects up to that
                threshold = 1

            if isinstance(threshold, int):
                iv_indices = np.where(probe_paramscan_ds.current < 0)[0]
                if iv_indices[0] == 0:
                    extreme_index = max(iv_indices)
                    extension = [extreme_index + (i + 1) for i in range(threshold)]
                    ext_iv_indices = np.concatenate((iv_indices, extension))
                else:
                    extreme_index = min(iv_indices)
                    extension = [extreme_index - (threshold - i) for i in range(threshold)]
                    ext_iv_indices = np.concatenate((extension, iv_indices))
                probe_paramscan_ds = probe_paramscan_ds.isel(time=ext_iv_indices)
            elif isinstance(threshold, float):
                probe_paramscan_ds = probe_paramscan_ds.where(probe_paramscan_ds.current < threshold, drop=True)
            elif print_fl:
                print('No threshold set, continuing with full sweep.')

            if scan_param == 'tilt':
                tilt = scan_param_value
            else:
                tilt = np.array(probe_paramscan_ds['tilt'].values, ndmin=1)[0]
            alpha = np.radians(tilt)

            if len(probe_paramscan_ds.time) == 0:
                if print_fl:
                    print('Time has no length, continuing...')
                continue

            shot_iv = iv.IVData(probe_paramscan_ds['voltage'].values,
                                -probe_paramscan_ds['current'].values,
                                probe_paramscan_ds['shot_time'].values,
                                sigma=probe_paramscan_ds['d_current'].values)

            # Generate a guess for a 4-parameter IV characteristic
            temp_guess = probe_paramscan_ds['ts_temperature'].mean().values
            dens_guess = probe_paramscan_ds['ts_density'].mean().values
            a_guess = magnum_probes[probe].get_sheath_exp_param(temp_guess, dens_guess, alpha)
            isat_guess = probe_paramscan_ds['current'].min('time').values
            v_f_guess = fitter.find_floating_pot_iv_data(shot_iv)

            initial_params = fitter.generate_guess(
                temperature=temp_guess,
                floating_potential=v_f_guess,
                isat=isat_guess,
                sheath_exp_param=a_guess
            )
            if print_fl:
                print(f"First guess initial_params: {[temp_guess, dens_guess, isat_guess, v_f_guess]}")
                print(f"Generated initial_params: {initial_params}")

            try:
                if multi_fit_fl:
                    shot_fit = shot_iv.multi_fit(print_fl=print_fl, stage_2_guess=initial_params, iv_fitter=fitter,
                                                 **kwargs)
                else:
                    shot_fit = fitter.fit_iv_data(shot_iv, sigma=shot_iv['sigma'], initial_vals=initial_params)

                dens = magnum_probes[probe].get_density(shot_fit.get_isat(), shot_fit.get_temp(), alpha=alpha,
                                                        mass=mass, Z=Z)
                d_dens = magnum_probes[probe].get_d_density(
                    shot_fit.get_isat(),
                    shot_fit.get_isat_err(),
                    shot_fit.get_temp(),
                    shot_fit.get_temp_err(),
                    alpha=alpha,
                    mass=mass,
                    Z=Z,
                )
                if isinstance(dens, collections.Iterable):
                    dens = dens[0]
                    d_dens = d_dens[0]

                fit_params = {
                    'fit_success_fl': True,
                    'temp': shot_fit.get_temp(),
                    'd_temp': shot_fit.get_temp_err(),
                    'isat': shot_fit.get_isat(),
                    'd_isat': shot_fit.get_isat_err(),
                    'a': shot_fit.get_sheath_exp(),
                    'd_a': shot_fit.get_sheath_exp_err(),
                    'v_f': shot_fit.get_floating_pot(),
                    'd_v_f': shot_fit.get_floating_pot_err(),
                    'dens': dens,
                    'd_dens': d_dens,
                    'chi2': shot_fit.chi2,
                    'reduced_chi2': shot_fit.reduced_chi2,
                }

                if temp_from_phi_fl:
                    phi = shot_iv.estimate_phi()
                    temp_phi = lpu.estimate_temperature(shot_iv.get_vf(), phi)
                    temp_fit_phi = lpu.estimate_temperature(shot_fit.get_floating_pot(), phi)
                    fit_params.update({
                        'phi': phi,
                        'temp_phi': temp_phi,
                        'temp_fit_phi': temp_fit_phi
                    })

                if ax is not None:
                    if ax is True:
                        fig, ax = plt.subplots()
                    ax.errorbar(shot_iv['V'], shot_iv['I'], yerr=shot_iv['sigma'],
                                ecolor='silver', color='silver', marker='+', zorder=1)
                    ax.plot(*shot_fit.get_fit_plottables())
            except RuntimeError as e:
                print(f'WARNING: Failed on {scan_param}={scan_param_value} with probe {probe}')
                print(e)
                fit_params = {label: np.NaN for label in fit_param_labels}
                fit_params['fit_success_fl'] = False

            fit_df = fit_df.append({
                scan_param: scan_param_value,
                'probe': probe,
                'B': np.around(probe_paramscan_ds['shot_b_field'].mean().values, decimals=1),
                'ts_temp': probe_paramscan_ds['ts_temperature'].max().values,
                'ts_dens': probe_paramscan_ds['ts_density'].max().values,
                **fit_params,
            }, ignore_index=True)

    return fit_df


DEFAULT_PROBES = ('S', 'L', 'B')
DEFAULT_SETTINGS = {'mode': 3, 'trimming_vals': (0.3, 0.02, 0.02), 'sat_region': -30}
DEFAULT_PROBES_SETTINGS = ((DEFAULT_PROBES, DEFAULT_SETTINGS), )


def fit_multi_magnum_ds(angle_scan_ds, probes_settings=DEFAULT_PROBES_SETTINGS, **kwargs):
    fit_dfs = []
    if not isinstance(probes_settings, dict):
        probes_settings = dict(probes_settings)

    for probes, settings in probes_settings.items():
        if not isinstance(probes, collections.Iterable):
            probes = tuple(probes)
        fit_df = fit_magnum_ds(angle_scan_ds, probes=probes, **settings, **kwargs)
        fit_dfs.append(fit_df)

    return pd.concat(fit_dfs)


def combine_fit_ds(fit_df, magnum_subset_ds, probes=('L', 'S', 'B', 'R'), index_dim='tilt'):
    fit_ds = fit_df.to_xarray().set_coords(['probe', index_dim]).swap_dims({'index': index_dim}).drop('index')

    probe_datasets = []
    for i, probe in enumerate(probes):
        probe_datasets.append(fit_ds.where(fit_ds.probe == probe, drop=True).drop('probe'))

    probe_da = xr.DataArray(list(probes), dims=['probe'], name='probe')
    restacked_fit_ds = xr.concat(probe_datasets, dim=probe_da)

    combined_ds = xr.merge([magnum_subset_ds, restacked_fit_ds])
    combined_ds['normal_tilt'] = 90 - combined_ds['tilt']
    return combined_ds


def interpolate_ts_position(dataset, uncertainty=1.5, offset=0, aggregate_dims=('probe', 'tilt'), probes=None,
                            print_fl=True):
    if probes is None:
        probes = lpu.MagnumProbes()

    if aggregate_dims is None:
        ts_temp = dataset.ts_temperature
        ts_d_temp = dataset.ts_d_temperature
        ts_dens = dataset.ts_density
        ts_d_dens = dataset.ts_d_density
    else:
        ts_temp = dataset.ts_temperature.mean(aggregate_dims)
        ts_d_temp = dataset.ts_temperature.std(aggregate_dims)
        ts_dens = dataset.ts_density.mean(aggregate_dims)
        ts_d_dens = dataset.ts_density.std(aggregate_dims)

    ts_probe_temps = {}
    ts_probe_denss = {}
    ts_probe_d_temps = {}
    ts_probe_d_denss = {}
    for probe, probe_pos in probes.probe_position.items():
        probe_pos += offset
        probe_pos_limits = [probe_pos, probe_pos - uncertainty, probe_pos + uncertainty]
        if print_fl:
            print(probe, probe_pos)
        ts_probe_temps[probe.upper()] = ts_temp.interp(ts_radial_pos=probe_pos_limits).values
        ts_probe_denss[probe.upper()] = ts_dens.interp(ts_radial_pos=probe_pos_limits).values
        ts_probe_d_temps[probe.upper()] = ts_d_temp.interp(ts_radial_pos=probe_pos_limits).values
        ts_probe_d_denss[probe.upper()] = ts_d_dens.interp(ts_radial_pos=probe_pos_limits).values

    return ts_probe_temps, ts_probe_denss, ts_probe_d_temps, ts_probe_d_denss

# ############ Plotting routines ############ #

def plot_anglescan_ivs(anglescan_ds, sup_title=None):
    iv_fig, iv_ax = plt.subplots(2, 2, sharex=True)
    for i, probe in enumerate(anglescan_ds['probe'].values):
        iv_ax[i % 2][i // 2].set_title(probe)

        ds = anglescan_ds.sel(probe=probe)
        ds.set_coords('voltage')['current'].plot.line(ax=iv_ax[i % 2][i // 2], x='voltage', hue='tilt')
        iv_ax[i % 2][i // 2].axhline(0, linestyle='--', color='grey', linewidth=1)
        iv_ax[i % 2][i // 2].set_ylabel('I (A)')
        iv_ax[i % 2][i // 2].set_xlabel(r'$V_P$ (V)')

    iv_fig.suptitle(f'IV characteristics for {sup_title}')
    plt.show()


def plot_anglescan_ivs_vertical(anglescan_ds):
    fig, ax = plt.subplots(4, sharex=True, sharey=True)
    anglescan_ds.set_coords('voltage').sel(probe='L')['current'].plot.line(x='voltage', hue='tilt', ax=ax[0])
    anglescan_ds.set_coords('voltage').sel(probe='S')['current'].plot.line(x='voltage', hue='tilt', ax=ax[1])
    anglescan_ds.set_coords('voltage').sel(probe='B')['current'].plot.line(x='voltage', hue='tilt', ax=ax[2])
    anglescan_ds.set_coords('voltage').sel(probe='R')['current'].plot.line(x='voltage', hue='tilt', ax=ax[3])
    ax[0].axhline(0, linestyle='--', color='grey', linewidth=1)
    ax[1].axhline(0, linestyle='--', color='grey', linewidth=1)
    ax[2].axhline(0, linestyle='--', color='grey', linewidth=1)
    ax[3].axhline(0, linestyle='--', color='grey', linewidth=1)
    plt.show()


def plot_anglescan_averaged_ts(anglescan_ds, sup_title=None):
    ts_temp = anglescan_ds.ts_temperature.mean(['probe', 'tilt'])
    ts_d_temp = anglescan_ds.ts_temperature.std(['probe', 'tilt'])
    ts_dens = anglescan_ds.ts_density.mean(['probe', 'tilt'])
    ts_d_dens = anglescan_ds.ts_density.std(['probe', 'tilt'])

    fig, ax = plt.subplots(2)
    ax[0].set_title('Temperature')
    ax[1].set_title('Density')
    # ax[0].errorbar(helium_anglescan_ds.ts_radial_pos, ts_temp, yerr=helium_anglescan_ds.ts_d_temperature.mean('shot_number'), color='silver', ecolor='silver')
    ax[0].errorbar(anglescan_ds.ts_radial_pos, ts_temp, yerr=ts_d_temp)
    ax[1].errorbar(anglescan_ds.ts_radial_pos, ts_dens, yerr=ts_d_dens)

    for i in [0, 1]:
        for probe_pos in [-6, 4, 14, 24]:
            ax[i].axvline(x=probe_pos, color='black', linewidth=1, linestyle='dashed')
    plt.show()
    fig.suptitle(f'Thomson Scattering profiles for {sup_title}')


def plot_anglescan_multi_ts(anglescan_ds, ax=None, sup_title=None, probe='S'):
    ts_temp = anglescan_ds['ts_temperature'].sel(probe=probe)
    ts_d_temp = anglescan_ds['ts_d_temperature'].sel(probe=probe)
    ts_dens = anglescan_ds['ts_density'].sel(probe=probe)
    ts_d_dens = anglescan_ds['ts_d_density'].sel(probe=probe)

    if ax is None:
        fig, ax = plt.subplots(2)

    ax[0].set_title('Temperature')
    ax[1].set_title('Density')
    ts_temp.plot.line(hue='tilt', ax=ax[0])
    ts_dens.plot.line(hue='tilt', ax=ax[1])

    for i in [0, 1]:
        for probe_pos in [-6, 4, 14, 24]:
            ax[i].axvline(x=probe_pos, color='black', linewidth=1, linestyle='dashed')
    plt.show()

    plt.suptitle(f'Thomson Scattering profiles for {sup_title}')


def fit_by_upper_index(iv_data_ds, upper_index, ax=None, multi_fit_fl=False, plot_fl=True):
    """
    This has been superceded by the IVData method fit_to_minimum()
    """
    import flopter.core.fitters as f

    # Determine which way round the sweep goes
    if iv_data_ds['voltage'].values[0] < iv_data_ds['voltage'].values[-1]:
        iv_data_trimmed_ds = iv_data_ds.isel(time=slice(0, upper_index))
    else:
        iv_data_trimmed_ds = iv_data_ds.isel(time=slice(iv_data_ds.time.size - upper_index, -1))

    if plot_fl:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        fig.suptitle(f'upper_index = {upper_index}')
        ax.errorbar(iv_data_trimmed_ds['voltage'].values, -iv_data_trimmed_ds['current'].values,
                    yerr=iv_data_trimmed_ds['d_current'].values,
                    linestyle='none', color='k', ecolor='k', label='Sweep-averaged IV', zorder=2)

        # Plot the whole IV in an inset axis
        inner_ax = plt.axes([0.2, 0.35, .2, .2])
        (-iv_data_ds.set_coords('voltage')['current']).plot(x='voltage', ax=inner_ax)
        inner_ax.axvline(x=iv_data_trimmed_ds.max('time')['voltage'].values, **c.AX_LINE_DEFAULTS)
        inner_ax.set_title('Whole IV')
        inner_ax.set_xlabel('V')
        inner_ax.set_ylabel('I')
        inner_ax.set_xticks([])
        inner_ax.set_yticks([])

    shot_iv = iv.IVData(iv_data_trimmed_ds['voltage'].values,
                        -iv_data_trimmed_ds['current'].values,
                        iv_data_trimmed_ds['shot_time'].values,
                        sigma=iv_data_trimmed_ds['stderr_current'].values)
    # shot_iv = iv.IVData(iv_data_ds['voltage'].values,
    #                     -iv_data_ds['current'].values,
    #                     iv_data_ds['shot_time'].values,
    #                     sigma=iv_data_ds['stderr_current'].values)

    if multi_fit_fl:
        shot_fit = shot_iv.multi_fit(sat_region=-52)
    else:
        fitter = f.FullIVFitter()
        shot_fit = fitter.fit_iv_data(shot_iv, sigma=shot_iv['sigma'])

    if plot_fl:
        chi_2_str = r"$\chi^2_{red}$"
        ax.plot(*shot_fit.get_fit_plottables(),
                label=f'Fit - T_e={shot_fit.get_temp():.3g}, {chi_2_str} = {shot_fit.reduced_chi2:.3g}')

        ax.legend()

    return shot_fit

# Density scan / non-tilt-indexed scan plotting functions

def plot_densscan_multi_ts(densscan_ds, ax=None, sup_title=None):
    ts_temp = densscan_ds['ts_temperature']
    ts_d_temp = densscan_ds['ts_d_temperature']
    ts_dens = densscan_ds['ts_density']
    ts_d_dens = densscan_ds['ts_d_density']

    if ax is None:
        fig, ax = plt.subplots(2)

    colourmap = plt.get_cmap('nipy_spectral')
    cycler = mpl.cycler(color=[colourmap(k) for k in np.linspace(0, 1, len(densscan_ds['shot_number'].values))])

    ax[0].set_title('Temperature')
    ax[0].set_prop_cycle(cycler)
    ax[1].set_title('Density')
    ax[1].set_prop_cycle(cycler)

    ts_temp.plot.line(hue='shot_number', ax=ax[0])
    ts_dens.plot.line(hue='shot_number', ax=ax[1])

    for i in [0, 1]:
        for probe_pos in [-6, 4, 14, 24]:
            ax[i].axvline(x=probe_pos, color='black', linewidth=1, linestyle='dashed')
    plt.show()

    plt.suptitle(f'Thomson Scattering profiles for {sup_title}')


def plot_densscan_paramspace(densscan_ds, ax=None, sup_title=None):
    if ax is None:
        fig, ax = plt.subplots()

    densscan_ds.set_coords(['ts_temp_max', 'ts_dens_max']).plot.scatter(x='ts_temp_max', y='ts_dens_max', ax=ax)

    plt.show()
    math_str = r'$T_e n_e$'
    plt.suptitle(f'Thomson Scattering {math_str} parameter space for {sup_title}')


def plot_densscan_ivs(densscan_ds):
    probes = np.array(densscan_ds.probe.values, ndmin=1)

    if len(probes) == 1:
        fig, ax = plt.subplots()
        densscan_ds.set_coords('voltage')['current'].plot.line(x='voltage', hue='shot_number', ax=ax)
    else:
        fig, ax = plt.subplots(len(probes))
        for i, probe in enumerate(probes):
            ds = densscan_ds.sel(probe=probe)
            ds.set_coords('voltage')['current'].plot.line(x='voltage', hue='shot_number', ax=ax[i])

    return fig, ax


def plot_densscan_timeseries(densscan_ds):
    probes = np.array(densscan_ds.probe.values, ndmin=1)

    if len(probes) == 1:
        fig, ax = plt.subplots(2, sharex=True)
        densscan_ds['current'].plot.line(x='time', hue='shot_number', ax=ax[0])
        densscan_ds['voltage'].plot.line(x='time', hue='shot_number', ax=ax[1])

    else:
        fig, ax = plt.subplots(2, len(probes), sharex=True)
        for i, probe in enumerate(probes):
            ds = densscan_ds.sel(probe=probe)
            ds['current'].plot.line(x='time', hue='shot_number', ax=ax[i][0])
            ds['voltage'].plot.line(x='time', hue='shot_number', ax=ax[i][1])

    return fig, ax
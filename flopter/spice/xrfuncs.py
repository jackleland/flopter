import os
import glob
import numpy as np
import pathlib as pth
import flopter.spice.splopter as spl
import flopter.spice.tdata as td
import xarray as xr
import flopter.core.constants as c
import flopter.core.fitdata as fd
import flopter.core.ivdata as iv
import flopter.core.fitters as fts
import flopter.core.lputils as lpu

NON_STANDARD_VARIABLES = {'t', 'WallPot', 'ProbePot', 'npartproc', 'Nz', 'Nzmax', 'Ny', 'count', 'Npc', 'snumber', 'nproc'}
DESIRED_VARIABLES = (td.DEFAULT_REDUCED_DATASET | NON_STANDARD_VARIABLES) - \
                    {td.OBJECTSCURRENTFLUXE, td.OBJECTSCURRENTFLUXI}
DEFAULT_INITIAL_PARAMS = {
    'temperature': 1.0,
    'floating_potential': -2.5,
    'isat': -50,
    'sheath_exp_param': 0.01,
}
DEFAULT_MF_TRIM_VALS = (0.3, 0.3, 0.02)
DEFAULT_CURRENT_RENORM = 3.5
DEFAULT_DENSITY = 1e18
PROBE_DESIGNATIONS = {
    'angled_flush': {'theta_p': 10.0, 'recession': 0.0, 'gap': 1.0e-3},
    'angled_recessed': {'theta_p': 10.0, 'recession': 1.0e-3, 'gap': 1.0e-3},
    'flat_flush': {'theta_p': 0.0, 'recession': 0.0, 'gap': 1.0e-3},
    'flat_recessed': {'theta_p': 0.0, 'recession': 1.0e-3, 'gap': 1.0e-3},
    'flat_flush_gapless': {'theta_p': 0.0, 'recession': 0.0, 'gap': 0.0}
}
FLUSH_SQUARE_PROBE = lpu.AngledTipProbe(a=5e-3, b=5e-3, L=5e-3, g=1e-3, d_perp=0e-4,
                                        theta_f=0.0, theta_p=0.0)


def get_initial_params(angle, splopter=None, probe=FLUSH_SQUARE_PROBE):
    temperature = DEFAULT_INITIAL_PARAMS['temperature']
    density = DEFAULT_DENSITY

    if splopter is not None:
        try:
            simulation_params = splopter.parser.get_commented_params()
            temperature = simulation_params[c.ELEC_TEMP]
            density = simulation_params[c.ELEC_DENS]
        except:
            temperature = DEFAULT_INITIAL_PARAMS['temperature']
            density = DEFAULT_DENSITY

    isat = (probe.get_isat(temperature, density, np.radians(angle)) / probe.a)*DEFAULT_CURRENT_RENORM
    floating_potential = DEFAULT_INITIAL_PARAMS['floating_potential']
    sheath_exp_param = DEFAULT_INITIAL_PARAMS['sheath_exp_param']
    return {
        'temperature': temperature,
        'floating_potential': floating_potential,
        'isat': isat,
        'sheath_exp_param': sheath_exp_param,
    }


def homogenise_run(path):
    splopter = spl.Splopter(path, reduce=DESIRED_VARIABLES, ignore_tzero_fl=True, version=2.14,
                            store_dataframe_fl=True, ignore_a_file=True)
    splopter.parse_input()
    iv_data, raw_iv_data = splopter.homogenise(reduced_bin_fract=0.1)
    return splopter.iv_df, iv_data


def get_dummy_fit(iv_data, fitter):
    dummy_params = np.nan * np.zeros_like(fitter.default_values)
    dum_fit_data = fd.IVFitData(iv_data['V'], iv_data['I'], np.nan * np.zeros_like(iv_data['I']),
                                dummy_params, dummy_params, fitter,
                                sigma=iv_data['sigma'], chi2=np.nan, reduced_chi2=np.nan)
    return dum_fit_data


def straight_iv_fit(iv_data, cutoff=-2, all_initial_values=None):
    if all_initial_values is None:
        all_initial_values = DEFAULT_INITIAL_PARAMS

    fitter = fts.FullIVFitter()
    if cutoff is None:
        cutoff = fts.IVFitter.find_floating_pot(iv_data['V'][:-1], iv_data['I'][:-1])

    iv_region = np.where(iv_data['V'] <= cutoff)
    fit_iv_data = iv.IVData.non_contiguous_trim(iv_data, iv_region)

    initial_vals = fitter.generate_guess(**all_initial_values)

    try:
        fit_data = fitter.fit_iv_data(fit_iv_data, sigma=fit_iv_data['sigma'], initial_vals=initial_vals)
    except ValueError:
        print('CAUGHT: straight fit error')
        fit_data = get_dummy_fit(fit_iv_data, fitter)

    return fit_data


def experimental_iv_fit(iv_data, cutoff=None, show_err_fl=True, all_initial_values=None):
    if all_initial_values is None:
        all_initial_values = DEFAULT_INITIAL_PARAMS

    fitter = fts.ExperimentalIVFitter()
    if cutoff is not None:
        iv_region = np.where(iv_data['V'] <= cutoff)
    else:
        V_f = fts.IVFitter.find_floating_pot(iv_data['V'][:-1], iv_data['I'][:-1])
        iv_region = np.where(iv_data['V'] < V_f)

    initial_vals = fitter.generate_guess(**all_initial_values)

    fit_iv_data = iv.IVData.non_contiguous_trim(iv_data, iv_region)
    try:
        fit_data = fitter.fit_iv_data(fit_iv_data, sigma=fit_iv_data['sigma'], initial_vals=initial_vals)
    except ValueError as e:
        print('CAUGHT: experimental fit error')
        if show_err_fl:
            print(e)
        fit_data = get_dummy_fit(fit_iv_data, fitter)

    return fit_data


def multi_iv_fit(iv_data, sat_region=-6, show_err_fl=True, all_initial_values=None, **kwargs):
    if all_initial_values is None:
        all_initial_values = DEFAULT_INITIAL_PARAMS

    if 'iv_fitter' in kwargs:
        fitter = kwargs['iv_fitter']
    else:
        fitter = fts.FullIVFitter()

    if 'stage_2_guess' not in kwargs:
        initial_vals = fitter.generate_guess(**all_initial_values)
    else:
        initial_vals = kwargs.pop('stage_2_guess')

    try:
        fit_data = iv_data.multi_fit(sat_region=sat_region, stage_2_guess=initial_vals, **kwargs)
    except ValueError as e:
        print('CAUGHT: multi fit error')
        if show_err_fl:
            print(e)
        fit_data = get_dummy_fit(iv_data, fitter)

    return fit_data


def norm_ion_fit(iv_data_orig, sigma=None, show_err_fl=False, v_redund=6, all_initial_values=None, voltage_cap=-30.0,
                 voltage_shift=None):
    if all_initial_values is None:
        all_initial_values = DEFAULT_INITIAL_PARAMS

    iv_data = iv_data_orig.copy()
    if sigma is None:
        iv_data['sigma'] = iv_data['sigma'] * 0.5
    else:
        iv_data['sigma'] = sigma
    if voltage_shift is None:
        V_f = fts.IVFitter.find_floating_pot(iv_data['V'][:-1], iv_data['I'][:-1])
    else:
        V_f = voltage_shift

    iv_region = np.where(np.logical_and(iv_data['V'] < (V_f - v_redund), iv_data['V'] > voltage_cap))
    fit_iv_data = iv.IVData.non_contiguous_trim(iv_data, iv_region)
    i_fitter = fts.IonCurrentSEFitter()
    initial_vals = i_fitter.generate_guess(**all_initial_values)

    try:
        fit_data = i_fitter.fit(np.float_power(np.abs(fit_iv_data['V'] - V_f), .75),
                                fit_iv_data['I_i'],
                                sigma=fit_iv_data['sigma'],
                                initial_vals=initial_vals)
    except ValueError as e:
        print('CAUGHT: ion fit value error')
        if show_err_fl:
            print(e)
        fit_data = get_dummy_fit(fit_iv_data, i_fitter)
    except RuntimeError as e:
        print('CAUGHT: ion fit runtime error')
        if show_err_fl:
            print(e)
        fit_data = get_dummy_fit(fit_iv_data, i_fitter)

    return fit_data


def norm_full_fit(iv_data, show_err_fl=False, all_initial_values=None):
    if all_initial_values is None:
        all_initial_values = DEFAULT_INITIAL_PARAMS

    V_f = fts.IVFitter.find_floating_pot(iv_data['V'][:-1], iv_data['I'][:-1])

    iv_region = np.where(iv_data['V'] < V_f)
    fit_iv_data = iv.IVData.non_contiguous_trim(iv_data, iv_region)

    voltage = fit_iv_data['V'] - V_f
    current = fit_iv_data['I']
    d_current = fit_iv_data['sigma']

    iv_fitter = fts.NormalisedIVFitter()
    initial_vals = iv_fitter.generate_guess(**all_initial_values)
    try:
        fit_data = iv_fitter.fit(voltage, current, sigma=d_current, initial_vals=initial_vals)
    except ValueError as e:
        print('CAUGHT: norm full fit error')
        if show_err_fl:
            print(e)
        fit_data = get_dummy_fit(fit_iv_data, iv_fitter)

    return fit_data


def norm_electron_fit(iv_data, sigma=None, show_err_fl=False, all_initial_values=None):
    if all_initial_values is None:
        all_initial_values = DEFAULT_INITIAL_PARAMS

    V_f = fts.IVFitter.find_floating_pot(iv_data['V'][:-1], iv_data['I'][:-1])

    if sigma is None:
        iv_data['sigma'] = iv_data['sigma'] * 0.5
    else:
        iv_data['sigma'] = sigma

    iv_region = np.where(iv_data['V'] < V_f)
    fit_iv_data = iv.IVData.non_contiguous_trim(iv_data, iv_region)

    voltage = fit_iv_data['V']
    current = fit_iv_data['I_e']
    d_current = sigma
    if sigma is None:
        d_current = fit_iv_data['sigma'] * 0.5

    elec_fitter = fts.ElectronCurrentFitter()
    initial_vals = elec_fitter.generate_guess(**all_initial_values)
    try:
        fit_data = elec_fitter.fit(voltage, current, sigma=d_current, initial_vals=initial_vals)
    except ValueError as e:
        print('CAUGHT: electron fit error')
        if show_err_fl:
            print(e)
        fit_data = get_dummy_fit(fit_iv_data, elec_fitter)

    return fit_data


def fitdata_to_dataset(fit_data, fit_prefix='fit', extras=None):
    fit_ds = xr.Dataset(fit_data.to_dict()).drop(['raw_y', 'sigma', 'fit_y'])
    if extras is not None:
        for label, value in extras.items():
            fit_ds[label] = value
    fit_ds = fit_ds.assign(
        voltage_min=fit_ds.raw_x.min(),
        voltage_max=fit_ds.raw_x.max()
    ).drop('raw_x').expand_dims(dim=['theta'])
    fit_ds = fit_ds.rename_vars({data_var: f'{fit_prefix}_{data_var}' for data_var in fit_ds.data_vars})

    return fit_ds


def generate_splopter_ivs(scans, all_run_dirs, skip_angles=('-1.0',), spice_dir=None):
    if spice_dir is None:
        spice_dir = pth.Path('/home/jleland/data/external_big/spice/')
    os.chdir(spice_dir)

    iv_dfs = {}
    iv_datas = {}
    failed_runs = []

    for scan in scans:
        for run_dir in all_run_dirs[scan]:
            run_path = spice_dir / run_dir

            bups = list(run_path.glob('backup_20*'))
            bups.sort()
            final_state_path = bups[-1]

            print(f'\n\n --- {run_dir} --- \n')
            if run_dir in iv_dfs:
                print(f"!!! Skipping {run_dir} (in iv_dfs)...")
                continue
            if any([angle in run_dir for angle in skip_angles]):
                print(f'!!! Skipping {run_dir} (in skip_angles)...')
                continue

            try:
                iv_df, iv_data = homogenise_run(final_state_path)

                iv_dfs[run_dir] = iv_df
                iv_datas[run_dir] = iv_data
            except Exception as e:
                failed_runs.append(run_dir)
                print(f'Failed on {run_dir}, {e}')

    print(f'The following runs failed \n'
          f'{failed_runs}\n\n')

    return iv_dfs, iv_datas, failed_runs


DEFAULT_ION_FIT_VOLTAGE_CAP = -15.0


def generate_dataset_dict(scans, all_run_dirs, iv_dfs, iv_datas, failed_runs=None):
    if failed_runs is None:
        failed_runs = set()

    datasets = {}
    probes = []
    thetas = []

    for scan in scans:
        for run_dir in all_run_dirs[scan]:
            if run_dir in failed_runs or run_dir not in iv_datas:
                print(f'Skipping {run_dir} as it failed...')
                continue

            angle = float(run_dir.split('/')[-1].split('-')[-1])
            #         probe = run_dir.split('/')[-2].split('_1')[0]
            run_str = run_dir.split('/')[-2].split('_')
            probe = '_'.join(run_str[0:2])
            if len(run_str) > 2 and run_str[2] == 'gapless':
                probe = probe + '_gapless'
            print(probe, angle)

            if angle not in thetas:
                thetas.append(angle)
            if probe not in probes:
                probes.append(probe)
            if probe not in datasets:
                datasets[probe] = []

            all_initial_params = get_initial_params(angle)

            iv_data = iv_datas[run_dir]
            iv_df = iv_dfs[run_dir]

            v_f = iv_data.get_vf()
            v_w = iv_df['voltage_wall'].mean()
            voltage_corr = iv_data['V'] - v_f
            voltage_cl = np.float_power(np.abs(voltage_corr), 0.75)

            iv_ds = iv_df.to_xarray().swap_dims({'index': 'voltage'}).drop('index').expand_dims(dim=['theta'])

            str_iv_fit_ds = fitdata_to_dataset(straight_iv_fit(iv_data, cutoff=None),
                                               fit_prefix='str_iv', extras={'run_dir': run_dir})
            # exp_iv_fit_ds = fitdata_to_dataset(experimental_iv_fit(iv_data, all_initial_values=all_initial_params),
            #                                    fit_prefix='expmt_iv')
            multi_iv_fit_ds = fitdata_to_dataset(multi_iv_fit(iv_data, all_initial_values=all_initial_params),
                                                 fit_prefix='mf_iv')
            # multi_iv_exp_fit_ds = fitdata_to_dataset(multi_iv_fit(iv_data, iv_fitter=fts.ExperimentalIVFitter(),
            #                                                       all_initial_values=all_initial_params),
            #                                          fit_prefix='mf_expmt_iv')
            norm_iv_fit_ds = fitdata_to_dataset(norm_full_fit(iv_data, all_initial_values=all_initial_params),
                                                fit_prefix='norm_iv')
            ion_fit_ds = fitdata_to_dataset(norm_ion_fit(iv_data, sigma=iv_df['d_current_i'], v_redund=0.1,
                                                         voltage_cap=DEFAULT_ION_FIT_VOLTAGE_CAP,
                                                         all_initial_values=all_initial_params),
                                            fit_prefix='ion')
            # ion_sh_fit_ds = fitdata_to_dataset(norm_ion_fit(iv_data, sigma=iv_df['d_current_i'], voltage_cap=-10,
            #                                                 voltage_shift=v_w, all_initial_values=all_initial_params),
            #                                 fit_prefix='ion_sh')
            elec_fit_ds = fitdata_to_dataset(norm_electron_fit(iv_data, all_initial_values=all_initial_params),
                                             fit_prefix='elec')

            ds = xr.merge([str_iv_fit_ds, norm_iv_fit_ds, ion_fit_ds,
                           elec_fit_ds, multi_iv_fit_ds, iv_ds]).assign_coords(
                theta=xr.DataArray([angle], dims='theta'),
                v_f=xr.DataArray([v_f], dims='theta'),
            ).assign(
                voltage_corr=xr.DataArray([voltage_corr], dims=['theta', 'voltage']),
                voltage_cl=xr.DataArray([voltage_cl], dims=['theta', 'voltage']),
            )
            datasets[probe].append(ds)

    return datasets, probes, thetas


def create_scan_probe_datasets(scans, all_run_dirs, skip_angles=('-1.0',), spice_dir=None):
    iv_dfs, iv_datas, failed_runs = generate_splopter_ivs(scans, all_run_dirs, skip_angles=skip_angles,
                                                          spice_dir=spice_dir)
    datasets, probes, thetas = generate_dataset_dict(scans, all_run_dirs, iv_dfs, iv_datas, failed_runs=failed_runs)
    return datasets, probes, thetas


def create_scan_dataset(scans, all_run_dirs, skip_angles=('-1.0', ), spice_dir=None, **kwargs):
    datasets, probes, thetas = create_scan_probe_datasets(scans, all_run_dirs, skip_angles=skip_angles,
                                                          spice_dir=spice_dir)
    if len(probes) == 1:
        return combine_1d_dataset(probes[0], datasets, add_probe_name_fl=True, **kwargs)
    else:
        return combine_2d_dataset(probes, datasets, **kwargs)


PROBE_THETA_PS = {
    'angled': 10.0,
    'flat': 0.0,
    'semi-angled': 5.0,
}
PROBE_RECESSIONS = {
    'recessed': 3.0e-4,
    'semi-recessed': 1.5e-4,
    'flush': 0.0e-4,
}


def combine_1d_dataset(probe_name, datasets, concat_dim='theta', theta_p=None, recession=None, add_probe_name_fl=False,
                       extra_dims=None):
    gap_desc = ''
    recession_descr = ''
    theta_p_descr = ''

    combined_ds = xr.concat(datasets[probe_name], dim=concat_dim).sortby(concat_dim)

    if theta_p is None:
        theta_p_descr = probe_name.split('_')[0]
        theta_p = PROBE_THETA_PS[theta_p_descr]
    if recession is None:
        recession_descr = probe_name.split('_')[1]
        recession = PROBE_RECESSIONS[recession_descr]
    angled_fl = theta_p != 0
    recessed_fl = recession != 0.0

    if 'gapless' in probe_name:
        gap = 0.0
        gap_desc = 'gapless'
    else:
        gap = 1.0e-3

    combined_ds = combined_ds.assign_coords(
        recession=recession,
        gap=gap,
        theta_p=theta_p,
        theta_p_rads=np.radians(theta_p),
        theta_rads=np.radians(combined_ds.theta),
        recession_descr=recession_descr,
        gap_desc=gap_desc,
        theta_p_descr=theta_p_descr,
        recessed_fl=recessed_fl,
        angled_fl=angled_fl
    )
    if add_probe_name_fl:
        combined_ds = combined_ds.expand_dims(dim=['probe']).assign_coords(probe=[probe_name])

    if extra_dims is not None:
        for dim, coord in extra_dims.items():
            combined_ds = combined_ds.expand_dims(dim=[dim]).assign_coords({dim: [coord]})

    return combined_ds


def combine_2d_dataset(probe_names, datasets, extra_dims=None, kwargs_1d=None):
    if kwargs_1d is not None:
        c1d_datasets = [combine_1d_dataset(probe_name, datasets, **kwargs_1d) for probe_name in probe_names]
    else:
        c1d_datasets = [combine_1d_dataset(probe_name, datasets) for probe_name in probe_names]
    probe_da = xr.DataArray(probe_names, dims='probe', coords={'probe': probe_names})
    combined_ds = xr.concat(c1d_datasets, dim=probe_da).drop(None)
    if extra_dims is not None:
        for dim, coord in extra_dims.items():
            combined_ds = combined_ds.expand_dims(dim=[dim]).assign_coords({dim: [coord]})
    return combined_ds


def get_run_dirs(scan_search_strs=('*',), angles_search_str='/alpha_yz_*', skippable_scans=None, skippable_runs=None,
                 single_sims=tuple(), disallowed_angles=None, allowed_angles=None, print_fl=True):
    if skippable_scans is None:
        skippable_scans = set()
    if skippable_runs is None:
        skippable_runs = set()

    all_angles = {'-1.0', '-2.0', '-3.0', '-4.0', '-5.0', '-6.0', '-7.0', '-8.0', '-9.0',
                  '10.0', '11.0', '12.0', '15.0', '20.0', '30.0'}

    if disallowed_angles is None and allowed_angles is not None:
        disallowed_angles = all_angles - allowed_angles
    elif disallowed_angles is None and allowed_angles is None:
        disallowed_angles = set()
    elif disallowed_angles is not None and allowed_angles is not None:
        if allowed_angles == all_angles - disallowed_angles:
            print('Warning: You do not need to specify both disallowed and allowed angles, only one is required.')
        else:
            raise ValueError('Conflicting allowed and disallowed angle lists, only one is required.')

    all_run_dirs = {}
    scans = set().union(*[glob.glob(scan_searchstr) for scan_searchstr in scan_search_strs]) - skippable_scans
    for scan in scans:
        if scan in single_sims:
            all_run_dirs[scan] = [scan]
        else:
            all_run_dirs[scan] = [scan_run for scan_run in glob.glob(scan + angles_search_str)
                                  if scan_run[-4:] not in disallowed_angles and scan_run not in skippable_runs]

    scans = list(scans)
    scans.sort()

    if print_fl:
        print_run_dirs(scans, all_run_dirs)

    return scans, all_run_dirs


def print_run_dirs(scans, all_run_dirs):
    for i, scan in enumerate(scans):
        print(f"[{i}]: {scan}")
        for j, run in enumerate(all_run_dirs[scan]):
            print(f"\t[{i},{j}]: {'/'.join(run.split('/')[-2:])}")

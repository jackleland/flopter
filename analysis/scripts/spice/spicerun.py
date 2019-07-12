from flopter.core import constants as c, fitters as f
from flopter.spice import splopter as spl
import matplotlib.pyplot as plt
import numpy as np
import flopter.spice.tdata as sd
from scipy.signal import savgol_filter
from scipy.io import loadmat


def run_param_scan(splopter):
    # splopter = spl.Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullgap/')
    # splopter_nogap = Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullnogap/')

    ivdata_g = splopter.trim(trim_end=0.8)
    # ivdata_ng = splopter_nogap.trim()

    fig = plt.figure()
    splopter.plot_iv(iv_data=ivdata_g, fig=fig, plot_tot=True, label='Gap')
    n_params = 4
    params = [[]]*n_params
    errors = [[]]*n_params
    trim_space = np.linspace(0.4, 0.5, 11)
    print(params)
    for trim_end in trim_space:
        ivdata = splopter.trim(trim_end=trim_end)
        ivfitdata = splopter.fit(ivdata)
        splopter.plot_f_fit(ivfitdata, fig=fig, plot_raw=False, plot_vf=False, label=str(trim_end))
        fit_params, fit_errors = ivfitdata.get_fit_params().split()
        for i in range(n_params):
            if len(params[i]) == 0:
                params[i] = [fit_params[i]]
            else:
                params[i].append(fit_params[i])
            if len(errors[i]) == 0:
                errors[i] = [min(fit_errors[i], fit_params[i])]
            else:
                errors[i].append(min(fit_errors[i], fit_params[i]))

    for j in range(n_params):
        plt.figure()
        print(np.shape(trim_space))
        print(np.shape(params[j]))
        print(np.shape(errors[j]))
        plt.errorbar(trim_space, params[j], yerr=errors[j])
    plt.show()


def run_gap_nogap_comparison():
    splopter_gap = spl.Splopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullgap/', prepare=False)
    splopter_nogap = spl.Splopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullnogap/', prepare=False)

    splopter_gap.prepare(make_denormaliser=False)
    splopter_nogap.prepare(make_denormaliser=False)

    ivdata_g = splopter_gap.trim()
    ivdata_ng = splopter_nogap.trim()

    ivdata_g2 = splopter_gap.trim(trim_beg=0.05, trim_end=0.45)
    ivdata_ng2 = splopter_nogap.trim(trim_beg=0.05, trim_end=0.45)

    # ifit_g = splopter_gap.fit(ivdata_g, f.IonCurrentSEFitter(), print_fl=True)
    # ifit_ng = splopter_nogap.fit(ivdata_ng, f.IonCurrentSEFitter(), print_fl=True)

    ffit_g2 = splopter_gap.fit(ivdata_g2, print_fl=True)
    ffit_ng2 = splopter_nogap.fit(ivdata_ng2, print_fl=True)

    fig1 = plt.figure()
    splopter_gap.plot_iv(fig=fig1, plot_tot=True, label='Gap')
    splopter_nogap.plot_iv(fig=fig1, plot_vf=True, plot_tot=True, label='No Gap')

    fig2 = plt.figure()
    splopter_gap.plot_f_fit(ffit_g2, fig=fig2, plot_vf=False, label=r'Gap - $T_e$ = {:.2g}, $I_{}$ = {:.3g}'
                            .format(ffit_g2.get_temp(errors_fl=False), r'{sat}', ffit_g2.get_isat(errors_fl=False)))
    splopter_nogap.plot_f_fit(ffit_ng2, fig=fig2, label=r'No Gap - $T_e$ = {:.2g}, $I_{}$ = {:.3g}'
                              .format(ffit_ng2.get_temp(errors_fl=False), '{sat}', ffit_ng2.get_isat(errors_fl=False)))

    # fig3 = plt.figure()
    # splopter_gap.plot_i_fit(ifit_g, fig=fig3, label='Gap ')
    # splopter_nogap.plot_i_fit(ifit_ng, fig=fig3, label='No Gap ')
    plt.legend()
    plt.show()


def run_histogram_extr(splopter=None, z_high=370.0, z_low=70.0, fig=None, show=False, normalise_v=True, species=2,
                       t_flag=False, fitter=None):
    """
        NOTE: This has been implemented more generally as a method - splopter.extract_histograms() - and therefore this
        function has now been deprecated.
    """
    # if not splopter:
    #     splopter = spl.Splopter('bin/data_local/', 'benchmarking/', 'disttest_fullnogap/', prepare=False)
    #
    # region = {'Region': [z_low, z_high, y_low, y_high]}
    # v_scale = 1000
    # mass = {1: n.ELECTRON_MASS * splopter.denormaliser.mu,
    #         2: n.ELECTRON_MASS}
    #
    # hists = splopter.extract_histograms(region, denormalise=normalise_v, v_scale=v_scale)
    # hist_bins, hist = hists
    #
    # fitdata = get_histogram_fit(splopter, hist, hist_bins, fitter=fitter, v_scale=v_scale)
    #
    # if t_flag:
    #     T_e = (fitdata.fit_params[0].value * mass[species]) / (2 * n.ELEM_CHARGE)
    #     T_e_err = (fitdata.fit_params[0].error * mass[species]) / (2 * n.ELEM_CHARGE)
    #     print('T_e = {} +/- {}'.format(T_e, T_e_err))
    # else:
    #     fitdata.fit_params[0].value *= v_scale ** 2
    #     fitdata.fit_params[0].error *= v_scale ** 2
    #     # fitdata.fit_params[1].value *= v_scale
    #     # fitdata.fit_params[1].error *= v_scale
    # fitdata.print_fit_params()
    #
    # if not fig:
    #     fig = plt.figure()
    #
    # plt.plot(hist_bins, hist, label='z: {} - {}'.format(z_low, z_high))
    # plt.plot(hist_bins, fitdata.fit_y, label="T_e = {t:2.1f}eV".format(t=fitdata.get_param(c.ELEC_TEMP, errors_fl=False)))
    # plt.xlabel(r'Velocity ($m s^{-1}$)')
    # plt.ylabel(r'Normalised f(v)')
    # plt.legend()
    #
    # if show:
    #     plt.show()
    #
    # return fitdata
    pass


def get_histogram_fit(splopter, hist, hist_bins, fitter=None, v_scale=1, plot_fl=False):
    if not fitter:
        fitter = f.GaussianVelElecEvFitter()
        fitter.set_mass_scaler(0.5)

    # estimate values for the initial guess based on the
    grad_fv = np.gradient(hist)
    sm_grad_fv = savgol_filter(grad_fv, 21, 2)
    min = np.argmin(sm_grad_fv)
    max = np.argmax(sm_grad_fv)

    if plot_fl:
        plt.figure()
        plt.plot(sm_grad_fv)
        plt.plot(grad_fv)
        plt.axvline(min)
        plt.axvline(max)
        plt.show()

    v_0_guess = hist_bins[int((max + min) / 2)]
    # print(((hist_bins[min] - hist_bins[max]) * mass[species] * v_scale ** 2 * 0.5) / (2 * n.ELEM_CHARGE))
    t_e_guess = splopter.denormaliser.temperature / (v_scale**2)
    guess = [t_e_guess, v_0_guess]

    return fitter.fit(hist_bins, hist, initial_vals=guess)


def run_multihistogram_extr(z_high=370.0, z_low=70.0, num_samples=11, fig=None, show=None):
    splopter = spl.Splopter('bin/data_local/', 'benchmarking/', 'disttest_fullnogap/', prepare=False)
    # path = 'bin/data_local/benchmarking/disttest_fullnogap/'
    nproc = int(np.squeeze(splopter.tdata.nproc))

    u_pars = {}
    assert isinstance(num_samples, int)
    if num_samples <= 1:
        num_samples = 2
    z_vals = np.linspace(z_low, z_high, num_samples)
    for z in z_vals[:-1]:
        u_pars[z] = np.array([], dtype=np.float64)

    ralpha = (-splopter.tdata.alphayz / 180.0) * np.pi
    rbeta = ((90.0 - splopter.tdata.alphaxz) / 180) * np.pi

    for i in range(nproc):
        num = str(i).zfill(2)
        filename = splopter.tfile_path.replace('.mat', '{}.mat'.format(num))
        p_file = loadmat(filename)

        # [print(key+': ', str(np.shape(array))) for key, array in p_file.items() if '__' not in key]
        for j in range(len(z_vals) - 1):
            print('int(j)', int(j))
            print('z_vals[int(j)]', z_vals[int(j)])
            print('z_vals[int(j+1)]', z_vals[int(j+1)])
            indices = np.where((p_file['z'] > z_vals[int(j)]) & (p_file['z'] <= z_vals[int(j+1)]) & (p_file['stype'] == 2))
            u_pars[z_vals[int(j)]] = np.append(u_pars[z_vals[int(j)]],
                                               (p_file['uy'][indices] * np.cos(ralpha) * np.cos(rbeta))
                                               - (p_file['uz'][indices] * np.sin(ralpha)))

    # u_par = (u_pars['uy'] * np.cos(ralpha) * np.cos(rbeta)) - (u_pars['uz'] * np.sin(ralpha))
    # u_par = (u_pars['ux'] * np.sin(rbeta) * np.cos(ralpha)) + \
    #         (u_pars['uy'] * np.cos(ralpha) * np.cos(rbeta)) - \
    #         (u_pars['uz'] * np.sin(ralpha))

    # sp.io.savemat('test.mat', u_par)
    print('Finished compiling ')

    if not fig:
        fig = plt.figure()

    # plt.hist(u_pars['u_par'], bins=500)
    # plt.title('z_high = {}, z_low = {}'.format(z_high, z_low))
    # run_spice_df_analysis(splopter, fig=fig)

    for z_val, u_par in u_pars.items():
        print(u_par)
        hist, gaps = np.histogram(u_par, bins='auto', density=True)
        plt.plot(gaps[:-1], hist, label='z_low = {}'.format(z_val))

    plt.legend()

    if show:
        plt.show()
        

def run_current_comparison():
    splopter_prep = spl.Splopter('bin/data/', 'benchmarking_sam/', 'disttest_fullnogap/', prepare=False)
    splopter_top = spl.Splopter('bin/data/', 'bms_distruns/', 'disttest_fullnogap_top/', prepare=False)
    splopter_bottom = spl.Splopter('bin/data/', 'bms_distruns/', 'disttest_fullnogap_bottom/', prepare=False)
    splopters = {'prep': splopter_prep, 'top': splopter_top, 'bottom': splopter_bottom}

    currents = {}
    times = {}
    diagnostics = {}
    for name, sp in splopters.items():
        currents[name] = sp.tdata.objectscurrente[0]
        times[name] = sp.tdata.t[1:]
        diagnostics[name] = [diag for diag_name, diag in sp.tdata.diagnostics.items()
                             if 'eHist' in diag_name]

    print(np.shape(diagnostics.items()))

    for name in splopters.keys():
        plt.figure(1)
        plt.plot(times[name], currents[name], label=name)
        plt.legend()

        plt.figure(2)
        for i, diag in enumerate(diagnostics[name]):
            plt.plot(diag, label=name.join(str(i)))
        plt.legend()

    # current_prep = splopter_prep.tdata.objectscurrente
    # time_prep = splopter_prep.tdata.t[1:]
    # print(np.shape(time_prep), np.shape(current_prep))
    # current_top = splopter_top.tdata.objectscurrente
    # time_top = splopter_top.tdata.t[1:]

    # for i in range(len(current_prep)):
    #     plt.figure()
    #     # current_combo = np.concatenate((current_prep[i], current_run0[i]))
    #     plt.plot(time_prep, current_prep[i], label='prep')
    #     plt.plot(time_top, current_top[i], label='top')
    #     # plt.plot(current_combo)
    #     plt.legend()

    plt.figure()
    plt.imshow(splopter_prep.tdata.pot)
    plt.colorbar()

    plt.figure()
    plt.imshow(splopter_top.tdata.pot)
    plt.colorbar()

    plt.figure()
    plt.imshow(splopter_bottom.tdata.pot)
    plt.colorbar()

    print(splopter_prep.tdata.diagnostics.keys())
    print(splopter_top.tdata.diagnostics.keys())

    plt.show()


def run_maxwellian_comparison():
    splopter_sheath = spl.Splopter('bin/data/', 'benchmarking_sam/', 'disttest_fullnogap/', prepare=False)
    splopter_whole = spl.Splopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullnogap/', prepare=False)

    fig = plt.figure()
    run_spice_df_analysis(splopter_sheath, fig=fig)
    run_spice_df_analysis(splopter_whole, fig=fig)
    plt.show()


def get_diag_index(diag_name, parser):
    """Find index of diagnostic within an input file"""
    for i in range(len([section for section in parser.sections() if c.INF_SEC_DIAG in section]) - 1):
        diag_section_name = c.INF_SEC_DIAG + str(i)
        if parser.get(diag_section_name, c.INF_DIAG_NAME).strip('\\\'') == diag_name:
            return i
    print('No diagnostic region found matching "{}"'.format(diag_name))
    return None


def get_hist_index(hist_name, parser):
    """Find the index of diagnostics which are histograms within an input file"""
    count = -1
    for i in range(len([section for section in parser.sections() if c.INF_SEC_DIAG in section]) - 1):
        diag_section_name = c.INF_SEC_DIAG + str(i)
        if parser.getint(diag_section_name, c.INF_DIAG_PROPERTY) == 3:
            count += 1
            if parser.get(diag_section_name, c.INF_DIAG_NAME).strip('\\\'') == hist_name:
                return count
    print('No diagnostic region making histograms found matching "{}"'.format(hist_name))
    return None


def run_spice_df_analysis(splopter, fig=None, show=False):
    assert isinstance(splopter, spl.Splopter)

    splopter.prepare(homogenise=False, make_denormaliser=True)

    tdata = splopter.tdata
    print(splopter.tdata.diagnostics.keys())

    # Get all arrays in the t-file which contain diagnostic histograms and put them into
    hist_names = [hist_name for hist_name in tdata.diagnostics.keys()
                  if 'eHist' in hist_name and any(tdata.diagnostics[hist_name] != 0.0)]
    diagnostic_histograms = {}
    for hist_name in hist_names:
        diagnostic = hist_name[:-2]
        if diagnostic in diagnostic_histograms:
            diagnostic_histograms[diagnostic].append(tdata.diagnostics[hist_name])
        else:
            diagnostic_histograms[diagnostic] = [tdata.diagnostics[hist_name]]

    fvarrays = tdata.fvarrays
    fvbin = tdata.fvbin
    fvperparraycount = tdata.fvperparraycount
    fvlimits = tdata.fvlimits
    histlimits = tdata.histlimits
    current = tdata.objectscurrente[0]
    time = tdata.t[:-1]

    # vx = adata['vxav02']
    # vx2 = adata['vx2av02']
    # vy = adata['vyav02']
    # vy2 = adata['vy2av02']
    # vz = adata['vzav02']
    # vz2 = adata['vz2av02']
    # t1 = adata['temperature01']
    # t2 = adata['temperature02']

    # print('hists', np.shape(diagnostic_histograms))
    # print('fvarrays   ', np.shape(fvarrays), fvarrays)
    print('fvbin   ', np.shape(fvbin), fvbin)
    # print('fvperparraycount   ', np.shape(fvperparraycount), fvperparraycount)
    print('fvlimits   ', np.shape(fvlimits), fvlimits)
    print('histlimits   ', np.shape(histlimits), histlimits)

    if not fig:
        plt.figure()

    for name, data in diagnostic_histograms.items():
        diag_index = get_hist_index(name, splopter.parser)
        print(diag_index)
        for i in range(len(data)):
            hist_x = np.linspace(fvlimits[(diag_index*3)+i][0], fvlimits[(diag_index*3)+i][1], fvbin)
            g_fitter = f.GaussianVelFitter()
            guess = [100.0, 1.0, 10, 100]
            bounds = [
                [0.0, 0.0, -np.inf, 0.0],
                [np.inf, np.inf, np.inf, np.inf]
            ]
            # fit_data = m_fitter.fit(ehist1x, ehist1, guess, bounds=bounds)
            # fit_data.print_fit_params()
            guess_func = g_fitter.fit_function(hist_x, *guess)

            plt.figure(i)
            plt.plot(hist_x, data[i], label=name)
            plt.legend()

    # for name, data in diagnostic_histograms.items():
    #     plt.figure()
    #     for i in range(len(data)):
    #         plt.plot(data[i])

    # plt.plot(*fit_data.get_fit_plottables())
    # plt.plot(ehist1x, guess_func)
    # plt.show()

    #
    # plt.figure()
    # plt.imshow(pot)

    # plt.figure()
    # plt.plot(time, current)

    # plt.figure()
    # for t in np.linspace(0.1, 10, 101):
    #     guess = [t, 0.1, 10]
    #     guess_func = m_fitter.fit_function(ehist1x, *guess)
    #     plt.plot(ehist1x, guess_func)
    #
    # plt.figure()
    # for m in np.linspace(0.1, 10, 101):
    #     guess = [1.0, m, 10]
    #     guess_func = m_fitter.fit_function(ehist1x, *guess)
    #     plt.plot(ehist1x, guess_func)

    if show:
        plt.show()


def draw_potential(splopter=None, t_dict_label=sd.POT, plot_obj_fl=False):
    if not splopter:
        splopter = spl.Splopter('bin/data_local/', 'benchmarking/', 'disttest_fullnogap/', prepare=False)
        # splopter = spl.Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullgap/', prepare=False)

    plasma_parameter = np.flip(splopter.tdata.t_dict[t_dict_label], 0)
    objects_raw = np.flip(splopter.tdata.objectsenum, 0)
    probe_obj_indices = splopter.parser.get_probe_obj_indices()
    objects = np.zeros(np.shape(plasma_parameter))

    wall_indices = np.where(plasma_parameter == 0)
    probe_objs = [np.where(objects_raw == index + 1) for index in probe_obj_indices]

    if plot_obj_fl:
        plt.figure()
        plt.imshow(objects_raw, cmap='Greys')
        plt.colorbar()

    plasma_parameter[wall_indices] = np.NaN
    for probe_obj in probe_objs:
        plasma_parameter[probe_obj] = np.NaN
    objects[wall_indices] = 3.0
    for probe_obj in probe_objs:
        objects[probe_obj] = 1.5

    plt.figure()
    plt.imshow(objects, cmap='Greys', extent=[0, len(plasma_parameter[0]) / 2, 0, len(plasma_parameter) / 2])

    ax = plt.gca()
    im = ax.imshow(plasma_parameter, extent=[0, len(plasma_parameter[0]) / 2, 0, len(plasma_parameter) / 2], interpolation=None)
    plt.xlabel(r'y / $\lambda_D$', fontsize=15)
    plt.ylabel(r'z / $\lambda_D$', fontsize=15)
    # plt.title('Electrostatic potential for a flush mounted probe', fontsize=20)
    plt.quiver([200], [200], splopter.tdata.by, splopter.tdata.bz, scale=5)
    plt.colorbar(im, fraction=0.035, pad=0.04)
    plt.show()


def test():
    flpt = spl.Splopter('bin/data_local/', 'benchmarking/', 'nogap/')

    vf = flpt.get_vf()
    phi = flpt.get_plasma_potential()

    mu = flpt.denormaliser.mu
    const = np.log(0.6 * np.sqrt((2 * np.pi) / mu))
    temperature = (vf - phi)/const
    print(temperature)


def test2():
    flpt = spl.Splopter('bin/data_local/', 'benchmarking/', 'nogap/')

    # plt.figure()

    print(flpt.tdata.diagnostics.keys())
    e_hist_before = flpt.tdata.diagnostics['eHistx1']
    i_hist_before = flpt.tdata.diagnostics['iHistx1']
    plt.plot(e_hist_before, label='eHistBefore')
    plt.plot(i_hist_before,label='iHistBefore')

    flpt.denormalise()

    e_hist_after = flpt.tdata.diagnostics['eHistx1']
    i_hist_after = flpt.tdata.diagnostics['iHistx1']
    # plt.plot(e_hist_after, label='eHistAfter')
    # plt.plot(i_hist_after, label='iHistAfter')
    plt.legend()

    plt.show()


_DEFAULT_HIST_REGIONS = {
    'Sheath': [75, 90, 0, 822],
    'Lower-mid': [165, 180, 0, 822],
    'Upper-mid': [255, 270, 0, 822],
    'Injection': [355, 370, 0, 822]
}


def run_multi_hist_analysis(splopter=None, species=2, fitter=None, show_fl=False, regions=None):
    if not fitter:
        fitter = f.GaussianVelElecEvFitter()
    if not regions or not isinstance(regions, dict):
        regions = _DEFAULT_HIST_REGIONS

    v_scale = 1000
    raw_hists = splopter.extract_histograms(list(regions.values()), denormalise=True, v_scale=v_scale, species=species)
    fitted_hists = {}

    for i, (bins, histogram) in enumerate(raw_hists):
        # Retrieve label and coords from regions dictionary
        region_label = list(regions.keys())[i]
        region_coords = list(regions.values())[i]

        # Fit histogram with fitter and populate the fitted_hists dictionary
        fitdata = get_histogram_fit(splopter, histogram, bins, fitter=fitter, v_scale=v_scale)

        fitdata.fit_params[0].value *= v_scale ** 2
        fitdata.fit_params[0].error *= v_scale ** 2
        # fitdata.fit_params[1].value *= v_scale
        # fitdata.fit_params[1].error *= v_scale
        fitdata.print_fit_params()

        dict_label = r'{label} - $T_e$ = {t:2.1f}eV'.format(label=region_label,
                                                            t=fitdata.get_param(c.ELEC_TEMP, errors_fl=False))
        fitted_hists[dict_label] = fitdata

        plt.figure()
        plt.plot(bins, histogram, label='z: {} - {}'.format(region_coords[0], region_coords[1]))
        plt.plot(bins, fitdata.fit_y, label="T_e = {t:2.1f}eV".format(t=fitdata.get_param(c.ELEC_TEMP, errors_fl=False)))
        plt.xlabel(r'Velocity ($m s^{-1}$)')
        plt.ylabel(r'Normalised f(v)')
        plt.legend()

    plt.figure()
    plt.axvline(linestyle='--', color='black')
    for region_label, histogram in fitted_hists.items():
        plt.plot(histogram.raw_x, histogram.raw_y, label=region_label)
    plt.legend()
    plt.xlabel(r'Velocity ($km s^{-1}$)')
    plt.ylabel(r'Normalised f(v)')

    # print(fitdata_sheath.get_param(c.ELEC_TEMP, errors_fl=False) / fitdata_inj.get_param(c.ELEC_TEMP, errors_fl=False))
    if show_fl:
        plt.show()


def injection_dist_function(gauss_fl=True, show_fl=True):
    splopter = spl.Splopter('bin/data_local/', 'tests/', 'injtestperp_halfnogap2/')
    # splopter = spl.Flopter('bin/data_local/', 'tests/', 'injtest_halfnogap1/')
    splopter.prepare(homogenise=False)
    v_scale = 1000

    i_inj_path = splopter.afile_path.replace('.mat', '.i_inj')
    e_inj_path = splopter.afile_path.replace('.mat', '.e_inj')
    i_inj = -splopter.denormaliser(np.loadtxt(i_inj_path), c.CONV_VELOCITY) / v_scale
    e_inj = -splopter.denormaliser(np.loadtxt(e_inj_path), c.CONV_VELOCITY) / v_scale

    # runpath = 'data/'
    # i_inj_path = runpath + 'injtest_halfnogap.i_inj'
    # e_inj_path = runpath + 'injtest_halfnogap.e_inj'
    # i_inj = np.loadtxt(i_inj_path)
    # e_inj = np.loadtxt(e_inj_path)

    i_hist, i_gaps = np.histogram(i_inj, density=True, bins='auto')
    i_bins = (i_gaps[:-1] + i_gaps[1:]) / 2
    e_hist, e_gaps = np.histogram(e_inj, density=True, bins='auto')
    e_bins = (e_gaps[:-1] + e_gaps[1:]) / 2

    t_e_guess = splopter.denormaliser.temperature / (v_scale ** 2)
    guess = [t_e_guess, 1]

    if gauss_fl:
        i_fitter = f.GaussianVelIonEvFitter(mu=splopter.denormaliser.mu)
        i_fdata = get_histogram_fit(splopter, i_hist, i_bins, i_fitter, v_scale=v_scale)
    else:
        i_fitter = f.MaxwellianVelFitter(mu=splopter.denormaliser.mu)
        i_fdata = i_fitter.fit(i_bins, i_hist, initial_vals=guess)
    i_fdata.print_fit_params()

    if gauss_fl:
        e_fitter = f.GaussianVelElecEvFitter()
        e_fdata = get_histogram_fit(splopter, e_hist, e_bins, e_fitter, v_scale=v_scale)
    else:
        e_fitter = f.MaxwellianVelFitter(mu=1)
        e_fdata = e_fitter.fit(e_bins, e_hist, initial_vals=guess)
    e_fdata.print_fit_params()

    plt.figure()
    plt.plot(i_bins, i_hist)
    plt.plot(i_bins, i_fdata.fit_y)
    plt.title('Ion injection DF')

    plt.figure()
    plt.plot(e_bins, e_hist)
    plt.plot(e_bins, e_fdata.fit_y)
    plt.title('Electron injection DF')

    if show_fl:
        plt.show()


def extract_density(splopter):
    nproc = int(np.squeeze(splopter.tdata.nproc))
    print(splopter.afile.keys())
    for i in range(nproc):
        num = str(i).zfill(2)
        filename = splopter.tfile_path.replace('.mat', '{}.mat'.format(num))
        p_file = loadmat(filename)
        if len(p_file.keys()) != 13:
            print(i)

    plt.figure()
    plt.imshow(np.flip(splopter.afile['dens01'], 0))
    plt.colorbar()

    plt.figure()
    plt.imshow(np.flip(splopter.afile['temperature01'], 0))
    plt.colorbar()

    for sp in [1, 2]:
        for proc_num in range(32):
            print(sp, proc_num, int(float(sp)/(proc_num + 1)) * (proc_num + 1))

    plt.figure()
    # plt.imshow(np.flip(p_file['temperature01'], 0))
    # plt.colorbar()

    for label, data in splopter.afile.items():
        if isinstance(data, np.ndarray):
            shape = np.shape(data)
        else:
            shape = 1
        print(label, shape)

    plt.show()


def compare_larmor_radii(files, mount='bin/data/', folder='benchmarking_mass/'):
    import gc
    splopters = {}
    for file in files:
        sp = spl.Splopter(mount, folder, file, prepare=True)
        print(sp.tdata.diagnostics.keys())

        print(len(sp.tdata.diagnostics['eTrackery']), np.size(sp.tdata.diagnostics['iTrackery']))

        print('Tracking Ions')
        i_splits = np.where(np.abs(np.diff(sp.tdata.diagnostics['iTrackery'], axis=0)) > 0.5)[0] + 1
        iy_tracks = np.split(sp.tdata.diagnostics['iTrackery'], i_splits)
        iz_tracks = np.split(sp.tdata.diagnostics['iTrackerz'], i_splits)

        iy_tracks.sort(key=len, reverse=True)
        iz_tracks.sort(key=len, reverse=True)

        # print(i_splits.shape, len(iy_tracks), iy_tracks[200])
        # for i, t in enumerate(iy_tracks):
        #     print('{}) {}'.format(i, len(t)))

        i_tracks = zip(iy_tracks[:200], iz_tracks[:200])

        print('Tracking Electrons')
        # Find positions within tracking array where target seems to change by selecting elements with a large jump
        e_splits = np.where(np.abs(np.diff(sp.tdata.diagnostics['eTrackery'], axis=0)) > 0.5)[0] + 1

        # Split tracking diagnostic array into different tracks using points found above
        ey_tracks = np.split(sp.tdata.diagnostics['eTrackery'], e_splits)
        ez_tracks = np.split(sp.tdata.diagnostics['eTrackerz'], e_splits)

        # Sort the tracks in descending order by length
        ey_tracks.sort(key=len, reverse=True)
        ez_tracks.sort(key=len, reverse=True)

        # Create a zip of the tracks individual components
        e_tracks = zip(ey_tracks[:200], ez_tracks[:200])

        # 'iTrackery', 'iTrackerz', 'iTrackerux', 'iTrackeruy', 'iTrackeruz',
        # 'eTrackery', 'eTrackerz', 'eTrackerux', 'eTrackeruy', 'eTrackeruz'

        print('Lengths. ion: {}, electron: {}'.format(len(i_splits), len(e_splits)))

        print('Plotting 1')
        plt.figure()
        for y_track, z_track in i_tracks:
            plt.plot(y_track, z_track)
        plt.title(file + ' - Ions')
        plt.xlabel('Timestep')
        plt.ylabel(r'Z Position ($\lambda_D$)')

        plt.figure()
        print('Plotting 2')
        for y_track, z_track in e_tracks:
            plt.plot(y_track, z_track)
        plt.title(file + ' - Electrons')
        plt.xlabel('Timestep')
        plt.ylabel(r'Z Position ($\lambda_D$)')

        print('...finished plotting')

        # plt.figure()
        # # plt.plot(sp.tdata.diagnostics['iTrackery'], sp.tdata.diagnostics['iTrackerz'], 'x', label='Ion track')
        # plt.plot(sp.tdata.diagnostics['eTrackery'], sp.tdata.diagnostics['eTrackerz'], 'x', label='Electron track')
        # plt.legend()
        # plt.title(file)
        # plt.xlabel(r'Y Position ($\lambda_D$)')
        # plt.ylabel(r'Z Position ($\lambda_D$)')

        # deallocate the splopter object

        del sp
        gc.collect()

    plt.show()


if __name__ == '__main__':
    # run_param_scan()
    # run_maxwellian_comparison()
    # run_current_comparison()
    # test2()
    # splopter = spl.Splopter('bin/data_local/', 'benchmarking/', 'disttest_fullnogap/', prepare=False)
    # splopter = spl.Flopter('bin/data_local/', 'benchmarking/', 'gap/')

    # splopter = spl.Flopter('bin/data/', 'tests/', 'nproctest_fullnogap/', prepare=True)
    # splopter = spl.Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe2_fullgap/', prepare=True)
    # splopter = spl.Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe2_fullnogap/', prepare=True)
    # splopter = spl.Flopter('bin/data/', 'angledtip/', 'angledtiptest/', prepare=False)
    # splopter = spl.Flopter('bin/data/', 'angledtip/', 'angledtiptest1/', prepare=False)
    # splopter = spl.Flopter('bin/data/', 'test/', 'floatingpottest/', prepare=False)

    # Radii comparison files
    # comparison_files = ['benchmark_ms_g_hg/', 'massrationtest_ms_g_hg/', 'ionmasstest_ms_g_hg/',
    #                     'speciesmasstest_ms_g_hg/']
    # compare_larmor_radii(comparison_files)

    # splopter = spl.Splopter('bin/data/', 'benchmarking_mass/', 'benchmark_ms_g_hg/', prepare=True)
    # splopter = spl.Splopter('bin/data/', 'benchmarking_mass/', 'massrationtest_ms_g_hg/', prepare=True)
    # splopter = spl.Splopter('bin/data/', 'benchmarking_mass/', 'ionmasstest_ms_g_hg/', prepare=True)
    # splopter = spl.Splopter('bin/data/', 'benchmarking_mass/', 'speciesmasstest_ms_g_hg/', prepare=True)
    splopter = spl.Splopter('bin/data/', 'magnum/', 'sprobe_1e19_nopadding1/')
    # splopter = spl.Splopter('bin/data/', 'magnum/', 'sprobe_1e19_nopadding1/')

    # splopter.plot_2d_variable(show_fl=False)

    # splopter.analyse_iv(show_fl=True)
    # splopter.plot_1d_variable(variable_label=c.DIAG_WALL_POT, time_dep_fl=True, diagnostic_fl=True)
    # splopter.plot_1d_variable(variable_label=sd.NZ)

    # extract_density(splopter)
    splopter.prepare(homogenise=True, make_denormaliser=False)
    regions = splopter.parser.get_hist_diag_regions(2)
    # run_multi_hist_analysis(regions=regions, splopter=splopter, show_fl=True, species=2)
    # run_gap_nogap_comparison()

    # run_multi_hist_analysis(splopter=splopter, fitter=f.GaussianVelElecEvFitter(), show_fl=False)
    # draw_potential()

    # run_histogram_extr(z_high=100, z_low=70, show=True, species=1, fit_guess=[6, 15000],
    #                    fitter=f.Gaussian1DIonEvFitter(mu=450))
    # run_histogram_extr(z_high=370, z_low=340, show=True, fig=fig)
    # draw_potential()

    # injection_dist_function(show_fl=False)
    # injection_dist_function(gauss_fl=False)



import flopter as fl
from fitters import Gaussian1DFitter, Maxwellian3Fitter, SimpleIVFitter, IonCurrentSEFitter
import constants as c
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from scipy.io import loadmat


def run_param_scan():
    flopter_gap = fl.Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullgap/')
    # flopter_nogap = Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullnogap/')

    ivdata_g = flopter_gap.trim(trim_end=0.8)
    # ivdata_ng = flopter_nogap.trim()

    fig = plt.figure()
    flopter_gap.plot_iv(iv_data=ivdata_g, fig=fig, plot_tot=True, label='Gap')
    n_params = 4
    params = [[]]*n_params
    errors = [[]]*n_params
    trim_space = np.linspace(0.4, 0.5, 11)
    print(params)
    for trim_end in trim_space:
        ivdata = flopter_gap.trim(trim_end=trim_end)
        ivfitdata = flopter_gap.fit(ivdata)
        flopter_gap.plot_f_fit(ivfitdata, fig=fig, plot_raw=False, plot_vf=False, label=str(trim_end))
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
    flopter_gap = fl.Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullgap/')
    flopter_nogap = fl.Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullnogap/')

    ivdata_g = flopter_gap.trim()
    ivdata_ng = flopter_nogap.trim()

    ivdata_g2 = flopter_gap.trim(trim_end=0.5)
    ivdata_ng2 = flopter_nogap.trim(trim_end=0.5)

    ifit_g = flopter_gap.fit(ivdata_g, IonCurrentSEFitter(), print_fl=True)
    ifit_ng = flopter_nogap.fit(ivdata_ng, IonCurrentSEFitter(), print_fl=True)

    ffit_g2 = flopter_gap.fit(ivdata_g2, print_fl=True)
    ffit_ng2 = flopter_nogap.fit(ivdata_ng2, print_fl=True)

    fig1 = plt.figure()
    flopter_gap.plot_iv(fig=fig1, plot_tot=True, label='Gap')
    flopter_nogap.plot_iv(fig=fig1, plot_vf=True, plot_tot=True, label='No Gap')

    fig2 = plt.figure()
    flopter_gap.plot_f_fit(ffit_g2, fig=fig2, label='Gap ', plot_vf=False)
    flopter_nogap.plot_f_fit(ffit_ng2, fig=fig2, label='No Gap ')

    fig3 = plt.figure()
    flopter_gap.plot_i_fit(ifit_g, fig=fig3, label='Gap ')
    flopter_nogap.plot_i_fit(ifit_ng, fig=fig3, label='No Gap ')
    plt.legend()
    plt.show()


def run_histogram_extr(z_high=370.0, z_low=70.0, fig=None, show=False, normalise_v=True, species=2):
    flopter = fl.Flopter('bin/data_local/', 'benchmarking/', 'disttest_fullnogap/', prepare=False)
    flopter.prepare(denormalise=True, homogenise=False)
    # path = 'bin/data_local/benchmarking/disttest_fullnogap/'
    nproc = int(np.squeeze(flopter.tdata.nproc))

    u_par = np.array([])
    ralpha = (-flopter.tdata.alphayz / 180.0) * 3.141591
    rbeta = ((90.0 - flopter.tdata.alphaxz) / 180) * 3.14159

    for i in range(nproc):
        num = str(i).zfill(2)
        filename = flopter.tfile_path.replace('.mat', '{}.mat'.format(num))
        p_file = loadmat(filename)

        # [print(key+': ', str(np.shape(array))) for key, array in p_file.items() if '__' not in key]
        indices = np.where((p_file['z'] > z_low) & (p_file['z'] <= z_high) & (p_file['stype'] == species))
        u_par = np.append(u_par, (p_file['uy'][indices] * np.cos(ralpha) * np.cos(rbeta))
                          - (p_file['uz'][indices] * np.sin(ralpha)))

    print('Finished compiling ')
    if normalise_v:
        u_par = -flopter.denormaliser(u_par, c.CONV_VELOCITY)

    if not fig:
        fig = plt.figure()

    hist, gaps = np.histogram(u_par, bins='auto', density=True)

    # guess = [5000.0, 180000.0]
    guess = [6000, 34000]
    fitter = Gaussian1DFitter()
    fitdata = fitter.fit(gaps[:-1], hist, initial_vals=guess)
    fitdata.print_fit_params()
    guess_data = fitter.fit_function(gaps[:-1], *guess)

    plt.plot(gaps[:-1], hist, label='z: {} - {}'.format(z_low, z_high))
    plt.plot(gaps[:-1], fitdata.fit_y, label=fitter.name)
    plt.plot(gaps[:-1], guess_data, label='Initial Guess')
    plt.legend()

    plt.figure(2)
    plt.plot(gaps[:-1], fitdata.get_residual())

    if show:
        plt.show()


def run_multihistogram_extr(z_high=370.0, z_low=70.0, num_samples=11, fig=None, show=None):
    flopter = fl.Flopter('bin/data_local/', 'benchmarking/', 'disttest_fullnogap/', prepare=False)
    # path = 'bin/data_local/benchmarking/disttest_fullnogap/'
    nproc = int(np.squeeze(flopter.tdata.nproc))

    u_pars = {}
    assert isinstance(num_samples, int)
    if num_samples <= 1:
        num_samples = 2
    z_vals = np.linspace(z_low, z_high, num_samples)
    for z in z_vals[:-1]:
        u_pars[z] = np.array([], dtype=np.float64)

    ralpha = (-flopter.tdata.alphayz / 180.0) * 3.141591
    rbeta = ((90.0 - flopter.tdata.alphaxz) / 180) * 3.14159

    for i in range(nproc):
        num = str(i).zfill(2)
        filename = flopter.tfile_path.replace('.mat', '{}.mat'.format(num))
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
    # run_spice_df_analysis(flopter, fig=fig)

    for z_val, u_par in u_pars.items():
        print(u_par)
        hist, gaps = np.histogram(u_par, bins='auto', density=True)
        plt.plot(gaps[:-1], hist, label='z_low = {}'.format(z_val))

    plt.legend()

    if show:
        plt.show()
        

def run_current_comparison():
    flopter_prep = fl.Flopter('bin/data/', 'benchmarking_sam/', 'disttest_fullnogap/', prepare=False)
    flopter_top = fl.Flopter('bin/data/', 'bms_distruns/', 'disttest_fullnogap_top/', prepare=False)
    flopter_bottom = fl.Flopter('bin/data/', 'bms_distruns/', 'disttest_fullnogap_bottom/', prepare=False)
    flopters = {'prep': flopter_prep, 'top': flopter_top, 'bottom': flopter_bottom}

    currents = {}
    times = {}
    diagnostics = {}
    for name, f in flopters.items():
        currents[name] = f.tdata.objectscurrente[0]
        times[name] = f.tdata.t[1:]
        diagnostics[name] = [diag for diag_name, diag in f.tdata.diagnostics.items()
                             if 'eHist' in diag_name]

    print(np.shape(diagnostics.items()))

    for name in flopters.keys():
        plt.figure(1)
        plt.plot(times[name], currents[name], label=name)
        plt.legend()

        plt.figure(2)
        for i, diag in enumerate(diagnostics[name]):
            plt.plot(diag, label=name.join(str(i)))
        plt.legend()

    # current_prep = flopter_prep.tdata.objectscurrente
    # time_prep = flopter_prep.tdata.t[1:]
    # print(np.shape(time_prep), np.shape(current_prep))
    # current_top = flopter_top.tdata.objectscurrente
    # time_top = flopter_top.tdata.t[1:]

    # for i in range(len(current_prep)):
    #     plt.figure()
    #     # current_combo = np.concatenate((current_prep[i], current_run0[i]))
    #     plt.plot(time_prep, current_prep[i], label='prep')
    #     plt.plot(time_top, current_top[i], label='top')
    #     # plt.plot(current_combo)
    #     plt.legend()

    plt.figure()
    plt.imshow(flopter_prep.tdata.pot)
    plt.colorbar()

    plt.figure()
    plt.imshow(flopter_top.tdata.pot)
    plt.colorbar()

    plt.figure()
    plt.imshow(flopter_bottom.tdata.pot)
    plt.colorbar()

    print(flopter_prep.tdata.diagnostics.keys())
    print(flopter_top.tdata.diagnostics.keys())

    plt.show()


def run_maxwellian_comparison():
    flopter_sheath = fl.Flopter('bin/data/', 'benchmarking_sam/', 'disttest_fullnogap/', prepare=False)
    flopter_whole = fl.Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullnogap/', prepare=False)

    fig = plt.figure()
    run_spice_df_analysis(flopter_sheath, fig=fig)
    run_spice_df_analysis(flopter_whole, fig=fig)
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


def run_spice_df_analysis(flopter, fig=None, show=False):
    assert isinstance(flopter, fl.Flopter)

    flopter.prepare(homogenise=False, denormalise=True)

    tdata = flopter.tdata
    print(flopter.tdata.diagnostics.keys())

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
        diag_index = get_hist_index(name, flopter.parser)
        print(diag_index)
        for i in range(len(data)):
            hist_x = np.linspace(fvlimits[(diag_index*3)+i][0], fvlimits[(diag_index*3)+i][1], fvbin)
            g_fitter = Gaussian1DFitter()
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

def draw_potential():
    # flopter = fl.Flopter('bin/data/', 'benchmarking_sam/', 'disttest_fullnogap/', prepare=False)
    flopter = fl.Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullgap/', prepare=False)

    pot = np.flip(flopter.tdata.pot, 0)
    objects = np.zeros(np.shape(pot))

    wall_indices = np.where(pot == 0)
    probe_indices = np.where(pot > 10.0)

    pot[wall_indices] = np.NaN
    pot[probe_indices] = np.NaN
    objects[wall_indices] = 3.0
    objects[probe_indices] = 1.5

    plt.figure()
    plt.imshow(objects, cmap='Greys', extent=[0, len(pot[0]) / 2, 0, len(pot) / 2])

    # plt.figure()
    # plt.contour(edges, extent=[0, len(pot[0]) / 2, 0, len(pot) / 2], linewidths=0.5, cmap='Greys')
    # plt.axhline(70, linewidth=1, color='black')
    ax = plt.gca()
    im = ax.imshow(pot, extent=[0, len(pot[0]) / 2, 0, len(pot) / 2], interpolation=None)
    plt.xlabel(r'y / $\lambda_D$', fontsize=15)
    plt.ylabel(r'z / $\lambda_D$', fontsize=15)
    plt.title('Electrostatic potential for a flush mounted probe', fontsize=20)
    plt.colorbar(im, fraction=0.035, pad=0.04)
    plt.show()


def test():
    flpt = fl.Flopter('bin/data_local/', 'benchmarking/', 'nogap/')

    vf = flpt.get_vf()
    phi = flpt.get_plasma_potential()

    mu = flpt.denormaliser.mu
    const = np.log(0.6 * np.sqrt((2 * np.pi) / mu))
    temperature = (vf - phi)/const
    print(temperature)


def test2():
    flpt = fl.Flopter('bin/data_local/', 'benchmarking/', 'nogap/')

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


def run_multi_hist_analysis():
    plt.figure()
    plt.axvline()
    run_multihistogram_extr(z_high=370, z_low=340)
    run_multihistogram_extr(z_high=280, z_low=250)
    run_multihistogram_extr(z_high=190, z_low=160)
    run_multihistogram_extr(z_high=100, z_low=70)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # run_gap_nogap_comparison()
    # run_param_scan()
    # run_maxwellian_comparison()
    # run_current_comparison()
    # test2()
    # run_multi_hist_analysis()
    # fig = plt.figure()
    # run_histogram_extr(z_high=100, z_low=70, show=True, fig=fig)
    # run_histogram_extr(z_high=370, z_low=340, show=True, fig=fig)
    # run_multihistogram_extr()
    draw_potential()

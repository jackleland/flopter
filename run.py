import flopter as fl
from fitters import Gaussian1DFitter, Maxwellian3Fitter, SimpleIVFitter, IonCurrentSEFitter
import constants as c
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
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


def run_histogram_extraction():
    flopter = fl.Flopter('bin/data/', 'benchmarking_sam/', 'disttest_fullnogap/', prepare=False)
    path = 'bin/data/benchmarking_sam/disttest_fullnogap/'
    nproc = int(np.squeeze(flopter.tdata.nproc))

    particles = {
        'uy': np.array([], dtype=np.float64),
        'uz': np.array([], dtype=np.float64),
        'ux': np.array([], dtype=np.float64)
    }
    z_high = 374.0
    z_low = 70.0
    for i in range(nproc):
        num = str(i).zfill(2)
        filename = flopter.tfile_path.replace('.mat', '{}.mat'.format(num))
        p_file = loadmat(filename)

        # [print(key+': ', str(np.shape(array))) for key, array in p_file.items() if '__' not in key]
        indices = np.where((p_file['z'] > z_low) & (p_file['z'] <= z_high))
        for label in particles.keys():
            p = p_file[label][indices]
            particles[label] = np.append(particles[label], p)
            print(np.shape(particles[label]))

    ralpha = -flopter.tdata.alphayz / 180.0 * 3.141591
    rbeta = (90.0 - flopter.tdata.alphaxz) / 180 * 3.14159

    u_par = (particles['ux'] * np.sin(rbeta) * np.cos(ralpha)) + \
            (particles['uy'] * np.cos(ralpha) * np.cos(rbeta)) - \
            (particles['uz'] * np.sin(ralpha))
    plt.hist(u_par, bins=300)
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
    run_maxwellian_fit(flopter_sheath, fig=fig)
    run_maxwellian_fit(flopter_whole, fig=fig)
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


def run_maxwellian_fit(flopter, fig=None, show=False):
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


if __name__ == '__main__':
    # run_gap_nogap_comparison()
    # run_param_scan()
    # run_maxwellian_comparison()
    # run_current_comparison()
    run_histogram_extraction()
    # test2()


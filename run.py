import flopter as fl
from fitters import Gaussian1DFitter, Maxwellian3Fitter, SimpleIVFitter, IonCurrentSEFitter
from normalisation import _ELECTRON_MASS as m_e
import matplotlib.pyplot as plt
import numpy as np


def run_param_scan():
    flopter_gap = fl.Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullgap/')
    # flopter_nogap = Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullnogap/', run_name='prebp', ext_run_name='prebpro')

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
    flopter_nogap = fl.Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullnogap/', run_name='prebp', ext_run_name='prebpro')

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


def run_maxwellian_fit():
    f_gap = fl.Flopter('bin/data/', 'benchmarking_sam/', 'disttest_fullnogap/',
                          run_name='distte', ext_run_name='disttest', prepare=False)
    # f_gap = fl.Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullnogap/',
    #                    run_name='prebp', ext_run_name='prebpro', prepare=False)
    f_gap.prepare(homogenise=False, denormalise=False)

    tdata = f_gap.tfile
    adata = f_gap.afile
    print(f_gap.tfile)
    print(f_gap.afile.keys())

    # Get all arrays in the t-file which contain diagnostic histograms and put them into
    hist_names = [hist_name for hist_name in tdata.diagnostics.keys() if 'Hist' in hist_name]
    diagnostic_histograms = {}
    for hist_name in hist_names:
        diagnostic = hist_name[:-2]
        if diagnostic in diagnostic_histograms:
            diagnostic_histograms[diagnostic].append(tdata.diagnostics[hist_name])
        else:
            diagnostic_histograms[diagnostic] = [tdata.diagnostics[hist_name]]

    ehist1 = f_gap.tfile.diagnostics['eHistSheathx1'][:, 0]
    ehist2 = f_gap.tfile.diagnostics['eHistSheathx2'][:, 0]
    ehist3 = f_gap.tfile.diagnostics['eHistSheathx3']
    ehist = np.sqrt(ehist1**2 + ehist2**2)
    pot = f_gap.tfile.pot
    temp = f_gap.tfile.temp

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
    # temp = tdata['Temp']

    print('hists', np.shape(diagnostic_histograms))
    print('ehist', np.shape(ehist))
    print('ehist2', np.shape(ehist2))
    print('fvarrays   ', np.shape(fvarrays), fvarrays)
    print('fvbin   ', np.shape(fvbin), fvbin)
    print('fvperparraycount   ', np.shape(fvperparraycount), fvperparraycount)
    print('fvlimits   ', np.shape(fvlimits), fvlimits)
    print('histlimits   ', np.shape(histlimits), histlimits)

    ehist1x = np.linspace(fvlimits[3][0], fvlimits[3][1], fvbin)
    ehist2x = np.linspace(fvlimits[4][0], fvlimits[4][1], fvbin)
    ehist3x = np.linspace(fvlimits[5][0], fvlimits[5][1], fvbin)

    # ehist2 = f_gap.trim_generic(ehist2, trim_beg=0.49)
    # ehist2x = f_gap.trim_generic(ehist2x, trim_beg=0.49)

    m_fitter = Gaussian1DFitter()
    guess = [100.0, 1.0, 10, 100]
    bounds = [
        [0.0,       0.0,    -np.inf,    0.0],
        [np.inf,    np.inf, np.inf,     np.inf]
    ]
    # fit_data = m_fitter.fit(ehist1x, ehist1, guess, bounds=bounds)
    # fit_data.print_fit_params()
    guess_func = m_fitter.fit_function(ehist1x, *guess)

    plt.figure()
    plt.plot(ehist2)
    plt.plot(ehist1)

    # for name, data in diagnostic_histograms.items():
    #     for i in range(len(data)):
    #         plt.figure(i)
    #         plt.plot(data[i], label=name)
    #         plt.legend()

    # for name, data in diagnostic_histograms.items():
    #     plt.figure()
    #     for i in range(len(data)):
    #         plt.plot(data[i])
    # plt.plot(*fit_data.get_fit_plottables())
    # plt.plot(ehist1x, guess_func)
    # plt.show()

    # plt.figure()
    # plt.imshow(temp)
    #
    # plt.figure()
    # plt.imshow(pot)

    plt.figure()
    plt.plot(time, current)

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

    plt.show()


def test():
    flpt = fl.Flopter('bin/data_local/', 'benchmarking/', 'nogap/', run_name='prebp', ext_run_name='prebpro')

    vf = flpt.get_vf()
    phi = flpt.get_plasma_potential()

    mu = flpt.denormaliser.mu
    const = np.log(0.6 * np.sqrt((2 * np.pi) / mu))
    temperature = (vf - phi)/const
    print(temperature)


if __name__ == '__main__':
    # run_gap_nogap_comparison()
    # run_param_scan()
    run_maxwellian_fit()
    # test()


import numpy as np
import matplotlib.pyplot as plt
import flopter.core.lputils as lpu
import flopter.core.fitters as fts

DECOMP_MODES = [
    None,
    lpu.decompose_sheath_exp_param,
    lpu.decompose_new_sheath_exp_param,
    lpu.decompose_2d_box_sheath_exp_param,
    lpu.decompose_alt_new_sheath_exp_param
]
MODE_LABELS = [
    None,
    (r'$\cot{\theta}$', r'$a\sin^{1/2}{\theta}\cdot[L + g]$'),
    (r'$\tan{\theta} + 2\tan{\theta_p}$', r'$a\sin^{1/2}{\theta}[(L+g)\tan{\theta} + L\tan{\theta_p} - d_{\perp}]$'),
    (r'$\cot{\theta_{tot}}$', r'$a\sin^{1/2}{\theta_{tot}}L_{exp,0}$'),
    (r'$\frac{\cos{\theta}\cos{\theta_p}}{\sin{\theta_{tot}}}$',
     r'$a\sin^{1/2}{\theta}[L + \frac{\cos{\theta}}{\sin{\theta_{tot}}}(g\sin{\theta} - d_{\perp}\cos{\theta})]$'),
]
MODE_CONSTANTS = [
    None,
    ('y_0', 'm'),
    ('m', 'y_0'),
    ('y_0', 'm'),
    ('y_0', 'm'),
]

DEFAULT_G = 60
DEFAULT_L = 300
DEFAULT_DEBYE = 5e-3 / DEFAULT_L


def perform_decomp(ds, sheath_label='ion_a', mode=1, length=DEFAULT_L, gap=DEFAULT_G,
                   lambda_d=DEFAULT_DEBYE):
    if mode in [1, 2, 3, 4]:
        print(length, gap, ds['recession'].values / lambda_d, np.degrees(ds['theta_p_rads'].values))
        x, y = DECOMP_MODES[mode](ds[sheath_label].values, ds['theta_rads'].values, length, gap,
                                  ds['recession'].values / lambda_d, ds['theta_p_rads'].values)
    else:
        raise ValueError('Mode must be 1, 2, 3, or 4')
    return x, y


def plot_decomp(ds, sheath_label='ion_a', mode=1, fit_fl=True, ax=None, kwargs_for_plot=None,
                kwargs_for_fitplot=None, colour='r', plot_label=True, length=DEFAULT_L, gap=DEFAULT_G):
    sheath_error_label = sheath_label.replace('_a', '_d_a')

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if kwargs_for_plot is None:
        kwargs_for_plot = {}
    if kwargs_for_fitplot is None:
        kwargs_for_fitplot = {}

    x, y = perform_decomp(ds, sheath_label=sheath_label, mode=mode, length=length, gap=gap)
    non_nan_y = ~np.isnan(y)
    x = x[non_nan_y]
    y = y[non_nan_y]

    yerr = ds[sheath_error_label].values[non_nan_y] * y

    if plot_label is True:
        plot_label = sheath_label

    ax.errorbar(x, y, yerr=yerr, color=colour, label=plot_label, **kwargs_for_plot)

    sl_fitter = fts.StraightLineFitter()
    fit_data = sl_fitter.fit(x, y)
    if fit_fl:
        c1, c2 = MODE_CONSTANTS[mode]
        fit_label = r'$c_1$ = {:.2g}, $c_2$ = {:.2g}'.format(fit_data.get_param(c1), fit_data.get_param(c2))
        ax.plot(*fit_data.get_fit_plottables(), color=colour, label=fit_label,
                **kwargs_for_fitplot)

    ax.set_xlabel(MODE_LABELS[mode][0])
    ax.set_ylabel(MODE_LABELS[mode][1])
    ax.legend()

    return ax, fit_data

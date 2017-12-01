from scipy.io import loadmat
from scipy import interpolate
from scipy.optimize import curve_fit
from normalisation import Denormaliser, TIME, LENGTH, POTENTIAL, CURRENT
from homogeniser import Spice2Homogeniser, IVData
from inputparser import InputParser
import matplotlib.pyplot as plt
import numpy as np
import os


def iv_characteristic_function(v, *parameters):
    I_0 = parameters[0]
    a = parameters[1]
    v_f = parameters[2]
    T_e = parameters[3]
    V = (v_f - v)/T_e
    return I_0 * (1 - np.exp(-V) + (a * np.float_power(np.absolute(V), 0.75)))


def simple_iv_characteristic_function(v, *parameters):
    I_0 = parameters[0]
    v_f = parameters[1]
    T_e = parameters[2]
    V = (v_f - v) / T_e
    return I_0 * (1 - np.exp(-V))


def ion_current_sl_function(v, *parameters):
    I_0 = parameters[0]
    a = parameters[1]
    return I_0 *(1 + (a*v))


def print_params(values, errors, labels=("I_0", "a", "v_float", "T_e")):
    print("FIT PARAMETERS")
    for i in range(len(values)):
        print("{a} = {b} +/- {c}".format(a=labels[i], b=values[i], c=errors[i]))


def get_runnames(data_dir):
    run_name = data_dir.split(os.sep)[-2]
    print(run_name)
    return 'noga', 'nogaph'


class Flopter(object):

    def __init__(self, data_mount_dir, group_name, folder_name, run_name='prebpro', ext_run_name='prebprobe'):
        ##################################
        #             Extract            #
        ##################################

        # Constants(ish)
        spice_dir = '/home/jleland/Spice/spice2/'
        input_dir = spice_dir + 'bin/inputs/'
        script_dir = spice_dir + 'bin/scripts/'

        # # Run specific data
        # data_mount_dir = 'bin/data/'
        # group_name = 'benchmarking_sam/'
        # folder_name = 'gapless_fullgrid/'
        # data_dir = spice_dir + data_mount_dir + group_name + folder_name

        # ------------------- Run specific data ------------------- #

        # Cumulus
        # data_mount_dir = 'bin/data/'
        # group_name = 'tests/'
        # folder_name = 'fullgridtest/'         # gapped fullgrid
        # folder_name = 'reversed_charge/'
        # folder_name = 'prebiased_probe/'

        # group_name = 'benchmarking_sam/'
        # folder_name = 'gapless_fullgrid/'
        # folder_name = 'gapless_halfgrid1/'
        # folder_name = 'gapless_halfgrid2/'
        # folder_name = 'nogaphalfgrid_tlong1/'
        # folder_name = 'prebprobe_fullgap/'
        # folder_name = 'prebprobe_fullnogap/'

        # Freia
        # data_mount_dir = 'bin/data_f/'
        # group_name = 'rundata'
        # folder/name '6-s-halfgrid'

        data_dir = spice_dir + data_mount_dir + group_name + folder_name

        # input_filename = input_dir + 'jleland.3.inp'
        # input_filename = input_dir + 'jleland.2.inp'
        # input_filename = data_dir + 's_benchmarking_nogap.inp'
        # input_filename = data_dir + 'reversede_ng_hg_sbm.inp'
        # input_filename = data_dir + 'prebiasprobe_ng_hg_sbm.inp'
        input_filename = data_dir + 'input.inp'

        # test = get_runnames(data_dir)

        # run_name = 'prebpro'
        # ext_run_name = 'prebprobe'
        file_suf = '.mat'
        tfile_pre = 't-{a}'.format(a=run_name)
        tfile_path = data_dir + tfile_pre + file_suf
        afile_path = data_dir + ext_run_name + file_suf
        tfile = loadmat(tfile_path)
        self.afile = loadmat(afile_path)

        # create flopter objects
        self.parser = InputParser(input_filename=input_filename)
        self.denormaliser = Denormaliser(input_parser=self.parser)
        self.homogeniser = Spice2Homogeniser(denormaliser=self.denormaliser, data=tfile)

        self.iv_data, self.raw_data = self.homogeniser.homogenise()

##################################
#            Trimming            #
##################################
    def trim(self, trim_beg=0.01, trim_end=0.35):
        full_length = len(self.iv_data['V'])
        # Cut off the noise in the electron saturation region
        t = self.iv_data['t'][int(full_length*trim_beg):int(full_length*trim_end)]
        V = self.iv_data['V'][int(full_length*trim_beg):int(full_length*trim_end)]
        I = self.iv_data['I'][int(full_length*trim_beg):int(full_length*trim_end)]
        I_e = self.iv_data['I_e'][int(full_length*trim_beg):int(full_length*trim_end)]
        I_i = self.iv_data['I_i'][int(full_length*trim_beg):int(full_length*trim_end)]

        # return V, I, I_i, I_e
        return IVData(V, I, t, i_current=I_i, e_current=I_e)

##################################
#             Fitting            #
##################################
    def get_vf(self, iv_data=None):
        if not iv_data:
            iv_data = self.iv_data

        iv_interp = interpolate.interp1d(iv_data['I'], iv_data['V'])
        v_float = iv_interp([0.0])
        return v_float

    def fit(self, iv_data, fit_i=False, fit_simple=False, fit_full=False, print_fl=False):
        # find the potential where the net current is zero using np.interp
        V, I, I_i = iv_data.split()
        v_float = self.get_vf()

        print('v_float = {a}'.format(a=v_float))
        e_temp = 1  # eV
        I_0_sam = 32.774
        a_sam = 0.0204

        if fit_full:
            # Parameters for the full fitting function [I_0, a, v_float, electron_temperature]
            # These are taken from Sam's paper

            params = [I_0_sam, a_sam, v_float, e_temp]
            bounds = ([-np.inf, 0, v_float - 0.01, -np.inf],
                      [np.inf, np.inf, v_float, np.inf])
            fparams, fcov = curve_fit(iv_characteristic_function, V, I, p0=params, bounds=bounds)
            I_fitted = iv_characteristic_function(V, *fparams)
            fstdevs = np.sqrt(np.diag(fcov))
            if print_fl:
                print_params(fparams, fstdevs)
            return V, I_fitted, fparams, fstdevs

        if fit_simple:
            # Parameters for the simple fitting function (no sheath expansion)
            params_simple = [I_0_sam, v_float, e_temp]
            bounds_simple = ([-np.inf, v_float - 0.01, -np.inf],
                            [np.inf, v_float, np.inf])
            I_sam = iv_characteristic_function(V, *params_simple)

            # Run fitting algorithm and create fitted function array

            fparams_simple, fcov_simple = curve_fit(simple_iv_characteristic_function, V, I, p0=params_simple, bounds=bounds_simple)
            I_fitted_simple = simple_iv_characteristic_function(V, *fparams_simple)
            fstdevs_simple = np.sqrt(np.diag(fcov_simple))
            if print_fl:
                print_params(fparams_simple, fstdevs_simple, labels=["I_0", "v_float", "T_e"])
            return V, I_fitted_simple, fparams_simple, fstdevs_simple

        ##################################################
        #         Straight Line Fitting Function         #
        ##################################################

        if fit_i:
            sl_V = np.power(np.abs(V), 0.75)
            sl_params = [I_0_sam, a_sam]
            sl_bounds = ([-np.inf, 0],
                         [np.inf, np.inf])

            sl_fit_params, sl_fit_cov = curve_fit(ion_current_sl_function, sl_V, I_i, p0=sl_params, bounds=sl_bounds)
            sl_I_fitted = ion_current_sl_function(sl_V, *sl_fit_params)
            sl_fstdevs = np.sqrt(np.diag(sl_fit_cov))
            if print_fl:
                print_params(sl_fit_params, sl_fstdevs, labels=["I_0", "a"])
            return sl_V, sl_I_fitted, sl_fit_params, sl_fstdevs

##################################
#             Print              #
##################################

# print fit parameters to console with errors
# print_params(sl_fit_params, sl_fstdevs, labels=["I_0", "a"])

##################################
#              Plot              #
##################################
    def plot_iv(self, fig=None, iv_data=None, plot_vf=False, plot_tot=False, plot_i=False, plot_e=False,
                label='Total', is_final=False):
        if not fig:
            fig = plt.figure()

        if not iv_data:
            iv_data = self.iv_data

        if plot_tot:
            plt.plot(iv_data['V'][:-1], iv_data['I'][:-1], label=label) #, linestyle='dashed')

        if plot_e:
            plt.plot(iv_data['V'][:-1], iv_data['I_e'][:-1], label='Electron')

        if plot_i:
            plt.plot(iv_data['V'][:-1], iv_data['I_i'][:-1], label='Ion')

        # plt.plot(V, I_i, label='Ion')
        # plt.plot(V, I_e, label='Electron')
        # plt.plot(V, I_i, label='Fitted section')

        plt.axhline(y=0, color='gray', linewidth=1, linestyle='dashed')
        if plot_vf:
            v_float = self.get_vf()
            plt.plot([v_float], [0.0], 'x', label=r'V$_{float}$')
            plt.axvline(x=v_float, color='gray', linewidth=1, linestyle='dashed')

        # plt.plot(V_full, I_fitted, label='Fit')
        # plt.plot(V, I_fitted_simple, label='Simple')
        # plt.plot(V, I_sam, label='Sam\'s Params')
        plt.xlabel(r'$\hat{V}$')
        plt.ylabel(r'$\hat{I}$')
        plt.legend()
        if is_final:
            plt.show()

    def plot_f_fit(self, V, I, I_fitted, fig=None, label='', plot_vf=True):
        if not fig:
            fig = plt.figure()

        plt.plot(V, I, 'x')
        plt.plot(V, I_fitted, label='{}Fitted'.format(label))

        plt.axhline(y=0, color='gray', linewidth=1, linestyle='dashed')
        if plot_vf:
            v_float = self.get_vf()
            plt.plot([v_float], [0.0], 'x', label=r'V$_{float}$')
            plt.axvline(x=v_float, color='gray', linewidth=1, linestyle='dashed')

        plt.xlabel(r'$\hat{V}$')
        plt.ylabel(r'$\hat{I}$')
        plt.legend()

    def plot_i_fit(self, sl_V, I_i, I_fitted, fig=None, label=''):
        if not fig:
            fig = plt.figure()

        plt.plot(sl_V, I_i, 'x')
        plt.plot(sl_V, I_fitted, label='{}Fitted'.format(label))
        plt.xlabel(r'$|V|^{3/4}$')
        plt.ylabel(r'$I_i$')
        # plt.legend()

######################
#    Raw Data Plot   #
######################
    def plot_raw(self, fig=None, plot_list=('V', 'I', 'I_e', 'I_i')):
        if not fig:
            fig = plt.figure()

        _PLOTTABLE = {
            'V': 'Raw Voltage',
            'I': 'Raw Current',
            'I_e': 'Raw Electron Current',
            'I_i': 'Raw Ion Current'
        }

        if not plot_list:
            print('Could not plot raw, no variables given in plot_list')
            return

        for var in plot_list:
            if var in _PLOTTABLE.keys():
                plt.plot(self.iv_data['t'][:-1], self.raw_data[var][-1], label=_PLOTTABLE[var])
        plt.legend()
        plt.show()


if __name__ == '__main__':

    flopter_gap = Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullgap/')
    flopter_nogap = Flopter('bin/data/', 'benchmarking_sam/', 'prebprobe_fullnogap/', run_name='prebp', ext_run_name='prebpro')

    ivdata_g = flopter_gap.trim()
    ivdata_ng = flopter_nogap.trim()

    ivdata_g2 = flopter_gap.trim(trim_end=0.5)
    ivdata_ng2 = flopter_nogap.trim(trim_end=0.5)

    ifit_g = flopter_gap.fit(ivdata_g, fit_i=True, print_fl=True)
    ifit_ng = flopter_nogap.fit(ivdata_ng, fit_i=True, print_fl=True)

    ffit_g2 = flopter_gap.fit(ivdata_g2, fit_full=True, print_fl=True)
    ffit_ng2 = flopter_nogap.fit(ivdata_ng2, fit_full=True, print_fl=True)

    fig1 = plt.figure()
    flopter_gap.plot_iv(fig=fig1, plot_tot=True, label='Gap')
    flopter_nogap.plot_iv(fig=fig1, plot_vf=True, plot_tot=True, label='No Gap')

    fig3 = plt.figure()
    flopter_gap.plot_f_fit(ffit_g2[0], ivdata_g2['I'], ffit_g2[1], fig=fig3, label='Gap ', plot_vf=False)
    flopter_nogap.plot_f_fit(ffit_ng2[0], ivdata_ng2['I'], ffit_ng2[1], fig=fig3, label='No Gap ')


    fig2 = plt.figure()
    flopter_gap.plot_i_fit(ifit_g[0], ivdata_g['I_i'], ifit_g[1], fig=fig2, label='Gap ')
    flopter_nogap.plot_i_fit(ifit_ng[0], ivdata_ng['I_i'], ifit_ng[1], fig=fig2, label='No Gap ')
    plt.legend()
    plt.show()




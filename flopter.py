import glob
import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import interpolate
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import argrelmax, savgol_filter

from classes.fitdata import IVFitData
from classes.ivdata import IVData
from classes.spicedata import Spice2TData
from constants import POTENTIAL, CURRENT, ELEC_CURRENT, ION_CURRENT, TIME
from homogeniser import Spice2Homogeniser
from inputparser import InputParser
from normalisation import Denormaliser
from fitters import IVFitter, FullIVFitter


class IVAnalyser(ABC):
    """
    Abstract base class for the analysis of Langmuir Probe data.
    """
    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def trim(self):
        pass

    @abstractmethod
    def denormalise(self):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @staticmethod
    def trim_generic(data, trim_beg=0.0, trim_end=1.0):
        full_length = len(data)
        # Cut off the noise in the electron saturation region
        return data[int(full_length * trim_beg):int(full_length * trim_end)]

    # @abstractmethod
    # def plot(self):
    #     pass
    #
    # @abstractmethod
    # def save(self):
    #     pass
    

class Flopter(IVAnalyser):

    def __init__(self, data_mount_dir, group_name, folder_name, run_name=None, prepare=True):
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
        self.input_filename = data_dir + 'input.inp'

        if not run_name:
            file_suf = '.mat'
            tfile_pre = 't-'
            self.tfile_path, self.afile_path = self.get_runnames(data_dir, file_suf, tfile_pre)
        else:
            # run_name = 'prebpro'
            # ext_run_name = 'prebprobe'
            run_name_short = run_name[:-2]

            file_suf = '.mat'
            tfile_pre = 't-{a}'.format(a=run_name_short)
            self.tfile_path = data_dir + tfile_pre + file_suf
            self.afile_path = data_dir + run_name + file_suf

        if self.tfile_path:
            self.tdata = Spice2TData(self.tfile_path)
        else:
            raise ValueError('No t-file given')

        if self.afile_path:
            self.afile = loadmat(self.afile_path)
        else:
            print('No a-file given, continuing without')
            self.afile = None

        self.parser = None
        self.denormaliser = None
        self.homogeniser = None
        self.iv_data = None
        self.raw_data = None

        if prepare:
            self.prepare()

    def prepare(self, homogenise=True, denormalise=True):
        # create flopter objects
        self.parser = InputParser(input_filename=self.input_filename)
        if denormalise:
            self.denormaliser = Denormaliser(dt=self.tdata.dt, input_parser=self.parser)
            self.tdata.converter = self.denormaliser
        if homogenise:
            self.homogeniser = Spice2Homogeniser(data=self.tdata, input_parser=self.parser)
            self.iv_data, self.raw_data = self.homogeniser.homogenise()

    def get_runnames(self, directory, file_suffix, tfile_prefix):
        cwd = os.getcwd()
        os.chdir(directory)
        tfile_glob_str = '{}{}*[!0-9]{}'.format(directory, tfile_prefix, file_suffix)
        tfile_name = glob.glob(tfile_glob_str)
        afile_name = glob.glob('{}[!{}]*[!0-9]{}'.format(directory, tfile_prefix, file_suffix))
        os.chdir(cwd)
        return next(iter(tfile_name), None), next(iter(afile_name), None)

##################################
#            Trimming            #
##################################
    def trim(self, trim_beg=0.01, trim_end=0.35):
        # Cut off the noise in the electron saturation region
        t = Flopter.trim_generic(self.iv_data[TIME], trim_beg=trim_beg, trim_end=trim_end)
        V = Flopter.trim_generic(self.iv_data[POTENTIAL], trim_beg=trim_beg, trim_end=trim_end)
        I = Flopter.trim_generic(self.iv_data[CURRENT], trim_beg=trim_beg, trim_end=trim_end)
        I_e = Flopter.trim_generic(self.iv_data[ELEC_CURRENT], trim_beg=trim_beg, trim_end=trim_end)
        I_i = Flopter.trim_generic(self.iv_data[ION_CURRENT], trim_beg=trim_beg, trim_end=trim_end)

        return IVData(V, I, t, i_current=I_i, e_current=I_e)

    def denormalise(self):
        self.iv_data = self.denormaliser(self.iv_data)
        self.raw_data = self.denormaliser(self.raw_data)
        self.tdata.denormalise()

##################################
#             Fitting            #
##################################
    def get_vf(self, iv_data=None):
        if not iv_data:
            iv_data = self.iv_data

        iv_interp = interpolate.interp1d(iv_data[CURRENT], iv_data[POTENTIAL])
        v_float = iv_interp([0.0])
        return v_float

    def get_plasma_potential(self, iv_data=None):
        if not iv_data:
            iv_data = self.iv_data

        vf = self.get_vf(iv_data)
        print(vf)

        gradient = np.gradient(-iv_data[CURRENT])
        gradient2 = np.gradient(gradient)
        peaks = argrelmax(gradient)

        smoothed_gradient = savgol_filter(gradient, 21, 2)
        smoothed_peaks = argrelmax(smoothed_gradient)

        # print(gradient)
        # print(peaks)

        plt.figure()
        # plt.plot(iv_data[POTENTIAL], iv_data[CURRENT])
        plt.plot(iv_data[POTENTIAL], gradient)
        plt.plot(iv_data[POTENTIAL], smoothed_gradient)
        # plt.plot(iv_data[POTENTIAL], gradient2)
        phi = iv_data[POTENTIAL][[p for p in smoothed_peaks[0] if iv_data[POTENTIAL][p] > vf][0]]
        # for p in [p for p in smoothed_peaks[0] if iv_data[POTENTIAL][p] > vf]:
        #     plt.axvline(iv_data[POTENTIAL][p], linewidth=1, color='r')
        plt.show()
        return phi

    def fit(self, iv_data, fitter=FullIVFitter(), print_fl=False, bounds=None, initial_vals=None):
        # TODO: Reimplement the finding of floating potential
        assert isinstance(fitter, IVFitter)

        fit_data = fitter.fit_iv_data(iv_data, initial_vals=initial_vals, bounds=bounds)
        if print_fl:
            fit_data.print_fit_params()

        return fit_data

    def temperature_fit(self, iv_data=None):
        vf = self.get_vf(iv_data)
        phi = self.get_plasma_potential(iv_data)
        const = np.log(0.6 * np.sqrt((2 * np.pi) / self.denormaliser.get_mu()))
        temperature = (phi - vf) / const
        return temperature

    # Guess parameters from previous fitting routines
        # These are taken from Sam's paper
        # v_float = self.get_vf()
        # e_temp = 1  # eV
        # I_0_sam = 32.774
        # a_sam = 0.0204

    # Parameters for the full fitting function [I_0, a, v_float, electron_temperature]
        # params = [I_0_sam, a_sam, v_float, e_temp]
        # bounds = ([-np.inf, 0, v_float - 0.01, -np.inf],
        #           [np.inf, np.inf, v_float, np.inf])

    # Parameters for the simple fitting function (no sheath expansion)
        # params_simple = [I_0_sam, v_float, e_temp]
        # bounds_simple = ([-np.inf, v_float - 0.01, -np.inf],
        #                 [np.inf, v_float, np.inf])

    # Parameters for the straight line fitting function
        # sl_V = np.power(np.abs(V), 0.75)
        # sl_params = [I_0_sam, a_sam]
        # sl_bounds = ([-np.inf, 0],
        #              [np.inf, np.inf])

##################################
#              Plot              #
##################################
    AX_LABEL_IV = 'IV'
    AX_LABEL_ION_FIT = 'Ion Current'
    _AXES_LABELS = {
        AX_LABEL_IV: [r'$\hat{V}$', r'$\hat{I}$'],
        AX_LABEL_ION_FIT: [r'$|\hat{V}|^{3/4}$', r'$\hat{I}_i$'],
    }

    def plot_iv(self, fig=None, iv_data=None, plot_vf=False, plot_tot=False, plot_i=False, plot_e=False,
                label='Total', is_final=False):
        if not fig:
            fig = plt.figure()

        if not iv_data:
            iv_data = self.iv_data

        if plot_tot:
            plt.plot(iv_data[POTENTIAL][:-1], iv_data[CURRENT][:-1], label=label) #, linestyle='dashed')

        if plot_e:
            plt.plot(iv_data[POTENTIAL][:-1], iv_data[ELEC_CURRENT][:-1], label='Electron')

        if plot_i:
            plt.plot(iv_data[POTENTIAL][:-1], iv_data[ION_CURRENT][:-1], label='Ion')

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

    def plot_f_fit(self, iv_fit_data, fig=None, label=None, plot_raw=True, plot_vf=True):
        if not fig:
            plt.figure()

        if plot_raw:
            plt.plot(*iv_fit_data.get_raw_plottables(), 'x')
        plt.plot(*iv_fit_data.get_fit_plottables(), label=label)

        plt.axhline(y=0, color='gray', linewidth=1, linestyle='dashed')
        if plot_vf:
            v_float = self.get_vf()
            plt.plot([v_float], [0.0], 'x', label=r'V$_{float}$')
            plt.axvline(x=v_float, color='gray', linewidth=1, linestyle='dashed')

        plt.xlabel(self._AXES_LABELS[self.AX_LABEL_IV][0])
        plt.ylabel(self._AXES_LABELS[self.AX_LABEL_IV][1])
        if label:
            plt.legend()

    def plot_i_fit(self, iv_fit_data, fig=None, label='', axes=None):
        if not fig:
            fig = plt.figure()

        plt.plot(*iv_fit_data.get_raw_plottables(), 'x')
        plt.plot(*iv_fit_data.get_fit_plottables(), label=label)
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
                plt.plot(self.iv_data[TIME][:-1], self.raw_data[var][-1], label=_PLOTTABLE[var])
        plt.legend()
        plt.show()

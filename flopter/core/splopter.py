import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.io import loadmat
from scipy.signal import argrelmax, savgol_filter

from flopter.classes.base import IVAnalyser
from flopter.classes.ivdata import IVData
import flopter.classes.spicedata as sd
from flopter.core import constants as c
from flopter.core.homogenisation import Spice2Homogeniser
from flopter.core.inputparser import InputParser
from flopter.core.normalisation import Denormaliser
from flopter.core.fitters import IVFitter, FullIVFitter, GaussianFitter
    

class Splopter(IVAnalyser):
    """
        Implementation of IVAnalyser for the analysis of probe data from SPICE
    """
    _tfile_prefix = 't-'
    _file_suffix = '.mat'
    _dump_suffix = '.2d.'

    def __init__(self, data_mount_dir, group_name, folder_name, run_name=None, prepare=False, reduce_fl=False):
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

        self.data_dir = spice_dir + data_mount_dir + group_name + folder_name

        # input_filename = input_dir + 'jleland.3.inp'
        # input_filename = input_dir + 'jleland.2.inp'
        # input_filename = data_dir + 's_benchmarking_nogap.inp'
        # input_filename = data_dir + 'reversede_ng_hg_sbm.inp'
        # input_filename = data_dir + 'prebiasprobe_ng_hg_sbm.inp'
        self.input_filename = self.data_dir + 'input.inp'

        # This appears to be in mostly for backwards compatibility with older SPICE runs
        if not run_name:
            self.tfile_path, self.afile_path = self.get_ta_filenames(self.data_dir)
        else:
            run_name_short = run_name[:-2]
            trun_name = '{a}{b}'.format(a=self._tfile_prefix, b=run_name_short)
            self.tfile_path = self.data_dir + trun_name + self._file_suffix
            self.afile_path = self.data_dir + run_name + self._file_suffix

        if self.tfile_path:
            if not run_name:
                self.tfile_path = self.data_dir + self.tfile_path
            self.tdata = sd.Spice2TData(self.tfile_path, deallocate=reduce_fl)
            if reduce_fl:
                self.tdata.reduce(sd.DEFAULT_REDUCED_DATASET)
        else:
            raise ValueError('No t-file given')

        if self.afile_path:
            if not run_name:
                self.afile_path = self.data_dir + self.afile_path
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

    def prepare(self, homogenise=True, make_denormaliser=True):
        """
            Check existence of, and then populate, the main flopter objects: inputparser, denormaliser and homogeniser.

            Default behaviour is to populate all, mandatory behaviour is only to create input parser. Homogeniser and
            Denormaliser creation is controlled through the boolean input flags. Note that specifying the
            make_denormaliser flag does not automatically denormalise all data, it merely populates the denormaliser
            object.
        """
        if not self.parser:
            self.parser = InputParser(input_filename=self.input_filename)

        if make_denormaliser and not self.denormaliser:
            self.denormaliser = Denormaliser(dt=self.tdata.dt, input_parser=self.parser)
            ratio = self.find_se_temp_ratio()
            self.denormaliser.set_se_temperature(ratio)
            self.tdata.converter = self.denormaliser
        elif make_denormaliser and self.denormaliser is not None:
            print('Cannot make_denormaliser, data has already been denormalised!')

        if homogenise and not self.homogeniser:
            self.homogeniser = Spice2Homogeniser(data=self.tdata, input_parser=self.parser)
            self.iv_data, self.raw_data = self.homogeniser.homogenise()
        elif homogenise and self.homogeniser is not None:
            print('Cannot homogenise, data has already been homogenised!')

    @classmethod
    def get_ta_filenames(cls, directory):
        # TODO: Update this to work with Paths
        cwd = os.getcwd()
        if os.path.exists(directory):
            os.chdir(directory)
        else:
            raise OSError('Directory \'{}\' invlaid.'.format(directory))
        tfile_all_glob_str = '{}*{}'.format(cls._tfile_prefix, cls._file_suffix)
        tfile_num_glob_str = '{}*[0-9][0-9]{}'.format(cls._tfile_prefix, cls._file_suffix)
        all_tfiles = glob.glob(tfile_all_glob_str)
        numbered_tfiles = glob.glob(tfile_num_glob_str)
        tfile_name = [tfile for tfile in all_tfiles if tfile not in numbered_tfiles]
        afile_name = glob.glob('[!{}]*[!0-9][!{}]{}'.format(cls._tfile_prefix, cls._dump_suffix, cls._file_suffix))
        os.chdir(cwd)
        return next(iter(tfile_name), None), next(iter(afile_name), None)

##################################
#            Trimming            #
##################################
    def trim(self, iv_data=None, trim_beg=0.01, trim_end=0.35):
        if not iv_data and self.iv_data:
            iv_data = self.iv_data
        elif not self.iv_data:
            print('self.iv_data not set, running homogenise now...')
            self.prepare(make_denormaliser=False)
            iv_data = self.iv_data

        # Cut off the noise in the electron saturation region
        t = Splopter.trim_generic(iv_data[c.TIME], trim_beg=trim_beg, trim_end=trim_end)
        V = Splopter.trim_generic(iv_data[c.POTENTIAL], trim_beg=trim_beg, trim_end=trim_end)
        I = Splopter.trim_generic(iv_data[c.CURRENT], trim_beg=trim_beg, trim_end=trim_end)
        I_e = Splopter.trim_generic(iv_data[c.ELEC_CURRENT], trim_beg=trim_beg, trim_end=trim_end)
        I_i = Splopter.trim_generic(iv_data[c.ION_CURRENT], trim_beg=trim_beg, trim_end=trim_end)

        return IVData(V, I, t, i_current=I_i, e_current=I_e)

    def denormalise(self, denormalise_t_data=False):
        self.iv_data = self.denormaliser(self.iv_data)
        self.raw_data = self.denormaliser(self.raw_data)
        if denormalise_t_data:
            self.denormalise_t_data()

    def denormalise_t_data(self):
        self.tdata.denormalise()

##################################
#             Fitting            #
##################################
    def get_vf(self, iv_data=None):
        if not iv_data:
            iv_data = self.iv_data

        iv_interp = interpolate.interp1d(iv_data[c.CURRENT], iv_data[c.POTENTIAL])
        v_float = iv_interp([0.0])
        return v_float

    def get_plasma_potential(self, iv_data=None):
        if not iv_data:
            iv_data = self.iv_data

        vf = self.get_vf(iv_data)
        print(vf)

        gradient = np.gradient(-iv_data[c.CURRENT])
        gradient2 = np.gradient(gradient)
        peaks = argrelmax(gradient)

        smoothed_gradient = savgol_filter(gradient, 21, 2)
        smoothed_peaks = argrelmax(smoothed_gradient)

        # print(gradient)
        # print(peaks)
        print(len(gradient))

        plt.figure()
        # plt.plot(iv_data[POTENTIAL], iv_data[CURRENT])
        plt.plot(iv_data[c.POTENTIAL], gradient)
        plt.plot(iv_data[c.POTENTIAL], smoothed_gradient)
        # plt.plot(iv_data[POTENTIAL], gradient2)
        phi = iv_data[c.POTENTIAL][[p for p in smoothed_peaks[0] if iv_data[c.POTENTIAL][p] > vf][0]]
        # for p in [p for p in smoothed_peaks[0] if iv_data[POTENTIAL][p] > vf]:
        #     plt.axvline(iv_data[POTENTIAL][p], linewidth=1, color='r')
        plt.show()
        return phi

    def fit(self, iv_data=None, fitter=None, print_fl=False, initial_vals=None, bounds=None):
        # TODO: Reimplement the finding of floating potential
        if not iv_data:
            iv_data = self.iv_data

        v_f = self.get_vf(iv_data)
        print(v_f)

        if fitter:
            assert isinstance(fitter, IVFitter)
        else:
            fitter = FullIVFitter()

        fit_data = fitter.fit_iv_data(iv_data, initial_vals=initial_vals, bounds=bounds)
        if print_fl:
            fit_data.print_fit_params()

        return fit_data

    def temperature_estimation(self, iv_data=None):
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

###########################################
#              DF Extraction              #
###########################################

    def find_se_temp_ratio(self, v_scale=1000, plot_fl=False, species=2, regions=None):
        # TODO: (14/01/19) Make function to find regions from simulation object geometry first, then tries to interpret
        # TODO: from the input file.
        # If no regions specified, find them from diagnostic regions in input file
        if not regions:
            regions = self.parser.get_hist_diag_regions(species)

        # If still not regions
        if not isinstance(regions, dict) or len(regions) == 0:
            print('WARNING: No viable sheath edge comparison regions provided or found, using defaults.')
            regions = {
                'Sheath Edge': [74, 90, 0, 1000],       # Sheath edge
                'Injection': [354, 370, 0, 1000]        # Injection area
            }

        hists = self.extract_histograms(list(regions.values()), denormalise=True, v_scale=v_scale, species=species)
        temperature = self.parser.get_commented_params()[c.ELEC_TEMP]

        fitdatas = []
        fitter = GaussianFitter(si_units=False, mu=1, v_scale=v_scale)
        for hist_data in hists:
            fitdata = fitter.fit(hist_data[0], hist_data[1], temp=temperature)
            fitdatas.append(fitdata)

        if plot_fl:
            plt.figure()
            for i in range(len(hists)):
                fitdatas[i].print_fit_params()
                plt.plot(*hists[i], label='Hist - {}'.format(list(regions.keys())[i]))
                plt.plot(*fitdatas[i].get_fit_plottables(), label='Fit')
                plt.legend()
            plt.show()

        ratio = fitdatas[0].fit_params[0].value / fitdatas[1].fit_params[0].value
        print('Sheath edge temperature ratio is: {}'.format(ratio))
        return ratio

    def extract_histograms(self, regions, denormalise=False, v_scale=1, species=2, deallocate_fl=True):
        if denormalise and not self.denormaliser:
            self.prepare(homogenise=False)
        elif not self.parser:
            self.prepare(homogenise=False, make_denormaliser=False)

        nproc = int(np.squeeze(self.tdata.nproc))
        region_vels = [np.array([])] * len(regions)
        ralpha = (-self.tdata.alphayz / 180.0) * 3.141591
        rbeta = ((90.0 - self.tdata.alphaxz) / 180) * 3.141591

        for i in range(nproc):
            num = str(i).zfill(2)
            filename = self.tfile_path.replace('.mat', '{}.mat'.format(num))
            p_file = loadmat(filename)

            for j, region in enumerate(regions):
                z_low = region[0]
                z_high = region[1]
                y_low = region[2]
                y_high = region[3]
                indices = np.where((p_file['z'] > z_low) & (p_file['z'] <= z_high)
                                   & (p_file['y'] > y_low) & (p_file['y'] <= y_high)
                                   & (p_file['stype'] == species))
                region_vels[j] = np.append(region_vels[j], (p_file['uy'][indices] * np.cos(ralpha) * np.cos(rbeta))
                                           - (p_file['uz'][indices] * np.sin(ralpha)))

        hists = []
        for i in range(len(region_vels)):
            if denormalise:
                region_vels[i] = self.denormaliser(region_vels[i], c.CONV_VELOCITY)
            if v_scale:
                region_vels[i] = region_vels[i] / v_scale

            hist, gaps = np.histogram(-region_vels[i], bins='auto', density=True)
            bins = (gaps[:-1] + gaps[1:]) / 2
            hists.append([bins, hist])

        if deallocate_fl:
            import gc
            del region_vels
            gc.collect()

        return hists

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
                label='Total', show_fl=False):
        if not fig:
            fig = plt.figure()

        if not iv_data:
            iv_data = self.iv_data

        if plot_tot:
            plt.plot(iv_data[c.POTENTIAL][:-1], iv_data[c.CURRENT][:-1], label=label)

        if plot_e:
            plt.plot(iv_data[c.POTENTIAL][:-1], iv_data[c.ELEC_CURRENT][:-1], label='Electron')

        if plot_i:
            plt.plot(iv_data[c.POTENTIAL][:-1], iv_data[c.ION_CURRENT][:-1], label='Ion')

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
        if show_fl:
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
    def plot_raw(self, fig=None, plot_list=('V', 'I', 'I_e', 'I_i'), show_fl=False):
        if not fig:
            fig = plt.figure()

        _PLOTTABLE = {
            c.POTENTIAL: 'Raw Voltage',
            c.CURRENT: 'Raw Current',
            c.ELEC_CURRENT: 'Raw Electron Current',
            c.ION_CURRENT: 'Raw Ion Current'
        }

        if not plot_list:
            print('Could not plot raw, no variables given in plot_list')
            return

        for var in plot_list:
            if var in _PLOTTABLE.keys():
                plt.plot(self.raw_data[c.TIME][:-1], self.raw_data[var][:-1], label=_PLOTTABLE[var])
        plt.legend()
        if show_fl:
            plt.show()

    def plot_1d_variable(self, variable_label=sd.OBJECTSCURRENTE, time_dep_fl=False, diagnostic_fl=False, fig=None,
                         show_fl=True):
        if not diagnostic_fl:
            y_var = self.tdata.t_dict[variable_label]
        else:
            y_var = self.tdata.diagnostics[variable_label]

        if time_dep_fl:
            x_var = self.tdata.t
        else:
            x_var = None

        if not fig:
            plt.figure()
        if x_var is not None and len(x_var) == len(y_var):
            plt.plot(x_var, y_var)
        else:
            plt.plot(y_var)

        if show_fl:
            plt.show()

    def plot_2d_variable(self, t_dict_label=sd.POT, plot_obj_fl=False, show_fl=True):
        plasma_parameter = np.flip(self.tdata.t_dict[t_dict_label], 0)
        objects_raw = np.flip(self.tdata.objectsenum, 0)
        probe_obj_indices = self.parser.get_probe_obj_indices()
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
        im = ax.imshow(plasma_parameter, extent=[0, len(plasma_parameter[0]) / 2, 0, len(plasma_parameter) / 2],
                       interpolation=None)
        plt.xlabel(r'y / $\lambda_D$', fontsize=15)
        plt.ylabel(r'z / $\lambda_D$', fontsize=15)
        # plt.title('Electrostatic potential for a flush mounted probe', fontsize=20)
        plt.quiver([200], [200], self.tdata.by, self.tdata.bz, scale=5)
        plt.colorbar(im, fraction=0.035, pad=0.04)
        if show_fl:
            plt.show()

    def analyse_iv(self, show_fl=True):
        iv_data = self.trim(trim_end=0.5)
        fit_data = self.fit(iv_data)
        self.plot_raw(plot_list=[c.CURRENT, c.ION_CURRENT, c.ELEC_CURRENT])
        self.plot_iv(plot_vf=True, plot_tot=True, show_fl=True)
        if show_fl:
            plt.show()

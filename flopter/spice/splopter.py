import glob
import os
import pathlib as pth
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.signal import argrelmax, savgol_filter

from flopter.core.ivanalyser import IVAnalyser
from flopter.core.ivdata import IVData
from flopter.core import constants as c
from flopter.core.fitters import IVFitter, FullIVFitter, GaussianFitter
from flopter.spice.inputparser import InputParser
from flopter.spice.normalise import Denormaliser
import flopter.spice.utils as ut
import flopter.spice.tdata as sd


class Splopter(IVAnalyser):
    """
    Implementation of IVAnalyser for the analysis of probe data from SPICE.
    Works on the same workflow of create, prepare, fit, plot, save.
    """
    TFILE_PREFIX = 't-'
    SPICE_EXTENSION = '.mat'
    DUMP_SUFFIX = '.2d.'

    DEFAULT_SPICE_REPO = pth.Path.home() / 'Spice' / 'spice2'
    COMPLETION_VOLTAGE = 9.95

    def __init__(self, spice_data_dir, run_name=None, reduce=None, check_completion_fl=False, ignore_tzero_fl=False,
                 check_voltage_error_fl=True, version=2.13, store_dataframe_fl=False, ignore_a_file=False):
        self.data_dir = pth.Path(spice_data_dir)

        if not self.is_code_output_dir(self.data_dir):
            print('Spice data directory is not valid, attempting to auto-fix.')
            test_data_dir = self.DEFAULT_SPICE_REPO / self.data_dir
            if not self.is_code_output_dir(test_data_dir):
                print(f'Passed Spice directory ({spice_data_dir}) doesn\'t seem to be valid.\n'
                      f'Continuing anyway.')

        # Make list of backup folders made for this simulation, if any
        self.backup_folders = [] + list(self.data_dir.glob('backup_*'))
        self.backup_folders.sort()

        if version not in self.VERSION_HOMOGENISERS:
            print(f"Version provided ({version}) not supported, finding nearest available supported version to use \n"
                  f"instead.")
            version = min(*self.VERSION_HOMOGENISERS, key=lambda x: abs(x - version))
            print(f"Continuing with version {version}")

        self.version = version
        self._homogenise_impl = self.VERSION_HOMOGENISERS[version]

        self.input_filename = self.data_dir / 'input.inp'
        if not self.input_filename.exists():
            self.input_filename = list(self.data_dir.glob('*.inp'))[0]

        # This is still in mostly for backwards compatibility with older SPICE runs
        if not run_name:
            self.tfile_path, self.afile_path = self.get_ta_filenames(self.data_dir)
        else:
            run_name_short = run_name[:-2]
            trun_name = '{a}{b}'.format(a=self.TFILE_PREFIX, b=run_name_short)
            self.tfile_path = trun_name + self.SPICE_EXTENSION
            self.afile_path = run_name + self.SPICE_EXTENSION

        if self.tfile_path:
            tfile_path = self.data_dir / self.tfile_path

            # TODO: (2019-12-12) This should probably be replaced with a concatenation of backup folders seeing as the
            # TODO: restarts definitely won't be working any time soon
            t = loadmat(tfile_path, variable_names=[sd.T])[sd.T]
            if not ignore_tzero_fl and np.mean(t) == 0.0:
                print(f'WARNING: Encountered t-zeroing in {tfile_path}')
                if len(self.backup_folders) > 0:
                    print('Looking for a suitable backup to use instead.')
                    found_backup_fl = False
                    for backup_data_dir in self.backup_folders:
                        assert self.is_code_output_dir(backup_data_dir)
                        backup_tfile_path, backup_afile_path = self.get_ta_filenames(backup_data_dir)
                        tfile_path = backup_data_dir / backup_tfile_path
                        t = loadmat(tfile_path, variable_names=[sd.T])[sd.T]
                        if np.mean(t) != 0.0:
                            found_backup_fl = True
                            break
                    if found_backup_fl:
                        print(f'Useable backup found at {tfile_path}')
                    else:
                        print(f'No usable backups found, reverting to t-zeroing parent folder... \n'
                              f'Carrying on but expect erroneous results.')
                        tfile_path = self.data_dir / self.tfile_path
                else:
                    print('There are no backups to utilise. Carrying on, but expect erroneous results.')

            self.tfile_path = tfile_path
            if reduce is True:
                self.tdata = sd.Spice2TData(self.tfile_path, deallocate=reduce)
                self.tdata.reduce(sd.DEFAULT_REDUCED_DATASET)
            else:
                self.tdata = sd.Spice2TData(self.tfile_path, variable_names=reduce)

            if check_voltage_error_fl \
                    and np.min(np.squeeze(self.tdata.diagnostics[c.DIAG_PROBE_POT])) > c.SWEEP_LOWER * 0.99:
                raise FailedSimulationException('Voltage data is malformed.')

        else:
            raise ValueError('No t-file found in directory')

        if self.afile_path and not ignore_a_file:
            self.afile_path = self.data_dir / self.afile_path
            self.afile = loadmat(self.afile_path)
        else:
            print('No a-file given, continuing without')
            self.afile = None

        if check_completion_fl:
            self.parser = InputParser(input_filename=self.input_filename)
            self.COMPLETION_VOLTAGE = 5
            log_files = list(self.data_dir.glob('log.ongoing.out'))
            if len(log_files) > 0:
                self.log_file = log_files[0]
                final_percentage = self.get_percentage_completed(self.log_file)
                if not self.parser.has_sufficient_sweep(final_percentage, voltage_threshold=self.COMPLETION_VOLTAGE):
                    raise FailedSimulationException('Simulation has not reached the required percentage through the '
                                                    'sweep (60%) to be analysed')
            else:
                print('No log files found, cannot check if simulation is sufficiently completed.')
        else:
            self.log_file = None

        # Flag to control whether the iv_data is also stored as a dataframe at homogenisation
        self.store_dataframe_fl = store_dataframe_fl
        self.iv_df = None

        self.parser = None
        self.denormaliser = None
        self.homogeniser = None
        self.iv_data = None
        self.raw_data = None

    def prepare(self, homogenise_fl=True, denormaliser_fl=True, find_se_temp_fl=True, backup_concat_fl=True):
        """
            Check existence of, and then populate, the main flopter objects:
            inputparser, denormaliser. Homogeniser has now been rolled into the
            single function 'homogenise()'

            Default behaviour is to populate all, mandatory behaviour is only to
            create input parser. Denormaliser creation is
            controlled through the appropriate boolean kwargs. Specifying
            denormaliser_fl does not automatically denormalise all data,
            it merely initialises the denormaliser object.

        """
        self.parse_input()

        if denormaliser_fl and not self.denormaliser:
            self.denormaliser = Denormaliser(dt=self.tdata.dt, input_parser=self.parser)
            if find_se_temp_fl:
                ratio = self.find_se_temp_ratio()
            else:
                ratio = 1.0
            self.denormaliser.set_se_temperature(ratio)
            self.tdata.converter = self.denormaliser
        elif denormaliser_fl and self.denormaliser is not None:
            print('Cannot make_denormaliser, data has already been denormalised!')

        if homogenise_fl and self.iv_data is None:
            if backup_concat_fl:
                self.iv_data, self.raw_data = self.homogenise(backups=self.backup_folders)
            else:
                self.iv_data, self.raw_data = self.homogenise()
        elif homogenise_fl and self.iv_data is not None:
            print('Cannot homogenise, data has already been homogenised!')

    def homogenise(self, **kwargs):
        """
        Homogenise function for creation of IVData objects for use in SPICE simulation analysis. Utilises a version
        dependent method defined elsewhere.

        """
        return self._homogenise_impl(self, **kwargs)

    def parse_input(self):
        if not self.parser:
            self.parser = InputParser(input_filename=self.input_filename)

    @staticmethod
    def get_h_remaining_lines(log_file):
        time_left_lines = []
        with open(str(log_file), 'r') as f:
            for line in f.readlines():
                if re.search('% done', line):
                    time_left_lines.append(line.strip())
        return time_left_lines

    @staticmethod
    def get_percentage_completed(log_file):
        return int(Splopter.get_h_remaining_lines(log_file)[-1].split()[0]) / 100

    @staticmethod
    def strip_head(current, prev_current, print_fl=False):
        # Primary method - find first point roughly equal to previous end value
        start_points = np.where(np.logical_and(current >= prev_current * 0.8, current <= prev_current * 1.2))
        if len(start_points[0]) > 0:
            for start_point in start_points[0]:
                if not any(current[start_point:] == 0.0):
                    if print_fl:
                        print('Successfully used primary method')
                    return current[start_point:], start_point

        # Secondary method - find last zero in current array and take the succeeding value
        zero_indices = np.where(current == 0.0)[0]
        if current[-1] != 0.0 and len(zero_indices) > 0:
            # Take start point to be the value one place along from the final 0
            start_point = np.max(zero_indices + 1)
            if print_fl:
                print('Successfully used secondary method')
            return current[start_point:], start_point

        return None, None

    @staticmethod
    def strip_tail(current, threshold=10000, print_fl=False):
        # Strip tail section if present
        big_spikes = np.where(np.abs(current) > threshold)
        if len(big_spikes[0]) == 0:
            if print_fl:
                print('No spikes found on tail.')
            return current, len(current)
        elif len(big_spikes[0]) >= 1:
            end_spike = big_spikes[0][np.argmax(np.abs(current[big_spikes]))]
            current = current[:end_spike]
            return current, end_spike
        return None, None

    @staticmethod
    def group_spikes(voltage, tolerance=3):
        # Determine location of start and end points of the spikes by locating values >3 sigma from mean difference
        bias_diff = np.diff(voltage)
        diff_outliers = np.where(np.abs(bias_diff) >= 0.06)[0]

        if len(diff_outliers) % 2 != 0:
            print('Found an odd number of spike transitions')

            start_spike = 0
            end_spike = 1
            pairs = []
            # Group by diffs where there is a change of sign
            while end_spike < len(diff_outliers):
                if np.sign(bias_diff[diff_outliers][start_spike]) != np.sign(bias_diff[diff_outliers][end_spike]):
                    pairs.append([diff_outliers[start_spike], diff_outliers[end_spike]])
                    start_spike = end_spike + 1
                    end_spike = start_spike + 1
                else:
                    end_spike += 1
            return pairs
        else:
            return list(zip(diff_outliers[::2], diff_outliers[1::2]))

    @staticmethod
    def prune_voltage(voltage):
        # Find consecutive problem values and group them
        edges = Splopter.group_spikes(voltage)

        # Construct a replacement array to replace each spike
        # TODO: (2020-01-05) This doesn't look ideal when looking closely at a high resolution voltage trace, but
        #  should be good enough for the time being.
        if len(edges) > 0:
            for start_spike, end_spike in edges:
                if start_spike == 0 or end_spike == len(voltage) - 1:
                    raise NotImplementedError('Edge cases not accounted for yet')

                start_v = voltage[start_spike]
                end_v = voltage[end_spike + 1]
                length = end_spike - start_spike
                replacement = np.linspace(start_v, end_v, length + 2)

                voltage[start_spike:end_spike + 2] = replacement
        return voltage

    @classmethod
    def get_ta_filenames(cls, directory):
        if not isinstance(directory, pth.Path) and isinstance(directory, str):
            directory = pth.Path(directory)
        elif not isinstance(directory, pth.Path):
            raise ValueError(f'Argument directory ({directory}) is not valid for this operation.')

        cwd = os.getcwd()
        if directory.exists():
            os.chdir(directory)
        else:
            raise OSError('Directory \'{}\' invlaid.'.format(directory))
        tfile_all_glob_str = '{}*{}'.format(cls.TFILE_PREFIX, cls.SPICE_EXTENSION)
        tfile_num_glob_str = '{}*[0-9][0-9]{}'.format(cls.TFILE_PREFIX, cls.SPICE_EXTENSION)
        all_tfiles = glob.glob(tfile_all_glob_str)
        numbered_tfiles = glob.glob(tfile_num_glob_str)
        tfile_name = [tfile for tfile in all_tfiles if tfile not in numbered_tfiles]
        afile_name = glob.glob('[!{}]*[!0-9][!{}]{}'.format(cls.TFILE_PREFIX, cls.DUMP_SUFFIX, cls.SPICE_EXTENSION))
        os.chdir(cwd)
        return next(iter(tfile_name), None), next(iter(afile_name), None)

    @staticmethod
    def is_code_output_dir(directory):
        return ut.is_code_output_dir(directory)

##################################
#            Trimming            #
##################################
    def trim(self, iv_data=None, trim_beg=0.01, trim_end=0.35):
        if not iv_data and self.iv_data:
            iv_data = self.iv_data
        elif not self.iv_data:
            print('self.iv_data not set, running homogenise now...')
            self.prepare(denormaliser_fl=False)
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

        return iv_data.get_vf()

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
            self.prepare(homogenise_fl=False)
        elif not self.parser:
            self.prepare(homogenise_fl=False, denormaliser_fl=False)

        nproc = int(np.squeeze(self.tdata.nproc))
        region_vels = [np.array([])] * len(regions)
        ralpha = (-self.tdata.alphayz / 180.0) * 3.141591
        rbeta = ((90.0 - self.tdata.alphaxz) / 180) * 3.141591

        for i in range(nproc):
            num = str(i).zfill(2)
            filename = str(self.tfile_path).replace('.mat', '{}.mat'.format(num))
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

    def plot_2d_variable(self, t_dict_label=sd.POT, plot_obj_fl=False, show_fl=True, b_arrow_loc=(200, 200)):
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
        plt.quiver(*b_arrow_loc, self.tdata.by, self.tdata.bz, scale=5)
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

    ############################################################################
    #                        Homogeniser Implementations                       #
    ############################################################################

    def get_tdata_raw_iv(self, tdata=None):
        """
        Collates and returns all raw data from a TData object to construct an IV
        characteristic.

        :param tdata:   A properly populated TData object
        :return:        A tuple of (time, probe_bias, probe_electron_current,
                        probe_ion_current). Efforts are made to ensure the
                        current arrays are the same length as time, but this is
                        not guaranteed. probe_bias is unlikely to be the same
                        length as the others.

        """
        if tdata is None:
            tdata = self.tdata

        # Extract relevant arrays from the matlab file
        probe_indices = self.parser.get_probe_obj_indices()
        probe_current_e = 0.0
        probe_current_i = 0.0

        time = np.squeeze(tdata.t)[:-1]
        for index in probe_indices:
            probe_current_e += np.squeeze(tdata.objectscurrente)[index]
            probe_current_i += np.squeeze(tdata.objectscurrenti)[index]
        probe_bias = np.squeeze(tdata.diagnostics[c.DIAG_PROBE_POT])
        return time, probe_bias, probe_current_e, probe_current_i

    def get_wall_potential(self, tdata=None):
        if tdata is None:
            tdata = self.tdata

        if c.DIAG_WALL_POT in tdata.diagnostics:
            wall_pot = np.squeeze(tdata.diagnostics[c.DIAG_WALL_POT])
        else:
            wall_pot = np.zeros_like(np.squeeze(tdata.diagnostics[c.DIAG_PROBE_POT]))
        return wall_pot

    def _homogenise_groupby_downsample(self, reduced_bin_fract=None, upper_voltage_limit=-15):
        """
        This homogeniser implementation is for simulations run with v2.14 and
        above. The voltage and current arrays should be the same length upon
        loading, which simplifies the homogenisation process considerably, but
        requires binning and averaging values of current 'measured' at the same
        voltage.

        """
        time, probe_bias, probe_current_e, probe_current_i = self.get_tdata_raw_iv(self.tdata)
        probe_current_tot = probe_current_e + probe_current_i

        wall_pot = self.get_wall_potential()

        assert isinstance(probe_current_tot, np.ndarray)

        # sweep_length = self.parser.get_sweep_length(len(probe_current_tot), probe_bias)
        sweep_length = 0

        sweep_slc = slice(sweep_length, len(probe_current_tot))
        probe_bias = np.squeeze(probe_bias)

        df = pd.DataFrame({
            'voltage': probe_bias[sweep_slc],
            'voltage_wall': wall_pot[sweep_slc],
            'current': probe_current_tot[sweep_slc],
            'current_e': probe_current_e[sweep_slc],
            'current_i': probe_current_i[sweep_slc],
        })

        voltage_groups = df.groupby('voltage').groups
        sizes = []
        for voltage_group in voltage_groups:
            voltage_group_size = len(voltage_groups[voltage_group])
            sizes.append(voltage_group_size)

        bin_size = int(np.median(sizes))

        # Group by voltages and discard those that are not the median length, i.e. the length of each step.
        filtered_df = df.groupby('voltage').filter(lambda x: len(x) == bin_size)
        if reduced_bin_fract:
            bin_size = int(reduced_bin_fract * bin_size)
            filtered_df = filtered_df.groupby('voltage').tail(bin_size)

        smoothed_df = filtered_df.groupby('voltage').mean()
        uncertainty = filtered_df.groupby('voltage').std()

        smoothed_df['current'] = -smoothed_df['current']
        smoothed_df['d_current'] = uncertainty['current'].values / np.sqrt(bin_size)
        smoothed_df['d_current_e'] = uncertainty['current_e'].values / np.sqrt(bin_size)
        smoothed_df['d_current_i'] = uncertainty['current_i'].values / np.sqrt(bin_size)

        # Make approximate time array
        try:
            dt = np.squeeze(self.tdata.dt)
        except:
            dt = 1
        smoothed_df['time'] = np.arange(0, smoothed_df['current'].size) * bin_size * dt

        # This moves the voltage from being index to being a column
        smoothed_df = smoothed_df.reset_index()

        n_particles = np.squeeze(self.tdata.nz * self.tdata.ny * self.tdata.npc)
        poisson_err = np.sqrt(n_particles) / n_particles

        smoothed_df = smoothed_df.drop(smoothed_df.index[-2:])
        smoothed_df = smoothed_df.loc[smoothed_df['voltage'] > upper_voltage_limit]

        sweep_data = IVData.from_dataset(smoothed_df, separate_current_fl=True)
        raw_data = IVData(probe_bias[:-1], probe_current_tot, time, sigma=poisson_err*probe_current_tot,
                          e_current=probe_current_e, i_current=probe_current_i)

        if self.store_dataframe_fl:
            self.iv_df = smoothed_df

        return sweep_data, raw_data

    def _homogenise_prepend_downsample(self, backups=None):
        """
        This method is for simulations run using v2.13 and below. Uses stored
        tdata and inputparser to get relevant simulation timing data. This
        involves prepending a requisite number of zeroes to the voltage array
        to make it a complete representation of the simulation time, and then
        downsampling this voltage to match the current array.

        """
        if backups is None or len(backups) <= 1:
            # If no backup directory, or only a single backup directory, exists then carry on as normal
            time, probe_bias, probe_current_e, probe_current_i = self.get_tdata_raw_iv(self.tdata)
            probe_current_tot = probe_current_e + probe_current_i
        else:
            # If multiple backup t-files need to be stitched together then create concatenated current/time/bias arrays
            concatted_time = np.array([])
            concatted_current = np.array([])
            concatted_current_i = np.array([])
            concatted_current_e = np.array([])
            concatted_bias = np.array([])
            prev_current = 0.0

            for i, bu_folder in enumerate(backups):
                # Loop through backups and stitch them together
                bu_probe_current = np.array([])

                print(f'Loading backup {bu_folder.name} ({i + 1} of {len(backups)}) for current and bias concatenation')
                bu_tfile_path, _ = self.get_ta_filenames(bu_folder)

                # Check whether log is completed
                bu_log_files = list(bu_folder.glob('log.ongoing.out'))
                if len(bu_log_files) > 0:
                    bu_log_file = bu_log_files[0]
                    final_percentage = self.get_percentage_completed(bu_log_file)
                else:
                    final_percentage = 0.0

                bu_tdata = sd.Spice2TData(bu_folder / bu_tfile_path, variable_names=[sd.T,
                                                                                     c.DIAG_PROBE_POT,
                                                                                     sd.OBJECTSCURRENTE,
                                                                                     sd.OBJECTSCURRENTI,
                                                                                     sd.OBJECTSENUM,
                                                                                     sd.OBJECTS])
                bu_time, bu_probe_bias, bu_probe_current_e, bu_probe_current_i = self.get_tdata_raw_iv(bu_tdata)
                bu_probe_current = bu_probe_current_e + bu_probe_current_i

                finished_fl = False

                if i == 0:
                    # If first step then whole current array is left for concatting and final current value is stored
                    # for next iteration to assist in stripping
                    prev_current = bu_probe_current[-1]

                if round(np.max(bu_probe_bias), 2) >= self.COMPLETION_VOLTAGE \
                        or self.parser.has_sufficient_sweep(final_percentage,
                                                            voltage_threshold=self.COMPLETION_VOLTAGE) \
                        or i + 1 == len(backups):
                    # If completed, or final backup, then strip off the end spike, zeroes and appended IV, if present
                    print(f'Detected finish, stripping tail for backup {i+1} of {len(backups)}')
                    finished_fl = True
                    bu_probe_current, end_point = self.strip_tail(bu_probe_current)
                    bu_probe_current_e = bu_probe_current_e[:end_point]
                    bu_probe_current_i = bu_probe_current_i[:end_point]

                if i >= 1:
                    # If not the first step then strip the leading zeroes/spikes
                    print(f'Stripping head for backup {i+1} of {len(backups)}')
                    bu_probe_current, start_point = self.strip_head(bu_probe_current, prev_current)
                    prev_current = bu_probe_current[-1]
                    bu_probe_current_i = bu_probe_current_i[start_point:]
                    bu_probe_current_e = bu_probe_current_e[start_point:]

                concatted_current = np.append(concatted_current, bu_probe_current)
                concatted_current_e = np.append(concatted_current_e, bu_probe_current_e)
                concatted_current_i = np.append(concatted_current_i, bu_probe_current_i)

                if finished_fl:
                    # Final version of voltage is stored in completed run, may need some minor pruning
                    concatted_bias = self.prune_voltage(bu_probe_bias)
                    concatted_time = np.arange(len(concatted_current))
                    break

            probe_current_tot = concatted_current
            probe_current_e = concatted_current_e
            probe_current_i = concatted_current_i
            probe_bias = concatted_bias
            time = concatted_time

        # Prepend missing elements to make array cover the same timespan as the builtin diagnostics and then
        # down-sample to get an array the same size as probe_current
        try:
            n = len(probe_bias)
            M = len(probe_current_tot)
        except TypeError as e:
            print('WARNING: Was not able to homogenise as the probe current or bias data is malformed.')
            return None, None

        N, r = self.parser.get_scaling_values(n, M)

        leading_values = np.zeros(N, dtype=np.int) + probe_bias[0]
        probe_bias_extended = np.concatenate([leading_values, probe_bias])[0:-r:r]

        n_particles = np.squeeze(self.tdata.nz * self.tdata.ny * self.tdata.npc)
        poisson_err = np.sqrt(n_particles) / n_particles

        # Extract the voltage and current for the sweeping region.
        sweep_length = self.parser.get_sweep_length(M, probe_bias_extended)
        t_sweep = time[sweep_length:]
        V_sweep = probe_bias_extended[sweep_length:]
        I_i_sweep = probe_current_i[sweep_length:]
        I_e_sweep = probe_current_e[sweep_length:]
        I_sweep = probe_current_tot[sweep_length:]
        sigma = I_sweep * poisson_err

        sweep_data = IVData(V_sweep, I_sweep, t_sweep, sigma=sigma, e_current=I_e_sweep, i_current=I_i_sweep)
        raw_data = IVData(probe_bias_extended, probe_current_tot, time, sigma=poisson_err*probe_current_tot,
                          e_current=probe_current_e, i_current=probe_current_i)

        if self.store_dataframe_fl:
            self.iv_df = sweep_data.to_dataframe(columns=[c.TIME, c.POTENTIAL, c.CURRENT,
                                                          c.SIGMA, c.ELEC_CURRENT, c.ION_CURRENT])

        return sweep_data, raw_data

    # Mapping of versions to homogeniser implementations
    VERSION_HOMOGENISERS = {
        2.13: _homogenise_prepend_downsample,
        2.14: _homogenise_groupby_downsample,
        3.05: _homogenise_groupby_downsample,
    }


class FailedSimulationException(Exception):
    pass

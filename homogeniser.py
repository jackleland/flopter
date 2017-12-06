from abc import ABC, abstractmethod
from scipy.io import loadmat
from normalisation import Denormaliser, TIME
from datatypes import IVData
import os
import numpy as np


class Homogeniser(ABC):
    """
    Abstract base class for the Homogeniser object.

    Takes data from some source and homogenises it for analysis within flopter. This class should be inherited from,
    and the homogenise and read_data methods overridden, in order to make an additional data source for flopter.

    Can separately be created and fed data using the set_data() method.
    """

    def __init__(self, source, filename=None, data=None):
        self.source = source
        self.data = data
        if filename and isinstance(filename, (str, os.PathLike)):
            self.filename = filename
            if not data:
                self.read_data()
        else:
            self.filename = None

    def set_filename(self, filename):
        self.filename = filename

    def get_filename(self):
        return self.filename

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def _prehomogenise_checks(self):
        if not self.data and isinstance(self.filename, (str, os.PathLike)):
            self.read_data()
        elif not self.data:
            raise ValueError("No data to homogenise")

    @abstractmethod
    def read_data(self):
        """
        Abstract method for reading data from Homogeniser.filename into the Homogeniser.data container. Note that
        data can be set directly with its setter 'set_data()'.

        Should not return anything, but instead populate the internal class variable Homogeniser.data
        """
        pass

    @abstractmethod
    def homogenise(self):
        """
        Abstract method for homogenising the data stored in Homogeniser.data
        :return:
            - An IV_Data object.
            - Raw data, preferably in a dictionary.
        """
        pass


class Spice2Homogeniser(Homogeniser):

    _SWEEP_LOWER = -9.95
    _SWEEP_UPPER = 10.05
    _PROBE_PARAMETER = 3

    def __init__(self, input_filename=None, denormaliser=None, **kwargs):
        super().__init__('Spice2', **kwargs)
        self.input_filename = input_filename
        self.denormaliser = denormaliser
        if not denormaliser and input_filename:
            self.input_filename = input_filename
            self.denormaliser = Denormaliser(input_filename)
        elif not denormaliser and not input_filename:
            raise ValueError('Need one of \'input_filename\' or \'denormaliser\' to not be None')
        self.parser = self.denormaliser.parser

    def read_data(self):
        self.data = loadmat(self.filename)

    def homogenise(self):
        self._prehomogenise_checks()

        # Extract relevant arrays from the matlab file
        probe_index = self.get_probe_index()

        time = self.denormaliser(np.squeeze(self.data['t'])[:-1], TIME)
        probe_current_e = np.squeeze(self.data['objectscurrente'])[probe_index]
        probe_current_i = np.squeeze(self.data['objectscurrenti'])[probe_index]
        probe_bias = np.squeeze(self.data['ProbePot'])
        probe_current_tot = probe_current_i + probe_current_e

        # Prepend missing elements to make array cover the same timespan as the builtin diagnostics and then
        # down-sample to get an array the same size as probe_current
        n = len(probe_bias)
        M = len(probe_current_tot)
        N, r = self.get_scaling_values(n, M)
        print(N, r)

        leading_values = np.zeros(N, dtype=np.int) + probe_bias[0]
        probe_bias_extended = np.concatenate([leading_values, probe_bias])[0:-r:r]

        # Extract the voltage and current for the sweeping region.
        sweep_length = self.get_sweep_length(M, probe_bias_extended)
        V_sweep = probe_bias_extended[sweep_length:]
        I_i_sweep = probe_current_i[sweep_length:]
        I_e_sweep = probe_current_e[sweep_length:]
        I_sweep = probe_current_tot[sweep_length:]

        sweep_data = IVData(V_sweep, I_sweep, time,
                            e_current=I_e_sweep, i_current=I_i_sweep)
        raw_data = IVData(probe_bias_extended, probe_current_tot, time,
                          e_current=probe_current_e, i_current=probe_current_i)

        return sweep_data, raw_data

    def get_probe_index(self):
        """
        Returns the index of the object within the simulation which is acting as a probe i.e. has 'param1' set to 3.
        The index can be used to reference the correct array in objectscurrent, for example. Assumes that the first
        probe found in the input file is the only probe.
        :return: {int} Index of probe in simulation object array.
        """
        num_blocks_section = self.parser['num_blocks']
        n = 0
        for shape in num_blocks_section:
            n_shape = self.parser.getint('num_blocks', shape)
            n += n_shape
            if n_shape > 0:
                shape_name = shape[:-1]
                for i in range(n_shape):
                    section = self.parser[shape_name + str(i)]
                    if int(section['param1']) == self._PROBE_PARAMETER:
                        return (n - n_shape) + i
        raise ValueError('Could not find a shape set to sweep voltage')

    def get_scaling_values(self, len_diag, len_builtin):
        """
        Calculates the scaling values (n' and r) which are needed to extend the diagnostic outputs to the right length
        and downsample them.
        :param len_diag:    length of raw diagnostic output array   (n)
        :param len_builtin: length of builtin output array          (M)
        :return n_leading:  size of array to prepend onto the diagnostic array
        :return ratio:      ratio of extended diagnostic output array to builtin output array (e.g. objectscurrent):
        """
        t_c = self.parser.getfloat('geom', 'tc')
        t_p = self.parser.getfloat('geom', 'tp')
        # t_c as a fraction of whole time
        t = t_c/t_p

        n_leading = t * len_diag / (1 - t)
        ratio = len_diag/(len_builtin*(1-t))
        return int(n_leading), int(ratio)

    def get_sweep_length(self, len_builtin, raw_voltage):
        t_a = self.parser.getfloat('geom', 'ta')
        t_p = self.parser.getfloat('geom', 'tp')
        # t_a as a fraction of whole time
        t = t_a / t_p

        sweep_length = int(t * len_builtin)

        initial_v = raw_voltage[0]
        if not self._is_within_bounds(initial_v, self._SWEEP_LOWER):
            corr_sweep_length = sweep_length
            while raw_voltage[corr_sweep_length] == initial_v and corr_sweep_length < len(raw_voltage):
                corr_sweep_length += 1
            sweep_length = corr_sweep_length
        return sweep_length

    @staticmethod
    def _is_within_bounds(value, comparison):
        return (value > comparison - 0.01) and (value < comparison + 0.01)

    def get_probe_geometry(self):
        # TODO: (25/10/2017) Write function to retrieve probe geometry from parser. Probably requires the definition
        # TODO: of a probe-geometry class first.
        pass










import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.io import loadmat

from classes.ivdata import IVData
from classes.spicedata import Spice2Data
from inputparser import InputParser
from constants import DIAG_PROBE_POT, INF_SEC_SHAPES, INF_SEC_GEOMETRY, INF_SEC_CONTROL, INF_SWEEP_PARAM
import constants as c


class Homogeniser(ABC):
    """
    Abstract base class for the Homogeniser object.

    Takes data from some source and homogenises it for analysis within flopter. This class should be inherited from,
    and the homogenise and read_data methods overridden, in order to make an additional data source for flopter.

    Can separately be created and fed data using the set_data() method.
    """

    def __init__(self, source, data_filename=None, data=None):
        self.source = source
        self.data = data
        if data_filename and isinstance(data_filename, (str, os.PathLike)):
            self.data_filename = data_filename
            if not data:
                self.read_data()
        else:
            self.data_filename = None

    def set_data_filename(self, filename):
        self.data_filename = filename

    def get_data_filename(self):
        return self.data_filename

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def _prehomogenise_checks(self):
        if not self.data and isinstance(self.data_filename, (str, os.PathLike)):
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

    def __init__(self, input_parser=None, input_filename=None, **kwargs):
        super().__init__('Spice2', **kwargs)

        if input_parser is not None and input_parser:
            self.parser = input_parser
            self.input_filename = None
        elif self.input_filename and isinstance(input_filename, (str, os.PathLike)):
            self.input_filename = input_filename
            self.parser = InputParser(self.input_filename)
        else:
            raise ValueError('No valid InputParser object given or able to be created')

    def read_data(self):
        self.data = Spice2Data(self.data_filename)

    def homogenise(self):
        self._prehomogenise_checks()

        # Extract relevant arrays from the matlab file
        probe_index = self.get_probe_index()

        time = np.squeeze(self.data.t)[:-1]
        probe_current_e = np.squeeze(self.data.objectscurrente)[probe_index]
        probe_current_i = np.squeeze(self.data.objectscurrenti)[probe_index]
        probe_bias = np.squeeze(self.data.diagnostics[DIAG_PROBE_POT])
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
        # TODO: make this return a slice object with each sweeping object index
        """
        Returns the index of the object within the simulation which is acting as a probe i.e. has 'param1' set to 3.
        The index can be used to reference the correct array in objectscurrent, for example. Assumes that the first
        probe found in the input file is the only probe.
        :return: {int} Index of probe in simulation object array.
        """
        num_blocks_section = self.parser[INF_SEC_SHAPES]
        n = 0
        for shape in num_blocks_section:
            n_shape = self.parser.getint(INF_SEC_SHAPES, shape)
            n += n_shape
            if n_shape > 0:
                shape_name = shape[:-1]
                for i in range(n_shape):
                    section = self.parser[shape_name + str(i)]
                    if int(section[INF_SWEEP_PARAM]) == self._PROBE_PARAMETER:
                        return (n - n_shape) + i
        raise ValueError('Could not find a shape set to sweep voltage')

    def get_scaling_values(self, len_diag, len_builtin):
        """
        Calculates the scaling values (n' and r) which are needed to extend the diagnostic outputs to the right length
        and downsample them for homogenisation of SPICE IV sweeps
        :param len_diag:    length of raw diagnostic output array   (n)
        :param len_builtin: length of builtin output array          (M)
        :return n_leading:  size of array to prepend onto the diagnostic array
        :return ratio:      ratio of extended diagnostic output array to builtin output array (e.g. objectscurrent):
        """
        t_c = self.parser.getfloat(INF_SEC_GEOMETRY, c.INF_TIME_SWEEP)
        t_p = self.parser.getfloat(INF_SEC_GEOMETRY, c.INF_TIME_END)
        # t_c as a fraction of whole time
        t = t_c/t_p

        n_leading = t * len_diag / (1 - t)
        ratio = len_diag/(len_builtin*(1-t))
        return int(n_leading), int(ratio)

    def get_sweep_length(self, len_builtin, raw_voltage):
        t_a = self.parser.getfloat(INF_SEC_GEOMETRY, c.INF_TIME_AV)
        t_p = self.parser.getfloat(INF_SEC_GEOMETRY, c.INF_TIME_END)
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










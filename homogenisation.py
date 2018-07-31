import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.io import loadmat

from classes.ivdata import IVData
from classes.spicedata import Spice2TData
from inputparser import InputParser
from constants import DIAG_PROBE_POT, INF_SEC_SHAPES, INF_SEC_GEOMETRY, INF_SEC_CONTROL, INF_SWEEP_PARAM
import constants as c


class Homogeniser(ABC):
    # TODO: (31/07/18) This class is probably no longer necessary as this can be implemented within each individual
    # TODO: IVAnalyser - might be better for streamlined implementation for this to still exist though, as a plug-and-
    # TODO: play part of the analysis routine. At the very least data doesn't need to be stored in the homogeniser.
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
        self.data = Spice2TData(self.data_filename)

    def homogenise(self):
        """
        Homogenise function for creaiton of IVData objects for use in SPICE simulation analysis. Uses specified
        inputfile or inputparser to get relevant simulation timing data and then slices up the current and voltage
        traces accordingly. Also uses InputParser methods to evaluate probe collection objects.
        :param data:
        :return:
        """
        self._prehomogenise_checks()

        # Extract relevant arrays from the matlab file
        probe_indices = self.parser.get_probe_obj_indices()
        probe_current_e = 0.0
        probe_current_i = 0.0

        time = np.squeeze(self.data.t)[:-1]
        for index in probe_indices:
            probe_current_e += np.squeeze(self.data.objectscurrente)[index]
            probe_current_i += np.squeeze(self.data.objectscurrenti)[index]
        probe_bias = np.squeeze(self.data.diagnostics[c.DIAG_PROBE_POT])
        probe_current_tot = probe_current_i + probe_current_e

        # Prepend missing elements to make array cover the same timespan as the builtin diagnostics and then
        # down-sample to get an array the same size as probe_current
        n = len(probe_bias)
        M = len(probe_current_tot)
        N, r = self.parser.get_scaling_values(n, M)

        leading_values = np.zeros(N, dtype=np.int) + probe_bias[0]
        probe_bias_extended = np.concatenate([leading_values, probe_bias])[0:-r:r]

        # Extract the voltage and current for the sweeping region.
        sweep_length = self.parser.get_sweep_length(M, probe_bias_extended)
        V_sweep = probe_bias_extended[sweep_length:]
        I_i_sweep = probe_current_i[sweep_length:]
        I_e_sweep = probe_current_e[sweep_length:]
        I_sweep = probe_current_tot[sweep_length:]

        sweep_data = IVData(V_sweep, I_sweep, time,
                            e_current=I_e_sweep, i_current=I_i_sweep)
        raw_data = IVData(probe_bias_extended, probe_current_tot, time,
                          e_current=probe_current_e, i_current=probe_current_i)

        return sweep_data, raw_data
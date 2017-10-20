from abc import ABC, abstractmethod
from scipy.io import loadmat
from normalisation import Denormaliser, TIME
import os
import numpy as np


class IVData(object):

    def __init__(self, voltage, total_current, e_current, i_current, time):
        self.data = {
            "I": total_current,
            "V": voltage,
            "I_e": e_current,
            "I_i": i_current,
            "t": time
        }


class Homogeniser(ABC):
    """
        Abstract base class for the Homogeniser object.

        Takes data from some source and homogenises it for analysis within flopter.

        Can separately be created and fed data using the set_data() method.
    """

    def __init__(self, source, filename=None, data=None):
        self.source = source
        self.data = data
        if filename and isinstance(self.filename, (str, os.PathLike)):
            self.filename = filename
            if not data:
                self.read_data()
        else:
            self.filename = None

    def set_filename(self, filename):
        self.filename = filename

    def set_data(self, data):
        self.data = data

    def _prehomogenise_checks(self):
        if not self.data and isinstance(self.filename, (str, os.PathLike)):
            self.read_data()
        elif not self.data:
            raise ValueError("No data to homogenise")

    @abstractmethod
    def read_data(self):
        pass

    @abstractmethod
    def homogenise(self):
        pass


class Spice2Homogeniser(Homogeniser):

    def __init__(self, input_filename):
        super().__init__('Spice2')
        self.input_filename = input_filename
        self.denormaliser = Denormaliser(input_filename)

    def read_data(self):
        self.data = loadmat(self.filename)

    def homogenise(self):
        self._prehomogenise_checks()

        # Extract relevant arrays from the matlab file
        time = self.denormaliser(np.squeeze(self.data['t'])[:-1], TIME)
        probe_current_e = np.squeeze(self.data['objectscurrente'])[2]
        probe_current_i = np.squeeze(self.data['objectscurrenti'])[2]
        probe_bias = np.squeeze(self.data['ProbePot'])
        probe_current_tot = probe_current_i + probe_current_e

        iv_data = IVData(probe_bias, probe_current_tot, probe_current_e, probe_current_i, time)

        return iv_data, self.data







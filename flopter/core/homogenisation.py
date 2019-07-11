import os
from abc import ABC, abstractmethod


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


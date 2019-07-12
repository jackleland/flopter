from abc import ABC, abstractmethod


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

    @classmethod
    def create_from_file(cls, filename):
        # TODO: Implement a saving and loading system
        pass

    # @abstractmethod
    # def plot(self):
    #     pass
    #
    # @abstractmethod
    # def save(self):
    #     pass

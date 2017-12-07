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
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def save(self):
        pass


class IVFitter(ABC):
    """
    Abstract base class for a Fitter object, which describes a model for fitting an IV curve.
    """
    def __init__(self):
        self.params = {}

    @abstractmethod
    def fit_function(self, v, *parameters):
        pass

    @abstractmethod
    def fit(self):
        pass

    def get_params(self):
        return self.params

    @abstractmethod
    def get_param_index(self, label):
        pass

    @abstractmethod
    def get_temp_index(self):
        pass

    @abstractmethod
    def get_isat_index(self):
        pass

    @abstractmethod
    def get_a_index(self):
        pass

    @abstractmethod
    def get_vf_index(self):
        pass

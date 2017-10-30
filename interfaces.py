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
    def fit(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def save(self):
        pass


class IVFitter(ABC):
    """
    Abstract base class for a Fitter object, which fits a model to an IV curve within an IV analyser.
    """
    @abstractmethod
    def fit(self):
        pass

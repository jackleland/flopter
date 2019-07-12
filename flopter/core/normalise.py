from inspect import signature
from abc import ABC, abstractmethod
from flopter.core import constants as c


class Converter(ABC):
    """
    Base class for a converter object, which normalises or denormalises an input variable.

    Principle method of conversion is to call the converter object itself with the desired conversion_type
    as a parameter. Subclasses need to override the given conversion functions defined below
    """
    def __init__(self):
        # Conversion types - stored in a dictionary
        self.CONVERSION_TYPES = {
            c.CONV_POTENTIAL: self._convert_potential,
            c.CONV_CURRENT: self._convert_current,
            c.CONV_LENGTH: self._convert_length,
            c.CONV_TIME: self._convert_time,
            c.CONV_DENSITY: self._convert_density,
            c.CONV_MASS: self._convert_mass,
            c.CONV_CHARGE: self._convert_charge,
            c.CONV_TEMPERATURE: self._convert_temperature,
            c.CONV_FLUX: self._convert_flux,
            c.CONV_VELOCITY: self._convert_velocity,
        }

    def __call__(self, variable, conversion_type=c.CONV_POTENTIAL, additional_arg=None):
        """
        :param variable:            Variable to be converted. Must be scalable, i.e. a float or a numpy array
        :param conversion_type:     Which conversion type should be done (length, potential etc)
        :param additional_arg:      Optional argument that may be required in the conversion procedure. Can be a list of
                                    arguments.
        :return:                    Converted version of variable

        Calls the function corresponding to the conversion_type argument passed. Feeds in 'variable' as an
        input into this function as well as any additional args should they be needed.
        """
        if conversion_type in self.CONVERSION_TYPES:
            # Get conversion function and associated metadata
            function = self.CONVERSION_TYPES[conversion_type]
            function_sig = signature(function)

            # Check if function needs an additional argument
            if additional_arg is not None and len(function_sig.parameters) > 1:
                return function(variable, *additional_arg)
            elif len(function_sig.parameters) > 1:
                # Raise error if too few arguments provided
                raise TypeError('\'{a}\' function type requires {b} arguments: {c}'
                                .format(a=conversion_type, b=len(function_sig.parameters), c=str(function_sig)))
            else:
                return function(variable)
        else:
            raise TypeError('The \'function\' argument should be a conversion type - one of: {a}'
                            .format(a=list(self.CONVERSION_TYPES.keys())))

    @abstractmethod
    def _convert_potential(self, potential):
        pass

    @abstractmethod
    def _convert_current(self, current):
        pass

    @abstractmethod
    def _convert_length(self, length):
        pass

    @abstractmethod
    def _convert_time(self, time):
        pass

    @abstractmethod
    def _convert_density(self, density):
        pass

    @abstractmethod
    def _convert_velocity(self, velocity):
        pass

    @abstractmethod
    def _convert_mass(self, mass):
        pass

    @abstractmethod
    def _convert_charge(self, charge):
        pass

    @abstractmethod
    def _convert_temperature(self, temperature):
        pass

    @abstractmethod
    def _convert_flux(self, flux):
        pass



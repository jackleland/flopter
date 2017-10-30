import numpy as np
from inputparser import InputParser
from inspect import signature
from abc import ABC, abstractmethod

# Constants for conversion types
POTENTIAL = 'potential'
CURRENT = 'current'
LENGTH = 'length'
TIME = 'time'

# Physical constants
_EPSILON_0 = 8.85418782e-12  # m^-3 kg^-1 s^4 A^2
_ELEM_CHARGE = 1.60217662e-19 # C
_PROTON_MASS = 1.6726219e-27 # kg
_ION_MASS = 2.01410178 * _PROTON_MASS # kg


class Converter(ABC):
    """
    Base class for a converter object, which normalises or denormalises an input variable.

    Principle method of conversion is to call the converter object itself with the desired conversion_type
    as a parameter. Subclasses need to override the given conversion functions defined below
    """
    def __init__(self):
        # Conversion types - stored in a dictionary
        self.CONVERSION_TYPES = {
            POTENTIAL: self._convert_potential,
            CURRENT: self._convert_current,
            LENGTH: self._convert_length,
            TIME: self._convert_time
        }

    def __call__(self, variable, conversion_type, additional_arg=None):
        """
        :param variable:            Variable to be converted. Must be scalable, i.e. a float or a numpy array
        :param conversion_type:     Which conversion type should be done (legnth, potential etc)
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
                            .format(a=str(*self.CONVERSION_TYPES)))

    @abstractmethod
    def _convert_potential(self, potential):
        pass

    @abstractmethod
    def _convert_current(self, current, dt):
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


class Denormaliser(Converter):
    """
    Class for de-normalising values given by SPICE simulations.

    Stores an input parser object containing the used input file and then builds conversion values from this.
    """

    def __init__(self, dimensions=2, input_parser=None, input_filename=None):
        """
        Creates a denormaliser object by using InputParser to parse an input file for some parameters (principally n_e,
        T_e, B and N_pc) which are needed for the denormalisation process.

        InputParser either passed directly or created from a filename
        :param dimensions           Int - specify the number of dimensions to be used for calculation of areas/volumes
        :param [input_parser]:      InputParser object to be used, must have already read the commented section.
        :param [input_filename]:    Directory of input file to be parsed
        """
        super().__init__()
        if input_parser is not None and input_parser.has_commented_params():
            self.parser = input_parser
        elif input_filename is not None:
            self.parser = InputParser(input_filename=input_filename)
        else:
            raise ValueError('No valid InputParser object given or able to be created')

        if dimensions in [2, 3]:
            self.dimensions = dimensions
        else:
            raise ValueError('Number of dimensions should be 2 or 3')

        self.simulation_params = self.parser.get_commented_params()
        self.debye_length = np.sqrt((_EPSILON_0 * self.simulation_params['T_e'])
                                    / (_ELEM_CHARGE * self.simulation_params['n_e']))
        self.omega_i = ((_ELEM_CHARGE * self.simulation_params['B'])
                        / _ION_MASS)
        self.K = (self.simulation_params['n_e'] * self.debye_length**self.dimensions) / float(self.parser.get('geom', 'Npc')[:-1])

    def _convert_potential(self, potential):
        return potential * (self.simulation_params['T_e'])

    def _convert_current(self, current, dt):
        return ((_ELEM_CHARGE * self.K * self.omega_i) / dt) * current

    def _convert_length(self, length):
        return self.debye_length * length

    def _convert_time(self, time):
        return self.omega_i * time

    def _convert_density(self, density):
        return self.simulation_params['n_e'] * density


class Normaliser(Converter):
    """
    Class for normalising values for use in SPICE simulations.
    """
    def __init__(self):
        super().__init__()

    def _convert_length(self, length):
        super()._convert_length(length)

    def _convert_current(self, current, dt):
        super()._convert_current(current, dt)

    def _convert_time(self, time):
        super()._convert_time(time)

    def _convert_potential(self, potential):
        super()._convert_potential(potential)

    def _convert_density(self, density):
        super()._convert_density(density)

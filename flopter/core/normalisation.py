import numpy as np
from flopter.spice.inputparser import InputParser
from inspect import signature
from abc import ABC, abstractmethod
from flopter.core import constants as c
from flopter.core.ivdata import IVData

# Physical constants
BOLTZMANN = 1.38064852e-23  # m^2 kg s^-2 K^-1
EPSILON_0 = 8.85418782e-12  # m^-3 kg^-1 s^4 A^2
ELEM_CHARGE = 1.60217662e-19 # C
PROTON_MASS = 1.6726219e-27 # kg
DEUTERIUM_MASS = 2.01410178 * PROTON_MASS # kg
ELECTRON_MASS = 9.10938356e-31 # kg
ATOMIC_MASS_UNIT = ELECTRON_MASS * 1822.888486
P_E_MASS_RATIO = PROTON_MASS / ELECTRON_MASS
I_E_MASS_RATIO = DEUTERIUM_MASS / ELECTRON_MASS


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


class Denormaliser(Converter):
    """
    Class for de-normalising values given by SPICE simulations.

    Stores an input parser object containing the used input file and then builds conversion values from this.
    """

    def __init__(self, dt=1.0, dimensions=2, input_parser=None, input_filename=None, edge_ratio=1, temperature=None):
        """
        Creates a denormaliser object by using InputParser to parse an input file for some parameters (principally n_e,
        T_e, B and N_pc) which are needed for the denormalisation process.

        InputParser either passed directly or created from a filename
        :param dt                   Float - value of dt from simulation (cannot be ascertained from the input file)
        :param dimensions           Int - specify the number of dimensions to be used for calculation of areas/volumes
        :param [input_parser]:      InputParser object to be used, must have already read the commented section.
        :param [input_filename]:    Directory of input file to be parsed

        One of input_parser or input_filename is required
        """
        super().__init__()

        # Check for input parser or create one from given filename
        if input_parser is not None and input_parser.has_commented_params():
            parser = input_parser
        elif input_filename is not None:
            parser = InputParser(input_filename=input_filename)
        else:
            raise ValueError('No valid InputParser object given or able to be created')

        # Check that valid number of dimensions given
        if dimensions in [2, 3]:
            self.dimensions = dimensions
        else:
            raise ValueError('Number of dimensions should be 2 or 3')

        self.dt = dt

        # Save parameters from commented params
        self.simulation_params = parser.get_commented_params()
        if temperature:
            self.temperature = temperature
        elif 'T_e' in self.simulation_params.keys():
            self.temperature = self.simulation_params[c.ELEC_TEMP]
        else:
            self.temperature = 1

        if edge_ratio:
            self.edge_ratio = edge_ratio
            self.sheath_edge_temp = self.temperature * edge_ratio

        self.ksi = float(parser.get(c.INF_SEC_PLASMA, c.INF_KSI))
        self.mu = float(parser.get(c.INF_SEC_PLASMA, c.INF_MU))
        self.tau = float(parser.get(c.INF_SEC_PLASMA, c.INF_TAU))

        self.debye_length = np.sqrt((EPSILON_0 * self.temperature)
                                    / (ELEM_CHARGE * self.simulation_params[c.ELEC_DENS]))
        self.omega_i = ((ELEM_CHARGE * self.simulation_params[c.INF_MAGNETIC_FIELD])
                        / (self.mu * ELECTRON_MASS))
        self.K = ((self.simulation_params[c.ELEC_DENS] * self.debye_length**self.dimensions)
                  / float(parser.get(c.INF_SEC_GEOMETRY, c.INF_PART_PER_CELL)[:-1]))

        self.CONVERSION_TYPES[c.CONV_IV] = self._convert_iv_data
        self.CONVERSION_TYPES[c.CONV_DIST_FUNCTION] = self._convert_distribution_function

    def __call__(self, variable, conversion_type=c.CONV_IV, additional_arg=None):
        return super().__call__(variable, conversion_type=conversion_type, additional_arg=additional_arg)

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_se_temperature(self, ratio):
        self.edge_ratio = ratio
        self.sheath_edge_temp = self.temperature * self.edge_ratio

    def _convert_potential(self, potential):
        return potential * self.temperature

    def _convert_current(self, current):
        return ((ELEM_CHARGE * self.K * self.omega_i) / self.dt) * current

    def _convert_length(self, length):
        return self.debye_length * length

    def _convert_time(self, time):
        return (1 / self.omega_i) * time

    def _convert_velocity(self, velocity):
        return (self.debye_length * self.omega_i) * velocity

    def _convert_density(self, density):
        return self.simulation_params[c.ELEC_DENS] * density

    def _convert_mass(self, mass):
        return self.mu * mass

    def _convert_charge(self, charge):
        return ELEM_CHARGE * charge

    def _convert_temperature(self, temperature):
        return self.temperature * temperature

    def _convert_flux(self, flux):
        return self.simulation_params[c.ELEC_DENS] * self.debye_length * self.omega_i * flux

    def _convert_iv_data(self, iv_data):
        time = self._convert_time(iv_data[c.TIME])
        potential = self._convert_potential(iv_data[c.POTENTIAL])
        current = self._convert_current(iv_data[c.CURRENT])
        current_e = self._convert_current(iv_data[c.ELEC_CURRENT])
        current_i = self._convert_current(iv_data[c.ION_CURRENT])

        return IVData(potential, current, time, i_current=current_i, e_current=current_e)

    def _convert_distribution_function(self, dist_function):
        conversion_factor = np.sqrt(self.temperature / DEUTERIUM_MASS) / self.ksi * np.sqrt(200 / self.mu)
        return conversion_factor * dist_function

    def get_mu(self):
        return self.mu


class Normaliser(Converter):
    """
    Class for normalising values for use in SPICE simulations.
    """
    def __init__(self):
        super().__init__()

    def _convert_mass(self, mass):
        super()._convert_mass(mass)

    def _convert_charge(self, charge):
        super()._convert_charge(charge)

    def _convert_temperature(self, temperature):
        super()._convert_temperature(temperature)

    def _convert_flux(self, flux):
        super()._convert_flux(flux)

    def _convert_length(self, length):
        super()._convert_length(length)

    def _convert_current(self, current):
        super()._convert_current(current)

    def _convert_time(self, time):
        super()._convert_time(time)

    def _convert_potential(self, potential):
        super()._convert_potential(potential)

    def _convert_density(self, density):
        super()._convert_density(density)
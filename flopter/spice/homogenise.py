import os
import numpy as np
from flopter.core.homogenise import Homogeniser
from flopter.core.ivdata import IVData
from flopter.spice.tdata import Spice2TData
from flopter.spice.inputparser import InputParser
from flopter.core import constants as c


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
        try:
            n = len(probe_bias)
            M = len(probe_current_tot)
        except TypeError as e:
            print('WARNING: Was not able to homogenise as the probe current or bias data is malformed.')
            return None, None

        N, r = self.parser.get_scaling_values(n, M)

        leading_values = np.zeros(N, dtype=np.int) + probe_bias[0]
        probe_bias_extended = np.concatenate([leading_values, probe_bias])[0:-r:r]

        n_particles = np.squeeze(self.data.nz * self.data.ny * self.data.npc)
        poisson_err = np.sqrt(n_particles) / n_particles

        # Extract the voltage and current for the sweeping region.
        sweep_length = self.parser.get_sweep_length(M, probe_bias_extended)
        V_sweep = probe_bias_extended[sweep_length:]
        I_i_sweep = probe_current_i[sweep_length:]
        I_e_sweep = probe_current_e[sweep_length:]
        I_sweep = probe_current_tot[sweep_length:]
        sigma = I_sweep * poisson_err

        sweep_data = IVData(V_sweep, I_sweep, time, sigma=sigma,
                            e_current=I_e_sweep, i_current=I_i_sweep)
        raw_data = IVData(probe_bias_extended, probe_current_tot, time,
                          e_current=probe_current_e, i_current=probe_current_i,
                          sigma=poisson_err*probe_current_tot)

        return sweep_data, raw_data

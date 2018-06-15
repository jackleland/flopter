import numpy as np
import external.readfastadc as radc
import constants as c


class MagnumAdcConfig(object):
    def __init__(self, sample_rate, sample_time=None, num_samples=None, channels=None):
        self.sample_rate = sample_rate
        if sample_time:
            self.sample_time = sample_time
            self.num_samples = sample_rate * sample_time
        elif num_samples:
            self.num_samples = num_samples
            self.sample_time = num_samples / sample_rate
        else:
            raise ValueError('Initiation of MagnumAdcConfig requires one of sample_time or num_samples to not be None.')

        if not channels:
            raise ValueError('Channel list (with amplification and offset) required for schema definition.')
        else:
            assert isinstance(channels, dict)
            self.channels = channels

    @classmethod
    def parse_from_header(cls, header):
        # TODO: Implement this? May not be necessary as header container may suffice
        pass


class MagnumAdcData(object):
    _DEFAULT_CHANNELS = {
        0: [1.0, 0],
        2: [1.0, 0],
        5: [1.0, 0],
        6: [1.0, 0],
        7: [1.0, 0]
    }
    _DEFAULT_SCHEMA = MagnumAdcConfig(250000.0, 20.0, channels=_DEFAULT_CHANNELS)

    def __init__(self, full_path, file):
        self.full_path = full_path
        self.filename = file
        self.header, self.data = radc.process_adc_file(full_path, file)
        self.channels = list(self.data.keys())

        # Create time array
        self.time = np.linspace(0, (self.header[c.MAGADC_NUM] / self.header[c.MAGADC_FREQ]), self.header[c.MAGADC_NUM])


from normalisation import Denormaliser, Normaliser
import constants as c


class FlopterData(object):
    def __init__(self, data, type=None, converter=None):
        self.data = data
        self.type = type
        self.converter = converter

    def set_converter(self, converter):
        self.converter = converter

    def convert(self, conversion_type=c.CONV_CURRENT):
        if not self.converter:
            raise ValueError('No converter specified')

        return self.converter(self.data, conversion_type)

    def denormalise(self, conversion_type=None):
        assert isinstance(self.converter, Denormaliser)
        return self.convert(conversion_type=conversion_type)

    def normalise(self, conversion_type=None):
        assert isinstance(self.converter, Normaliser)
        return self.convert(conversion_type=conversion_type)

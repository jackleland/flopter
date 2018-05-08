import flopter as fl
import numpy as np
import pathlib as pth
import matplotlib.pyplot as plt
import classes.magnumdata as md


class Magopter(fl.IVAnalyser):
    _FOLDER_STRUCTURE = '/Data/Magnum/'

    def __init__(self, directory, filename):
        super().__init__()
        # Check for leading/trailing forward slashes?
        self.directory = directory
        self.file = filename
        self.full_path = '{}{}{}{}'.format(pth.Path.home(), self._FOLDER_STRUCTURE, directory, filename)

        self.m_data = md.MagnumAdcData(self.full_path, filename)

    def prepare(self):
        super().prepare()

    def trim(self):
        super().trim()

    def denormalise(self):
        super().denormalise()

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)

    @staticmethod
    def trim_generic(data, trim_beg=0.0, trim_end=1.0):
        return super().trim_generic(data, trim_beg, trim_end)

    @classmethod
    def create_from_file(cls, filename):
        super().create_from_file(filename)


if __name__ == '__main__':
    folder = '2018-05-01_Leland/'
    file = '2018-05-01_12h_55m_47s_TT_06550564404491814477.adc'
    magopter = Magopter(folder, file)

    print(magopter.m_data.channels)
    # length = len(magopter.t_file)
    # for i in range(1, 20):
    #     split = int(length / i)
        # plt.figure()
        # plt.title('i = {}'.format(i))
        # plt.log
        # for j in range(i):
        #     plt.semilogy(magopter.t_file[j*split:j+1*split], label='j = {}'.format(j))

    # plt.show()

    m = 101000
    n = 115500
    print(magopter.m_data.data[5].shape)
    plt.figure()
    # for ch in magopter.m_data.channels:
    #     plt.plot(magopter.m_data.data[ch])
    plt.plot(magopter.m_data.data[5][m:n], magopter.m_data.data[6][m:n])
    # plt.plot(np.linspace(m, n, n - m), np.log(np.abs(magopter.t_file[m:n])))

    plt.figure()
    plt.plot(magopter.m_data.data[5][m:n])
    plt.plot(magopter.m_data.data[6][m:n])
    plt.show()

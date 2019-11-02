import flopter.core.constants
import numpy as np
import pathlib as pth


def get_ksi(density, temp, q, m_i, B):
    debye = get_lambda_d(density, temp)
    larmor = get_larmor_r(temp, B, m_i, q)
    return larmor / debye


def get_lambda_d(density, temp):
    return np.sqrt((flopter.core.constants.EPSILON_0 * temp) / (flopter.core.constants.ELEM_CHARGE * density))


def get_larmor_r(temp, B, m_i, q):
    return np.sqrt((flopter.core.constants.ELEM_CHARGE * temp * m_i)) / (q * flopter.core.constants.ELEM_CHARGE * B)


def get_mu(m_i, m_e):
    return m_i / m_e


def is_code_output_dir(directory):
    # TODO: (2019-09-12) -  This could be split into two functions, one which returns output_files, particle files
    #  etc. and another which runs the test. Then a single robust method can be used for this and get_ta_filenames()
    if not isinstance(directory, pth.Path) and isinstance(directory, str):
        directory = pth.Path(directory)
    if directory.is_dir():
        output_files = set(directory.glob('t-*.mat'))
        particle_files = set(directory.glob('t-*[0-9][0-9].mat'))
        a_file = set(directory.glob('[!t-]*[!.2d].mat'))
        t_file = output_files - particle_files

        # Directory is deemed to be a spice directory if it has a t-file and particle_files
        return len(t_file) == 1 and len(particle_files) > 1
    else:
        print(f'{directory} is not a directory')
        return False

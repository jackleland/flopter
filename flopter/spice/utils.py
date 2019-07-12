import flopter.core.constants
from flopter.core import normalise as nrm
import numpy as np


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
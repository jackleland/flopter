import numpy as np

import flopter.core.constants
from flopter.core import normalise as nrm
from abc import ABC, abstractmethod


class LangmuirProbe(ABC):
    @abstractmethod
    def get_collection_area(self, alpha):
        pass

    @abstractmethod
    def get_analytical_iv(self, voltage, v_f, alpha, temp, dens):
        pass

    def get_density(self, sat_current, temperature, alpha, gamma_i=1, mass=1):
        c_s = sound_speed(temperature, gamma_i=gamma_i, mass=mass)
        A_coll = self.get_collection_area(alpha)
        return electron_density(sat_current, c_s, A_coll)

    def get_d_density(self, sat_current, d_sat_current, temperature, d_temperature, alpha, gamma_i=1, mass=1):
        c_s = sound_speed(temperature, gamma_i=gamma_i, mass=mass)
        A_coll = self.get_collection_area(alpha)
        n_e = electron_density(sat_current, c_s, A_coll)

        d_c_s = d_sound_speed(c_s, temperature, d_temperature)
        d_A_coll = np.abs(self.get_collection_area(alpha + np.radians(0.8)) - A_coll)

        return d_electron_density(n_e, c_s, d_c_s, A_coll, d_A_coll, sat_current, d_sat_current)


class AngledTipProbe(LangmuirProbe):
    def __init__(self, a, b, L, g, d_perp, theta_f, theta_p):
        self.a = a
        self.b = b
        self.L = L
        self.g = g
        self.d_perp = d_perp
        self.theta_f = theta_f
        self.theta_p = theta_p

    def get_collection_area(self, alpha):
        return calc_probe_collection_area(self.a, self.b, self.L, self.g, self.d_perp, alpha, self.theta_p, self.theta_f)

    def get_analytical_iv(self, voltage, v_f, alpha, temp, dens, mass=1, gamma_i=1.0, c_1=0.9, c_2=0.6, print_fl=False):
        return analytical_iv_curve(voltage, v_f, temp, dens, alpha, self.get_collection_area(alpha), c_1=c_1, c_2=c_2,
                                   gamma_i=gamma_i, mass=mass, L=self.L, g=self.g, print_fl=print_fl)


class FlushCylindricalProbe(LangmuirProbe):
    def __init__(self, radius, g, d_perp):
        self.radius = radius
        self.d_perp = d_perp
        self.g = g

    def get_collection_area(self, alpha):
        d = (self.d_perp / np.sin(alpha)) - self.g
        theta_c = 2 * np.arccos((self.radius - d) / self.radius)
        A_coll = np.sin(alpha) * ((np.pi * self.radius**2) - (self.radius**2 / 2)) * (theta_c - 2 * np.sin(theta_c))
        return A_coll

    def get_analytical_iv(self, voltage, v_f, alpha, temp, dens, mass=1, gamma_i=1.0, c_1=0.9, c_2=0.6, print_fl=False):
        analytical_iv_curve(voltage, v_f, temp, dens, alpha, self.get_collection_area(alpha), c_1=c_1, c_2=c_2,
                            gamma_i=gamma_i, mass=mass, L=(2 * self.radius), g=self.g, print_fl=print_fl)


def calc_probe_collection_area(a, b, L, g, d_perp, theta_perp, theta_p, theta_f, print_fl=False):
    # d = max(0, ((d_perp - (g * np.tan(theta_perp)))
    #         / (np.sin(theta_p) + (np.tan(theta_perp) * np.cos(theta_p)))))
    # h_coll = max(0, (g * np.tan(theta_perp) - d_perp) * np.cos(theta_perp))
    d, h_coll = calc_probe_exposed_lengths(g, d_perp, theta_perp, theta_p)
    if print_fl:
        print("d = {}, h_coll = {}".format(d, h_coll))
    L_exp = (L / np.cos(theta_p)) - d
    return (0.5 * (a + b - (d / np.tan(theta_f))) * L_exp * np.sin(theta_perp + theta_p)) + (h_coll * b)


def calc_probe_exposed_lengths(g, d_perp, theta_perp, theta_p):
    d = np.array((d_perp - (g * np.tan(theta_perp)))
                 / (np.sin(theta_p) + (np.tan(theta_perp) * np.cos(theta_p)))).clip(min=0)
    h_coll = np.array((g * np.tan(theta_perp) - d_perp) * np.cos(theta_perp)).clip(min=0)
    return d, h_coll


def calc_probe_collection_A_alt(a, b, L, theta_perp, theta_p):
    return (L / np.cos(theta_p)) * (a + b) * 0.5 * np.sin(theta_p + theta_perp)


def analytical_iv_curve(voltage, v_f, temp, dens, alpha, A_coll, c_1=0.9, c_2=0.6, gamma_i=1.0, mass=1, L=1, g=0.5,
                        print_fl=False):
    T_i = temp
    T_e = temp
    lambda_D = np.sqrt((flopter.core.constants.EPSILON_0 * T_e)
                       / (flopter.core.constants.ELEM_CHARGE * dens))
    c_s = np.sqrt((flopter.core.constants.ELEM_CHARGE * (T_e + (gamma_i * T_i)))
                  / (flopter.core.constants.PROTON_MASS * mass))
    I_0 = dens * flopter.core.constants.ELEM_CHARGE * c_s * A_coll
    a = ((c_1 + (c_2 / np.tan(alpha))) / np.sqrt(np.sin(alpha))) * (lambda_D / (L + g))
    if print_fl:
        print("a = {}, c_s = {}, lambda_d = {}, I_0 = {}".format(a, c_s, lambda_D, I_0))
    V = (v_f - voltage) / T_e
    I = I_0 * (1 + (a * np.float_power(np.abs(V), .75)) - np.exp(-V))
    return I


def calc_sheath_expansion_coeff(temp, density, L, g, alpha, c_1=0.9, c_2=0.6):
    lambda_D = np.sqrt((flopter.core.constants.EPSILON_0 * temp) / (flopter.core.constants.ELEM_CHARGE * density))
    a = ((c_1 + (c_2 / np.tan(alpha))) / np.sqrt(np.sin(alpha))) * (lambda_D / (L + g))
    return a


def sound_speed(T_e, gamma_i=1, mass=1):
    return np.sqrt((flopter.core.constants.ELEM_CHARGE * (T_e + (gamma_i * T_e))) / (
                flopter.core.constants.PROTON_MASS * mass))


def d_sound_speed(c_s, T_e, d_T_e):
    return np.abs((c_s * d_T_e) / (2 * T_e))


def electron_density(I_sat, c_s, A_coll):
    return I_sat / (flopter.core.constants.ELEM_CHARGE * c_s * A_coll)


def d_electron_density(n_e, c_s, d_c_s, A_coll, d_A_coll, I_sat, d_I_sat):
    return np.abs(n_e) * np.sqrt((d_c_s / c_s)**2 + (d_A_coll / A_coll)**2 + (d_I_sat / I_sat)**2)


class MagnumProbes(object):
    def __init__(self):
        L_small = 3e-3          # m
        a_small = 2e-3          # m
        b_small = 3e-3          # m
        g_small = 2e-3          # m
        d_perp_small = 3e-4     # m
        theta_f_small = np.radians(72)

        L_big = 5e-3            # m
        a_big = 4.5e-3          # m
        b_big = 6e-3            # m
        g_big = 1e-3            # m
        d_perp_big = 3e-4       # m
        theta_f_big = np.radians(73.3)

        L_lang = 5e-3           # m
        a_lang = 2e-3           # m
        b_lang = 3.34e-3        # m
        g_lang = 1e-3           # m
        d_perp_lang = 3e-4      # m
        theta_f_reg = np.radians(75)

        L_round = 4e-3          # m
        g_round = 5e-4          # m
        d_perp_round = 1e-4     # m

        theta_p = np.radians(10)

        self.probe_s = AngledTipProbe(a_small, b_small, L_small, g_small, d_perp_small, theta_f_small, theta_p)
        self.probe_b = AngledTipProbe(a_big, b_big, L_big, g_big, d_perp_big, theta_f_big, theta_p)
        self.probe_l = AngledTipProbe(a_lang, b_lang, L_lang, g_lang, d_perp_lang, theta_f_reg, theta_p)
        self.probe_r = FlushCylindricalProbe(L_round / 2, g_round, d_perp_round)
        self.probe_position = {
            'l': -6,
            's': 4,
            'b': 14,
            'r': 24
        }
        self.position_ind = ['l', 's', 'b', 'r']

    def __getitem__(self, item):
        item = item.lower()
        probes = {
            's': self.probe_s,
            'l': self.probe_l,
            'b': self.probe_b,
            'r': self.probe_r,
        }
        return probes[item]


class MagnumProbesOld(object):
    def __init__(self):
        L_small = 3e-3  # m
        a_small = 2e-3  # m
        b_small = 3e-3  # m
        g_small = 2e-3  # m
        theta_f_small = np.radians(72)

        L_large = 5e-3  # m
        a_large = 4.5e-3  # m
        b_large = 6e-3  # m
        g_large = 1e-3  # m
        theta_f_large = np.radians(73.3)

        L_reg = 5e-3  # m
        a_reg = 2e-3  # m
        b_reg = 3.34e-3  # m
        g_reg = 1e-3  # m
        theta_f_reg = np.radians(75)

        L_cyl = 4e-3  # m
        g_cyl = 5e-4  # m

        d_perp = 3e-4  # m
        theta_p = np.radians(10)

        self.probe_s = AngledTipProbe(a_small, b_small, L_small, g_small, d_perp, theta_f_small, theta_p)
        self.probe_l = AngledTipProbe(a_large, b_large, L_large, g_large, d_perp, theta_f_large, theta_p)
        self.probe_r = AngledTipProbe(a_reg, b_reg, L_reg, g_reg, d_perp, theta_f_reg, theta_p)
        self.probe_c = FlushCylindricalProbe(L_cyl / 2, g_cyl, d_perp)
        self.probes = {
            's': self.probe_s,
            'r': self.probe_r,
            'l': self.probe_l,
            'c': self.probe_c,
        }
        self.position = ['s', 'r', 'l', 'c']

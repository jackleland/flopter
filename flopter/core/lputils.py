import numpy as np

import flopter.core.constants as c
from flopter.core import normalise as nrm
from abc import ABC, abstractmethod


class LangmuirProbe(ABC):
    def __init__(self, g, d_perp):
        self.g = g
        self.d_perp = d_perp

    @abstractmethod
    def is_angled(self):
        pass

    @abstractmethod
    def get_collection_area(self, alpha):
        pass

    @abstractmethod
    def get_analytical_iv(self, voltage, v_f, alpha, temp, dens):
        pass

    @abstractmethod
    def get_2d_probe_length(self):
        pass

    @abstractmethod
    def get_2d_probe_height(self):
        pass

    @abstractmethod
    def get_3d_probe_depth(self):
        pass

    @abstractmethod
    def get_2d_collection_length(self, alpha):
        pass

    @abstractmethod
    def calc_exposed_lengths(self, alpha):
        pass

    @abstractmethod
    def get_sheath_exp_param(self, temp, dens, alpha, c_1=0.4, c_2=0.5):
        pass

    def get_density(self, sat_current, temperature, alpha, gamma_i=1, mass=1, Z=1):
        c_s = sound_speed(temperature, gamma_i=gamma_i, mass=mass, Z=Z)
        A_coll = self.get_collection_area(alpha)
        return electron_density(sat_current, c_s, A_coll, Z=Z)

    def get_d_density(self, sat_current, d_sat_current, temperature, d_temperature, alpha, gamma_i=1, mass=1, Z=1):
        c_s = sound_speed(temperature, gamma_i=gamma_i, mass=mass, Z=Z)
        A_coll = self.get_collection_area(alpha)
        n_e = electron_density(sat_current, c_s, A_coll, Z=Z)

        d_c_s = d_sound_speed(c_s, temperature, d_temperature)
        d_A_coll = np.abs(self.get_collection_area(alpha + np.radians(0.8)) - A_coll)

        return d_electron_density(n_e, c_s, d_c_s, A_coll, d_A_coll, sat_current, d_sat_current)

    def get_isat(self, temperature, density, alpha, gamma_i=1, mass=1):
        c_s = sound_speed(temperature, gamma_i=gamma_i, mass=mass)
        A_coll = self.get_collection_area(alpha)
        return density * c.ELEM_CHARGE * c_s * A_coll


class AngledTipProbe(LangmuirProbe):
    def __init__(self, a, b, L, g, d_perp, theta_f, theta_p):
        super().__init__(g, d_perp)
        self.a = a
        self.b = b
        self.L = L
        self.theta_f = theta_f
        self.theta_p = theta_p

    def get_collection_area(self, alpha):
        return calc_probe_collection_area(self.a, self.b, self.L, self.g, self.d_perp, alpha, self.theta_p)

    def get_2d_collection_length(self, alpha):
        d, h_coll = self.calc_exposed_lengths(alpha)
        L_tip = self.L / np.cos(self.theta_p)
        L_coll = ((L_tip - d) * np.sin(alpha + self.theta_p)) + (h_coll * np.cos(alpha))
        return L_coll

    def is_angled(self):
        return self.theta_p > 0

    def get_analytical_iv(self, voltage, v_f, alpha, temp, dens, mass=1, gamma_i=1.0, c_1=0.9, c_2=0.6, print_fl=False):
        return analytical_iv_curve(voltage, v_f, temp, dens, alpha, self.get_collection_area(alpha), c_1=c_1, c_2=c_2,
                                   gamma_i=gamma_i, mass=mass, L=self.L, g=self.g, print_fl=print_fl)

    def get_2d_probe_length(self):
        return self.L

    def get_2d_probe_height(self):
        return self.L * np.tan(self.theta_p)

    def get_3d_probe_depth(self):
        return max(self.b, self.a)

    def calc_exposed_lengths(self, alpha):
        return calc_probe_exposed_lengths(self.g, self.d_perp, alpha, self.theta_p)

    def get_sheath_exp_param(self, temp, dens, alpha, c_1=0.4, c_2=0.5, form='bergmann'):
        if form == 'bergmann':
            return calc_sheath_expansion_param(temp, dens, self.L, self.g, alpha, c_1=c_1, c_2=c_2)
        elif form == 'leland':
            return calc_new_sheath_expansion_param(temp, dens, self.L, self.g, alpha, self.d_perp, self.theta_p,
                                                   c_1=c_1, c_2=c_2)
        elif form == 'rotated':
            return calc_2d_box_sheath_expansion_param(temp, dens, self.L, self.g, alpha, self.d_perp, self.theta_p,
                                                      c_1=c_1, c_2=c_2)


class FlushCylindricalProbe(LangmuirProbe):
    def __init__(self, radius, g, d_perp):
        super().__init__(g, d_perp)
        self.radius = radius
        self.theta_p = 0.0

    def is_angled(self):
        return False

    def get_collection_area(self, alpha):
        d, h_coll = self.calc_exposed_lengths(alpha)
        theta_c = max(2 * np.arccos((self.radius - d) / self.radius), 0)
        l_arc_eff = (1 - np.cos((np.pi / 2) - theta_c)) * self.radius
        h_r = (self.radius - d) * np.sin(alpha)
        A_coll = (
            ((np.sin(alpha) * self.radius**2) * (np.pi - theta_c + (2 * np.sin(theta_c))))
            + (2 * h_coll * np.cos(alpha) * l_arc_eff)
            + (l_arc_eff * h_r)
        )
        return A_coll

    def get_2d_collection_length(self, alpha):
        d, h_coll = self.calc_exposed_lengths(alpha)
        L_coll = ((self.get_2d_probe_length() - d) * np.sin(alpha)) + (h_coll * np.cos(alpha))
        return L_coll

    def get_analytical_iv(self, voltage, v_f, alpha, temp, dens, mass=1, gamma_i=1.0, c_1=0.9, c_2=0.6, print_fl=False):
        analytical_iv_curve(voltage, v_f, temp, dens, alpha, self.get_collection_area(alpha), c_1=c_1, c_2=c_2,
                            gamma_i=gamma_i, mass=mass, L=(2 * self.radius), g=self.g, print_fl=print_fl)

    def get_2d_probe_length(self):
        return 2 * self.radius

    def get_2d_probe_height(self):
        return 0

    def get_3d_probe_depth(self):
        return 2 * self.radius

    def calc_exposed_lengths(self, alpha):
        return calc_probe_exposed_lengths(self.g, self.d_perp, alpha, 0.0)

    def get_sheath_exp_param(self, temp, dens, alpha, c_1=0.4, c_2=0.5):
        return calc_sheath_expansion_param(temp, dens, self.get_2d_probe_length(), self.g, alpha, c_1=c_1, c_2=c_2)


def calc_probe_collection_area(a, b, L, g, d_perp, theta_perp, theta_p, print_fl=False):
    # d = max(0, ((d_perp - (g * np.tan(theta_perp)))
    #         / (np.sin(theta_p) + (np.tan(theta_perp) * np.cos(theta_p)))))
    # h_coll = max(0, (g * np.tan(theta_perp) - d_perp) * np.cos(theta_perp))
    d, h_coll = calc_probe_exposed_lengths(g, d_perp, theta_perp, theta_p)
    if print_fl:
        print("d = {}, h_coll = {}".format(d, h_coll))
    L_exp = (L / np.cos(theta_p)) - d
    return ((a + (0.5 * (b - a) * (L_exp / L))) * L_exp * np.sin(theta_perp + theta_p)) + (h_coll * b)


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
    lambda_D = debye_length(T_e, dens)
    c_s = np.sqrt((c.ELEM_CHARGE * (T_e + (gamma_i * T_i)))
                  / (c.PROTON_MASS * mass))
    I_0 = dens * c.ELEM_CHARGE * c_s * A_coll
    a = ((c_1 + (c_2 / np.tan(alpha))) / np.sqrt(np.sin(alpha))) * (lambda_D / (L + g))
    if print_fl:
        print("a = {}, c_s = {}, lambda_d = {}, I_0 = {}".format(a, c_s, lambda_D, I_0))
    V = (v_f - voltage) / T_e
    I = I_0 * (1 + (a * np.float_power(np.abs(V), .75)) - np.exp(-V))
    return I


def debye_length(temp, density):
    return np.sqrt((c.EPSILON_0 * temp) / (c.ELEM_CHARGE * density))


def thermal_velocity(T_e, mass=1):
    return np.sqrt(c.ELEM_CHARGE * T_e / (c.PROTON_MASS * mass))


def sound_speed(T_e, T_i=None, gamma_i=1, mass=1, Z=1):
    if T_i is None:
        T_i = T_e
    return np.sqrt((Z * c.ELEM_CHARGE * (T_e + (gamma_i * T_i))) / (c.PROTON_MASS * mass))


def d_sound_speed(c_s, T_e, d_T_e):
    return np.abs((c_s * d_T_e) / (2 * T_e))


def electron_density(I_sat, c_s, A_coll, k=0.5, Z=1.0):
    return I_sat / (k * Z * c.ELEM_CHARGE * c_s * A_coll)


def d_electron_density(n_e, c_s, d_c_s, A_coll, d_A_coll, I_sat, d_I_sat):
    return np.abs(n_e) * np.sqrt((d_c_s / c_s)**2 + (d_A_coll / A_coll)**2 + (d_I_sat / I_sat)**2)


def ion_larmor_radius(T_e, B, mu=1, Z=1):
    v_therm = thermal_velocity(T_e, mass=mu)
    omega = ion_gyrofrequency(B, mu=mu, Z=Z)
    return v_therm / omega


def ion_gyrofrequency(B, mu=1, Z=1):
    return gyrofrequency(B, mu * c.PROTON_MASS, Z * c.ELEM_CHARGE)


def electron_gyrofrequency(B):
    return gyrofrequency(B, c.ELECTRON_MASS, c.ELEM_CHARGE)


def gyrofrequency(B, mass, charge):
    return (np.abs(charge) * B) / mass


def estimate_temperature(float_pot, plasma_pot, m_e=1.0, m_i=c.P_E_MASS_RATIO):
    """
    Estimates temperature using the differece between the floating and plasma
    potentials using standard equation (not OML).
    :param float_pot:   Floating potential (in V)
    :param plasma_pot:  Plasma potential (in V)
    :param m_e:         electron mass (in kg)
    :param m_i:         ion mass (in kg)
    :return:            Estimate of temperature (in eV)
    """
    return (float_pot - plasma_pot) / (np.log(0.6 * np.sqrt(2 * np.pi * m_e / m_i)))


def calc_sheath_expansion_param(temp, density, L, g, alpha, c_1=0.9, c_2=0.6):
    lambda_D = debye_length(temp, density)
    a = ((c_1 + (c_2 / np.tan(alpha))) / np.sqrt(np.sin(alpha))) * (lambda_D / (L + g))
    return a


def calc_new_sheath_expansion_param(temp, density, L, g, alpha, d_perp, theta_p, c_1=0.4, c_2=0.5):
    lambda_D = debye_length(temp, density)
    a = ((((c_1 * (np.tan(alpha) + (np.tan(theta_p)))) + c_2) * lambda_D)
         / ((((L + g) * np.tan(alpha)) + (L * np.tan(theta_p)) - d_perp) * np.sqrt(np.sin(alpha))))
    return a


def calc_2d_box_sheath_expansion_param(temp, density, L, g, theta, d_perp, theta_p, c_1=0.4, c_2=0.5, delta_0=0.0):
    lambda_D = debye_length(temp, density)
    L_eff = (L/np.cos(theta_p)) - ((d_perp + delta_0 - (g * np.tan(theta)))
                                   / ((np.cos(theta_p) * np.tan(theta)) + np.sin(theta_p)))
    theta_tot = theta_p + theta

    a = ((c_1 + c_2 * (1 / np.tan(theta_tot))) * lambda_D) / (np.sqrt(np.sin(theta_tot)) * (L_eff + (delta_0 / np.tan(theta_tot))))
    return a


def decompose_sheath_exp_param(a, theta, L, g, d_perp=0, theta_p=0):
    y = a * (L + g) * np.sqrt(np.sin(theta))
    x = np.cos(theta) / np.sin(theta)
    return x, y


def decompose_new_sheath_exp_param(a, theta, L, g, d_perp, theta_p):
    y = (a * np.sqrt(np.sin(theta)) * (((L + g) * np.tan(theta))
                                       + (L * np.tan(theta_p)) - d_perp))
    x = np.tan(theta) + np.tan(theta_p)

    return x, y


def decompose_alt_new_sheath_exp_param(a, theta, L, g, d_perp, theta_p):
    theta_tot = theta + theta_p
    y = (a * np.sqrt(np.sin(theta))
         * (L + ((np.cos(theta_p) / np.sin(theta_tot)) * ((g * np.sin(theta)) - (d_perp * np.cos(theta))))))
    x = (np.cos(theta_p) * np.cos(theta)) / np.sin(theta_tot)
    return x, y


def decompose_2d_box_sheath_exp_param(a, theta, L, g, d_perp, theta_p, delta_0=0.0):
    L_eff = (L / np.cos(theta_p)) - ((d_perp + delta_0 - (g * np.tan(theta)))
                                     / ((np.cos(theta_p) * np.tan(theta)) + np.sin(theta_p)))

    y = a * np.sqrt(np.sin(theta + theta_p)) * (L_eff + (delta_0 / np.tan(theta + theta_p)))
    x = np.cos(theta + theta_p) / np.sin(theta + theta_p)
    return x, y


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
        g_round = 1.5e-3        # m
        d_perp_round = 1e-4     # m

        theta_p = np.radians(10)

        self.probe_s = AngledTipProbe(a_small, b_small, L_small, g_small, d_perp_small, theta_f_small, theta_p)
        self.probe_b = AngledTipProbe(a_big, b_big, L_big, g_big, d_perp_big, theta_f_big, theta_p)
        self.probe_l = AngledTipProbe(a_lang, b_lang, L_lang, g_lang, d_perp_lang, theta_f_reg, theta_p)
        self.probe_r = FlushCylindricalProbe(L_round / 2, g_round, d_perp_round)
        self.probe_position = {
            'l': 6,
            's': -4,
            'b': -14,
            'r': -24
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

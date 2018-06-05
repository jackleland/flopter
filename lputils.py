import numpy as np
import normalisation as nrm


class FlushMountedProbe(object):
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

    def get_analytical_iv(self, voltage, v_f, alpha, temp, dens, mass=1, gamma_i=2.5, c_1=0.9, c_2=0.6):
        return analytical_iv_curve(voltage, v_f, temp, dens, alpha, self.get_collection_area(alpha), c_1=c_1, c_2=c_2,
                                   gamma_i=gamma_i, mass=mass, L=self.L, g=self.g)


def calc_probe_collection_area(a, b, L, g, d_perp, theta_perp, theta_p, theta_f, print_fl=True):
    d = max(0, ((d_perp - (g * np.tan(theta_perp)))
            / (np.sin(theta_p) + (np.tan(theta_perp) * np.cos(theta_p)))))
    h_coll = max(0, (g * np.tan(theta_perp) - d_perp) * np.cos(theta_perp))
    if print_fl:
        print("d = {}, h_coll = {}".format(d, h_coll))
    L_exp = (L / np.cos(theta_p)) - d
    return (0.5 * (a + b - (d / np.tan(theta_f))) * L_exp * np.sin(theta_perp + theta_p)) + (h_coll * b)


def calc_probe_exposed_lengths(g, d_perp, theta_perp, theta_p):
    d = max(0, ((d_perp - (g * np.tan(theta_perp)))
                / (np.sin(theta_p) + (np.tan(theta_perp) * np.cos(theta_p)))))
    h_coll = max(0, (g * np.tan(theta_perp) - d_perp) * np.cos(theta_perp))
    return d, h_coll


def calc_probe_collection_A_alt(a, b, L, theta_perp, theta_p):
    return (L / np.cos(theta_p)) * (a + b) * 0.5 * np.sin(theta_p + theta_perp)


def analytical_iv_curve(voltage, v_f, temp, dens, alpha, A_coll, c_1=0.9, c_2=0.6, gamma_i=2.5, mass=1, L=1, g=0.5,
                        print_fl=False):
    T_i = temp
    T_e = temp
    lambda_D = np.sqrt((nrm.EPSILON_0 * T_e) / (nrm.ELEM_CHARGE * dens))
    c_s = np.sqrt((nrm.ELEM_CHARGE * (T_e + gamma_i * T_i)) / (nrm.PROTON_MASS * mass))
    I_0 = dens * nrm.ELEM_CHARGE * c_s * A_coll
    a = ((c_1 + (c_2 / np.tan(alpha))) / np.sqrt(np.sin(alpha))) * (lambda_D / (L + g))
    if print_fl:
        print("a = {}, c_s = {}, lambda_d = {}, I_0 = {}".format(a, c_s, lambda_D, I_0))
    V = (v_f - voltage) / T_e
    I = I_0 * (1 + (a * np.float_power(np.abs(V), .75)) - np.exp(-V))
    return I

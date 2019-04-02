import numpy as np
import matplotlib.pyplot as plt
from flopter.core import normalisation as nrm, fitters as f

# Estimated parameters
# v_f = 8        # V
a_est = 0.0208  # V

# Known parameters
T_e = 5        # eV
T_i = T_e       # eV
v_f = 3 * T_e   # V - Approximation from Stangeby
n_e = 1e18      # m^-3

# Flush probe measurements
L = 5e-3            # m
g = 1e-3            # m
d = 2e-3            # m
d_perp = 3e-4       # m
A_probe = 1.335e-5  # m^2
a_coll = []
a_probe = []
alphas = np.arange(0.5, 11, 0.5)

plot_fl = False

# Constants
m_i = nrm.PROTON_MASS
m_e = nrm.ELECTRON_MASS
deg_freedom = 3
gamma_i = (deg_freedom + 2) / 2
c_1 = 0.9
c_2 = 0.6

# for v_max in [100, 150, 300]:
for v_max in [100]:
    V_range = np.arange(-2*v_max, 2*(v_f + 10), 1) / 2
    V = (v_f - V_range) / T_e

    num_samples = 20

    currents = {}
    noise = {}
    isats_def = []
    temps = []
    temps_stderr = []
    temps_err = []
    isats = []
    isats_err = []
    isats_stderr = []
    isats_raw = []
    failed_fits = []
    for alpha in alphas:
        theta_p = np.radians(10)
        theta_perp = np.radians(alpha)
        A_coll = (d * np.sin(np.abs(theta_perp) + np.abs(theta_p)) *
                  ((L / np.cos(theta_p)) - ((d_perp - (g * np.tan(theta_perp)))
                                            / (np.sin(theta_p) + (np.tan(theta_perp) * np.cos(theta_p))))))
        lambda_D = np.sqrt((nrm.EPSILON_0 * T_e) / (nrm.ELEM_CHARGE * n_e))
        c_s = np.sqrt((nrm.ELEM_CHARGE * (T_e + gamma_i * T_i)) / m_i)
        a = ((c_1 + (c_2 / np.tan(np.radians(alpha)))) / np.sqrt(np.sin(np.radians(alpha)))) * (lambda_D / (L + g))
        I_0 = n_e * nrm.ELEM_CHARGE * c_s * A_coll
        J_0 = I_0 / A_coll
        q_par = 8 * T_e * J_0
        print('Heat flux: {}'.format(q_par))
        # I_0 = n_e * nrm.ELEM_CHARGE * c_s * np.sin(np.radians(alpha)) * A_probe
        isats_def.append(I_0)

        I = I_0 * (1 + (a * np.float_power(np.abs(V), .75)) - np.exp(-V))
        currents[alpha] = I

        fitter = f.FullIVFitter()
        temps_t = []
        temps_err_t = []
        isats_t = []
        isats_err_t = []
        count = 0
        for i in range(num_samples):
            noisey = (0.09 * (np.random.rand(len(I)) - 0.5)) + I
            try:
                fit_data = fitter.fit(V_range, noisey)
            except RuntimeError:
                count += 1
                continue
            if i == 0 and alpha == 10.5 and v_max == 100 and plot_fl:
                # plt.figure()
                # plt.plot(V_range, I, label='Analytical')
                # plt.axvline(0, color='gray', linewidth=1, linestyle='dashed')
                # plt.xlabel('V (V)')
                # plt.ylabel('Current (A)')
                # plt.legend()

                # plt.figure()
                # plt.plot(V_range, noisey, label='Noisified')
                # plt.xlabel('V (V)')
                # plt.ylabel('Current (A)')
                # plt.legend()

                plt.figure()
                plt.plot(V_range, noisey, label='Noisified')
                plt.plot(V_range, fit_data.fit_y, label='Pseudo-measurement')
                plt.axvline(0, color='gray', linewidth=1, linestyle='dashed')
                plt.xlabel('V (V)')
                plt.ylabel('Current (A)')
                plt.legend()

            temps_t.append(fit_data.fit_params[fitter.get_temp_index()].value)
            temps_err_t.append(fit_data.fit_params[fitter.get_temp_index()].error)
            isats_t.append(fit_data.fit_params[fitter.get_isat_index()].value)
            isats_err_t.append(fit_data.fit_params[fitter.get_isat_index()].error)

        temps.append(np.array(temps_t).mean())
        temps_err.append(np.array(temps_err_t).mean())
        temps_stderr.append(np.array(temps_t).std() / np.sqrt(num_samples - count))
        isats.append(np.array(isats_t).mean() - I_0)
        isats_err.append(np.array(isats_err_t).mean())
        isats_stderr.append(np.array(isats_t).std() / np.sqrt(num_samples - count))
        isats_raw.append(np.array(isats_t).mean())
        failed_fits.append(count)

    plt.figure(10)
    plt.errorbar(alphas, temps, temps_stderr, label=r'$T_e - V_{}={}$'.format('{max}', v_max))
    # plt.hist(failed_fits, alphas, color='gray', alpha=0.5, label='Unfittable - V_{}={}'.format('{max}', v_max))

    plt.figure(11)
    plt.errorbar(alphas, isats_raw, isats_stderr, label=r'$I_{} - V_{}={}$'.format('{sat}', '{max}', v_max))
    if v_max == 100:
        plt.plot(alphas, isats_def, label=r'Defined $I_{sat}$', color='gray', linewidth=1, linestyle='dashed')
    # plt.hist(failed_fits, alphas, color='gray', alpha=0.5, label='Unfittable - V_{}={}'.format('{max}', v_max))

plt.figure(10)
plt.axhline(T_e, color='gray', linewidth=1, linestyle='dashed', label=r'Defined $T_e$')
plt.legend()
plt.xlabel(r'$\alpha$ ($^{\circ}$)')
plt.ylabel(r'$T_e$ (eV)')

plt.figure(11)
# plt.axhline(0, color='gray', linewidth=1, linestyle='dashed', label=r'Defined $I_{sat}$')
plt.legend()
plt.xlabel(r'$\alpha$ ($^{\circ}$)')
plt.ylabel(r'$I_sat$ (A)')
# plt.ylabel(r'Difference from defined $I_sat$ (A)')

plt.show()

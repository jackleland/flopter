import numpy as np
import matplotlib.pyplot as plt
import normalisation as nrm
import fitters as f
import pandas as pd

# Estimated parameters
v_f = 8        # V
a_est = 0.0208  # V

# Known parameters
T_i = 6       # eV
T_e = 6       # eV
n_e = 1e19  # m^-3

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
# for alpha in alphas:
#     theta_p = np.radians(10)
#     theta_perp = np.radians(alpha)
#     A_coll = (d * np.sin(np.abs(theta_perp) + np.abs(theta_p)) *
#               ((L / np.cos(theta_p)) - ((d_perp - (g * np.tan(theta_perp)))
#                                         / (np.sin(theta_p) + (np.tan(theta_perp) * np.cos(theta_p))))))
#     a_probe.append(A_probe * np.sin(theta_perp))
#     a_coll.append(A_coll)
# plt.figure(0)
# plt.plot(alphas, a_coll, label='A_coll')
# plt.plot(alphas, a_probe, label='A_probe')
# plt.legend()
# plt.show()
# exit()

# Constants
m_i = nrm._PROTON_MASS
m_e = nrm._ELECTRON_MASS
deg_freedom = 3
gamma_i = (deg_freedom + 2) / 2
c_1 = 0.9
c_2 = 0.6
# alpha_high = np.radians(10)
# alpha_low = np.radians(0.5)
# alphas = [alpha_low, alpha_high]

for v_max in [100, 150]:
    V_range = np.arange(-2*v_max, 20, 1) / 2
    V = (v_f - V_range) / T_e

    num_samples = 10000

    currents = {}
    noise = {}
    isats_def = []
    temps = []
    temps_stderr = []
    temps_err = []
    isats = []
    isats_err = []
    isats_stderr = []
    failed_fits = []
    for alpha in alphas:
        theta_p = np.radians(10)
        theta_perp = np.radians(alpha)
        A_coll = (d * np.sin(np.abs(theta_perp) + np.abs(theta_p)) *
                  ((L / np.cos(theta_p)) - ((d_perp - (g * np.tan(theta_perp)))
                                            / (np.sin(theta_p) + (np.tan(theta_perp) * np.cos(theta_p))))))
        lambda_D = np.sqrt((nrm._EPSILON_0 * T_e) / (nrm._ELEM_CHARGE * n_e))
        c_s = np.sqrt((nrm._ELEM_CHARGE * (T_e + gamma_i * T_i)) / m_i)
        a = ((c_1 + (c_2 / np.tan(np.radians(alpha)))) / np.sqrt(np.sin(np.radians(alpha)))) * (lambda_D / (L + g))
        I_0 = n_e * nrm._ELEM_CHARGE * c_s * A_coll
        # I_0 = n_e * nrm._ELEM_CHARGE * c_s * np.sin(np.radians(alpha)) * A_probe
        isats_def.append(I_0)

        I = I_0 * (1 + (a * np.float_power(np.abs(V), .75)) - np.exp(-V))
        currents[alpha] = I

        fitter = f.FullIVFitter(floating_potential=v_f)
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
            if i == 0 and alpha == 10.5 and plot_fl:
                plt.figure()
                plt.plot(V_range, I, label='Analytical')
                plt.xlabel('V (V)')
                plt.ylabel('Current (A)')
                plt.legend()

                plt.figure()
                plt.plot(V_range, noisey, label='Noisified')
                plt.xlabel('V (V)')
                plt.ylabel('Current (A)')
                plt.legend()

                plt.figure()
                plt.plot(V_range, noisey, label='Noisified')
                plt.plot(V_range, fit_data.fit_y, label='Pseudo-measurement')
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
        failed_fits.append(count)

    plt.figure(1)
    plt.errorbar(alphas, temps, temps_stderr, label=r'$T_e - V_{}={}$'.format('{max}', v_max))
    # plt.hist(failed_fits, alphas, color='gray', alpha=0.5, label='Unfittable - V_{}={}'.format('{max}', v_max))

    plt.figure(2)
    plt.errorbar(alphas, isats, isats_stderr, label=r'$I_{} - V_{}={}$'.format('{sat}', '{max}', v_max))
    # if v_max == 100:
    #     plt.plot(alphas, isats_def, label=r'Defined $I_{sat}$', color='gray', linewidth=1, linestyle='dashed')
    # plt.hist(failed_fits, alphas, color='gray', alpha=0.5, label='Unfittable - V_{}={}'.format('{max}', v_max))

plt.figure(1)
plt.axhline(T_e, color='gray', linewidth=1, linestyle='dashed', label=r'Defined $T_e$')
plt.legend()
plt.xlabel(r'$\alpha$ ($^{\circ}$)')
plt.ylabel(r'$T_e$ (eV)')

plt.figure(2)
plt.axhline(0, color='gray', linewidth=1, linestyle='dashed', label=r'Defined $I_{sat}$')
plt.legend()
plt.xlabel(r'$\alpha$ ($^{\circ}$)')
plt.ylabel(r'Difference from defined $I_sat$ (A)')

plt.show()

from scipy.io import loadmat
from scipy import interpolate
from scipy.optimize import curve_fit
from normalisation import Denormaliser, TIME, LENGTH, POTENTIAL, CURRENT
import matplotlib.pyplot as plt
import numpy as np


def iv_characteristic_function(v, *parameters):
    I_0 = parameters[0]
    a = parameters[1]
    v_f = parameters[2]
    T_e = parameters[3]
    V = (v_f - v)/T_e
    return I_0 * (1 - np.exp(-V) + (a * np.float_power(np.absolute(V), 0.75)))


def simple_iv_characteristic_function(v, *parameters):
    I_0 = parameters[0]
    v_f = parameters[1]
    T_e = parameters[2]
    V = (v_f - v) / T_e
    return I_0 * (1 - np.exp(-V))


def ion_current_sl_function(v, *parameters):
    I_0 = parameters[0]
    a = parameters[1]
    return I_0 + (a*v)


def print_params(values, errors, labels=("I_0", "a", "v_float", "T_e")):
    print("FIT PARAMETERS")
    for i in range(len(values)):
        print("{a} = {b} +/- {c}".format(a=labels[i], b=values[i], c=errors[i]))


##################################
#             Extract            #
##################################

spice_dir = '/home/jleland/Spice/spice2/'
data_mount_dir = 'bin/data/'
run_name = 'fullgridtest'
group_name = 'tests/'
data_dir = spice_dir + data_mount_dir + group_name + run_name + '/'
input_dir = spice_dir + 'bin/inputs/'
script_dir = spice_dir + 'bin/scripts/'
tfile_pre = 't-{a}'.format(a=run_name)
tfile_suf = '.mat'
tfile = loadmat(data_dir + tfile_pre + tfile_suf)
input_filename = data_dir + 'jleland.2.inp'
denormaliser = Denormaliser(input_filename)

# print( tfile.keys() )

# Extract relevant arrays from the matlab file
dt = np.squeeze(tfile['dt']).tolist()
time = denormaliser(np.squeeze(tfile['t'])[:-1], TIME)
objects = np.squeeze(tfile['objects'])
# probe_current_e = denormaliser(np.squeeze(tfile['objectscurrente'])[2], CURRENT, additional_arg=dt)
# probe_current_i = denormaliser(np.squeeze(tfile['objectscurrenti'])[2], CURRENT, additional_arg=dt)
#probe_bias = denormaliser(np.squeeze(tfile['ProbePot']), POTENTIAL)
#qn_potential = denormaliser(np.squeeze(tfile['QnPot']), POTENTIAL)
probe_current_e = np.squeeze(tfile['objectscurrente'])[2]
probe_current_i = np.squeeze(tfile['objectscurrenti'])[2]
probe_bias = np.squeeze(tfile['ProbePot'])
qn_potential = np.squeeze(tfile['QnPot'])
probe_current_tot = probe_current_i + probe_current_e
density = np.squeeze(tfile['rho'])


##################################
#             Prepare            #
##################################

# add on zeroes missing from time when diagnostics were not running and then
# remove 1/256 of data to get an array of size len(probe_current)
leading_zeroes = np.zeros(len(probe_bias), dtype=np.int)
probe_bias_double = np.concatenate([leading_zeroes, probe_bias])[0:-256:256]

t_shape = np.shape(time)
pct_shape = np.shape(probe_current_tot)
pb_shape = np.shape(probe_bias)

t_max = np.max(time)
t_min = np.min(time)
time_alt = np.linspace(t_min, t_max, pb_shape[0])

# Extract the voltage and current for the sweeping region.
V_full = np.trim_zeros(probe_bias_double, 'f')
full_length = len(V_full)
I_i_full = probe_current_i[-full_length:]
I_e_full = probe_current_e[-full_length:]
I_full = probe_current_tot[-full_length:]

##################################
#            Trimming            #
##################################

# Cutting region definition
trim_beg = 0.05
trim_end = 0.7

# Cut off the noise in the electron saturation region
V = V_full[int(full_length*trim_beg):int(full_length*trim_end)]
I = I_full[int(full_length*trim_beg):int(full_length*trim_end)]
I_e = I_e_full[int(full_length*trim_beg):int(full_length*trim_end)]
I_i = I_i_full[int(full_length*trim_beg):int(full_length*trim_end)]

##################################
#             Fitting            #
##################################


# find the potential where the net current is zero using np.interp
# v_float = np.interp(0.0, I, V)
iv_interp = interpolate.interp1d(I, V)
v_float = iv_interp(0.0)

print('v_float = {a}'.format(a=v_float))

# Parameters for the full fitting function [I_0, a, v_float, electron_temperature]
# These are taken from Sam's paper
e_temp = 6  # eV
I_0_sam = 32.774
a_sam = 0.0204
params = [I_0_sam, a_sam, v_float, e_temp]
bounds = ([-np.inf, 0, v_float - 0.01, -np.inf],
          [np.inf, np.inf, v_float, np.inf])

# Parameters for the simple fitting function (no sheath expansion)
params_simple = [I_0_sam, v_float, e_temp]
bounds_simple = ([-np.inf, v_float - 0.01, -np.inf],
                [np.inf, v_float, np.inf])
I_sam = iv_characteristic_function(V, *params)

# Run fitting algorithm and create fitted function array
fparams, fcov = curve_fit(iv_characteristic_function, V, I, p0=params, bounds=bounds)
fparams_simple, fcov_simple = curve_fit(simple_iv_characteristic_function, V, I, p0=params_simple, bounds=bounds_simple)
I_fitted = iv_characteristic_function(V_full, *fparams)
I_fitted_simple = simple_iv_characteristic_function(V, *fparams_simple)
fstdevs = np.sqrt(np.diag(fcov))
fstdevs_simple = np.sqrt(np.diag(fcov_simple))

##################################################
#         Straight Line Fitting Function         #
##################################################

sl_V = np.power(np.abs(V), 0.75)
sl_params = [I_0_sam, a_sam]
sl_bounds = ([-np.inf, 0],
             [np.inf, np.inf])

sl_fit_params, sl_fit_cov = curve_fit(ion_current_sl_function, sl_V, I_i, p0=sl_params, bounds=sl_bounds)
sl_I_fitted = ion_current_sl_function(sl_V, *sl_fit_params)
sl_fstdevs = np.sqrt(np.diag(sl_fit_cov))

##################################
#             Print              #
##################################

# print fit parameters to console with errors
print_params(fparams, fstdevs)
print_params(fparams_simple, fstdevs_simple, labels=["I_0", "v_float", "T_e"])
print_params(sl_fit_params, sl_fstdevs, labels=["I_0", "a"])

##################################
#              Plot              #
##################################

fig = plt.figure()
plt.plot(V_full, I_i_full, label='Untrimmed', linestyle='dashed')
plt.plot(V, I_i, label='Ion')
# plt.plot(V, I_e, label='Electron')
# plt.plot(V, I, label='Fitted section')
plt.plot([v_float], [0.0], 'x', label=r'V$_{float}$')
# plt.plot(V_full, I_fitted, label='Fit')
# plt.plot(V, I_fitted_simple, label='Simple')
# plt.plot(V, I_sam, label='Sam\'s Params')
plt.xlabel('V')
plt.ylabel('I')
plt.axhline(y=0, color='gray', linewidth=1, linestyle='dashed')
plt.axvline(x=v_float, color='gray', linewidth=1, linestyle='dashed')
plt.legend()
# plt.plot(I)
# plt.plot(V)
plt.show()

fig1 = plt.figure()
plt.plot(sl_V, I_i, 'x')
plt.plot(sl_V, sl_I_fitted, label='Fitted')
plt.xlabel(r'$|V|^{3/4}$')
plt.ylabel(r'$I_i$')
plt.legend()
plt.show()

# plt.plot(time, probe_current_tot)
# plt.plot(time, probe_bias_double)
# plt.plot(time_alt, probe_bias)
# plt.show()

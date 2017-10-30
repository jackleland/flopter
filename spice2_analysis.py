from scipy.io import loadmat
from scipy import interpolate
from scipy.optimize import curve_fit
from normalisation import Denormaliser, TIME, LENGTH, POTENTIAL, CURRENT
from homogeniser import Spice2Homogeniser
from inputparser import InputParser
import matplotlib.pyplot as plt
import numpy as np
import os


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
    return I_0 *(1 + (a*v))


def print_params(values, errors, labels=("I_0", "a", "v_float", "T_e")):
    print("FIT PARAMETERS")
    for i in range(len(values)):
        print("{a} = {b} +/- {c}".format(a=labels[i], b=values[i], c=errors[i]))


def get_runnames(data_dir):
    run_name = data_dir.split(os.sep)[-2]
    print(run_name)
    return 'noga', 'nogaph'


##################################
#             Extract            #
##################################

# Constants(ish)
spice_dir = '/home/jleland/Spice/spice2/'
input_dir = spice_dir + 'bin/inputs/'
script_dir = spice_dir + 'bin/scripts/'

# # Run specific data
# data_mount_dir = 'bin/data/'
# group_name = 'benchmarking_sam/'
# folder_name = 'gapless_fullgrid/'
# data_dir = spice_dir + data_mount_dir + group_name + folder_name

# ------------------- Run specific data ------------------- #

# Cumulus
data_mount_dir = 'bin/data/'
# group_name = 'tests/'
# folder_name = 'fullgridtest/'
# folder_name = 'reversed_charge/'

group_name = 'benchmarking_sam/'
folder_name = 'gapless_fullgrid/'
# folder_name = 'gapless_halfgrid1/'
# folder_name = 'gapless_halfgrid2/'
# folder_name = 'nogaphalfgrid_tlong1/'

# Freia
# data_mount_dir = 'bin/data_f/'
# group_name = 'rundata'
# folder/name '6-s-halfgrid'


data_dir = spice_dir + data_mount_dir + group_name + folder_name

# input_filename = input_dir + 'jleland.3.inp'
# input_filename = input_dir + 'jleland.2.inp'
input_filename = data_dir + 's_benchmarking_nogap.inp'
# input_filename = data_dir + 'reversede_ng_hg_sbm.inp'

run_name = 'gapless_'
ext_run_name = 'gapless_fu'
file_suf = '.mat'
tfile_pre = 't-{a}'.format(a=run_name)
tfile_path = data_dir + tfile_pre + file_suf
afile_path = data_dir + ext_run_name + file_suf
tfile = loadmat(tfile_path)
afile = loadmat(afile_path)

# create flopter objects
parser = InputParser(input_filename=input_filename)
denormaliser = Denormaliser(input_parser=parser)
homogeniser = Spice2Homogeniser(denormaliser=denormaliser, data=tfile)

iv_data, raw_data = homogeniser.homogenise()

##################################
#            Trimming            #
##################################

# Cutting region definition
trim_beg = 0.05
trim_end = 0.35

# Cut off the noise in the electron saturation region
full_length = len(iv_data['V'])
V = iv_data['V'][int(full_length*trim_beg):int(full_length*trim_end)]
I = iv_data['I'][int(full_length*trim_beg):int(full_length*trim_end)]
I_e = iv_data['I_e'][int(full_length*trim_beg):int(full_length*trim_end)]
I_i = iv_data['I_i'][int(full_length*trim_beg):int(full_length*trim_end)]

##################################
#             Fitting            #
##################################


# find the potential where the net current is zero using np.interp
# v_float = np.interp(0.0, I, V)
iv_interp = interpolate.interp1d(iv_data['I'], iv_data['V'])
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
I_fitted = iv_characteristic_function(iv_data['V'], *fparams)
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
plt.plot(iv_data['V'], iv_data['I'], label='Untrimmed', linestyle='dashed')
# plt.plot(V, I_i, label='Ion')
# plt.plot(V, I_e, label='Electron')
plt.plot(V, I, label='Fitted section')
plt.plot([v_float], [0.0], 'x', label=r'V$_{float}$')
# plt.plot(V_full, I_fitted, label='Fit')
# plt.plot(V, I_fitted_simple, label='Simple')
# plt.plot(V, I_sam, label='Sam\'s Params')
plt.xlabel(r'$\hat{V}$')
plt.ylabel(r'$\hat{I}$')
plt.axhline(y=0, color='gray', linewidth=1, linestyle='dashed')
plt.axvline(x=v_float, color='gray', linewidth=1, linestyle='dashed')
plt.legend()
# plt.plot(I)
# plt.plot(V)
# plt.show()

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

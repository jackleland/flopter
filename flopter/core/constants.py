
# IV Variable Labels
TIME = 't'
POTENTIAL = 'V'
CURRENT = 'I'
ELEC_CURRENT = 'I_e'
ION_CURRENT = 'I_i'

# Fit Variable Labels
ION_SAT = 'I_sat'       # Ion saturation current label
SHEATH_EXP = 'a'        # Sheath expansion parameter label
FLOAT_POT = 'V_f'       # Floating potential label
ELEC_TEMP = 'T_e'       # Electron temperature label
ELEC_DENS = 'n_e'       # Electron density label
ELEC_MASS = 'm_e'       # Electron mass label
ION_MASS = 'm_i'        # Ion mass label
FLOW_VEL = 'v_0'        # Flow velocity label
THERM_VEL = 'v_th'      # Thermal velocity label
DIST_SCALER = 'A'       # Distribution function scaler
AMPLITUDE = 'A'         # Periodic function amplitude
PERIOD = 'P'            # Periodic function period
OFFSET_Y = 'y_0'        # Periodic function y-offset
OFFSET_X = 'x_0'        # Periodic function x-offset
ST_DEV = 'sigma'        # Standard deviation in a distribution function
GRADIENT = 'm'          # Linear function gradient
EXP_SCALER = 'b'        # Exponential function x-scaler
TTIV_SCALER = 'alpha'   # Two-temperature IV scaler
TTIV_TEMP = 'T_e2'      # Two-temperature IV electron temperature label
TTIV_ISAT = 'I_sat2'      # Two-temperature IV saturation current label
TTIV_DENS = 'n_e2'      # Two-temperature IV electron density label
TTIV_VFLOAT = 'V_f2'    # Two-temperature IV floating potential label
TTIV_SHEX = 'a2'        # Two-temperature IV sheath exapnsion parameter label
SSE_PARAM = 'delta'     # Secondary Electron Emission Parameter

# FitData Dictionary Labels
RAW_X = 'raw_x'
RAW_Y = 'raw_y'
FIT_Y = 'fit_y'
ERROR_STRING = 'd_{}'
SIGMA = 'sigma'
CHI2 = 'chi2'
REDUCED_CHI2 = 'reduced_chi2'

# SPICE Homogenisation constants
SWEEP_LOWER = -9.95
SWEEP_UPPER = 10.05
PROBE_PARAMETER = 3
WALL_PARAMETER = 2
SPICE_SPECIES_ION = 1
SPICE_SPECIES_ELECTRON = 2

# Input File Variables Labels
INF_MAGNETIC_FIELD = 'B'
INF_KSI = 'ksi'
INF_MU = 'mu'
INF_TAU = 'tau'
INF_TIME_END = 'tp'
INF_TIME_AV = 'ta'
INF_TIME_SWEEP = 'tc'
INF_PART_PER_CELL = 'Npc'
INF_SWEEP_PARAM = 'param1'
INF_SHAPE_NAMES = ['rectangle', 'triangle', 'circle']    # Indexed in order listed in input file
INF_DIAGS_NUM = 'no_diag_reg'

# Input File Diagnostic Specific Labels
INF_DIAG_NAME = 'diag_name'
INF_DIAG_PROPERTY = 'record_property'
INF2_DIAG_ZLOW = 'z_low'
INF2_DIAG_ZHIGH = 'z_high'
INF2_DIAG_YLOW = 'y_low'
INF2_DIAG_YHIGH = 'y_high'
INF3_DIAG_ZLOW = 'zlow'
INF3_DIAG_ZHIGH = 'zhigh'
INF3_DIAG_YLOW = 'ylow'
INF3_DIAG_YHIGH = 'yhigh'
INF3_DIAG_XLOW = 'xlow'
INF3_DIAG_XHIGH = 'xhigh'

# Input File Section Headers
INF_SEC_PLASMA = 'plasma'
INF_SEC_GEOMETRY = 'geom'
INF_SEC_CONTROL = 'control'
INF_SEC_DIAG = 'diag_reg'
INF_SEC_DIAGS = 'num_diag_regions'
INF_SEC_SPECIE = 'specie'
INF_SEC_SHAPES = 'num_blocks'

# Constants for conversion types
CONV_POTENTIAL = 'potential'
CONV_CURRENT = 'current'
CONV_LENGTH = 'length'
CONV_TIME = 'time'
CONV_VELOCITY = 'velocity'
CONV_IV = 'iv_data'
CONV_DENSITY = 'density'
CONV_MASS = 'mass'
CONV_CHARGE = 'charge'
CONV_TEMPERATURE = 'temperature'
CONV_FLUX = 'flux'
CONV_DIST_FUNCTION = 'dist_function'


# Standardised diagnostic names for Spice
DIAG_PROBE_POT = 'ProbePot'
DIAG_WALL_POT = 'WallPot'
DIAG_DIST_FUNCTION_HIST = 'Hist'


# Labels for magnum fast adc reader's header
MAGADC_FREQ = 'freq'
MAGADC_NUM = 'number'
MAGADC_ACTIVE = 'active'
MAGADC_DSIZE = 'dsize'
MAGADC_TSIZE = 'tsize'
MAGADC_HSIZE = 'hsize'
MAGADC_CH_OFFSET = 'offset'
MAGADC_CH_SENS = 'sensitivity'
MAGADC_CH_NAME = 'name'


# Physical Constants
# TODO: Replace with scipy
BOLTZMANN = 1.38064852e-23  # m^2 kg s^-2 K^-1
EPSILON_0 = 8.85418782e-12  # m^-3 kg^-1 s^4 A^2
ELEM_CHARGE = 1.60217662e-19 # C
PROTON_MASS = 1.6726219e-27 # kg
DEUTERIUM_MASS = 2.01410178 * PROTON_MASS # kg
ELECTRON_MASS = 9.10938356e-31 # kg
ATOMIC_MASS_UNIT = ELECTRON_MASS * 1822.888486
P_E_MASS_RATIO = PROTON_MASS / ELECTRON_MASS
I_E_MASS_RATIO = DEUTERIUM_MASS / ELECTRON_MASS


# Default plotting settings for matplotlib.pyplot
AX_LINE_DEFAULTS = {
    'color': 'grey',
    'linewidth': .9,
    'linestyle': '--'
}

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

# Input File Section Headers
INF_SEC_PLASMA = 'plasma'
INF_SEC_GEOMETRY = 'geom'
INF_SEC_CONTROL = 'control'
INF_SEC_DIAG = 'diag_reg'
INF_SEC_SPECIE = 'specie'
INF_SEC_SHAPES = 'num_blocks'

# Constants for conversion types
CONV_POTENTIAL = 'potential'
CONV_CURRENT = 'current'
CONV_LENGTH = 'length'
CONV_TIME = 'time'
CONV_IV = 'iv_data'
CONV_DIST_FUNCTION = 'dist_function'

# Standardised diagnostic names for Spice
DIAG_PROBE_POT = 'ProbePot'
DIAG_DIST_FUNCTION_HIST = 'Hist'


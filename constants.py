
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

# Homogenisation constants
SWEEP_LOWER = -9.95
SWEEP_UPPER = 10.05
PROBE_PARAMETER = 3
WALL_PARAMETER = 2

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
INF_DIAG_NAME = 'diag_name'
INF_DIAG_PROPERTY = 'record_property'

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


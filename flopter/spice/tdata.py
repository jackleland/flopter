import scipy.io as sio
import numpy as np

import flopter.spice.normalise
from flopter.core import constants as c, normalise as norm

NZ = 'Nz'
NZMAX = 'Nzmax'
NY = 'Ny'
COUNT = 'count'
HPOS = 'hpos'
DELTAH = 'deltah'
NPC = 'Npc'
DT = 'dt'
DZ = 'dz'
NPROC = 'nproc'
Q = 'q'
M = 'm'
TEMP = 'Temp'
ZG = 'zg'
YG = 'yg'
RHO = 'rho'
ESCZ = 'Escz'
ESCY = 'Escy'
SURFACEMATRIX = 'surfacematrix'
POT = 'Pot'
POTVAC = 'Potvac'
SLICEPROC = 'sliceproc'
EDGECHARGE = 'edgecharge'
T = 't'
SNUMBER = 'snumber'
TOTALENERGY = 'totalenergy'
ESCT = 'Esct'
DPHIQN = 'dPHIqn'
PCHI = 'pchi'
BX = 'bx'
BY = 'by'
BZ = 'bz'
NPARTPROC = 'npartproc'
NODIAGREG = 'nodiagreg'
DIAGHISTORIES = 'diaghistories'
FVARRAYS = 'fvarrays'
FVBIN = 'fvbin'
FVPERARRAYCOUNT = 'fvperparraycount'
FVLIMITS = 'fvlimits'
HISTLIMITS = 'histlimits'
TIMEHISTORY_UPPER = 'timehistory_upper'
TIMEHISTORY_LOWER = 'timehistory_lower'
FLAGM = 'flagm'
EQUIPOTM = 'equipotm'
FLAG = 'flag'
ITERTIME_UPPER = 'itertime_upper'
ITERTIME_LOWER = 'itertime_lower'
INJRATE = 'injrate'
OBJECTS = 'objects'
EDGES = 'edges'
DIAGM = 'diagm'
OBJECTSENUM = 'objectsenum'
OBJECTSCURRENTI = 'objectscurrenti'
OBJECTSCURRENTE = 'objectscurrente'
OBJECTSCURRENTFLUXI = 'objectspowerfluxi'
OBJECTSCURRENTFLUXE = 'objectspowerfluxe'
RHO1 = 'rho01'
SOLW01 = 'solw01'
SOLNS01 = 'solns01'
RHO2 = 'rho02'
SOLW02 = 'solw02'
SOLNS02 = 'solns02'
KSI = 'ksi'
TAU = 'tau'
MU =  'mu'
ALPHAYZ = 'alphayz'
ALPHAXZ = 'alphaxz'
NC = 'Nc'
NA = 'Na'
NP = 'Np'
IREL = 'irel'
FLOATCONSTANT = 'floatconstant'

# MKS Variables - Spice-2 Specific
MKSN0 = 'mksn0'
MKSTE = 'mksTe'
MKSB = 'mksB'
MKSMAINIONM = 'mksmainionm'
MKSMAINIONQ = 'mksmainionq'
MKSPAR1 = 'mkspar1'
MKSPAR2 = 'mkspar2'
MKSPAR3 = 'mkspar3'


# MATLAB specific values returned by the loadmat function
HEADER = '__header__'
VERSION = '__version__'
GLOBALS = '__globals__'

VERSION_SPICE = 'version'  # used by SPICE


# Lists of labels specific to each version, those not in these lists are assumed to be diagnostic outputs.
_MATLAB_LABELS = {HEADER, VERSION, GLOBALS}
_GENERAL_LABELS = {VERSION_SPICE, NZ, NZMAX, NY, COUNT, HPOS, DELTAH, NPC, DT, DZ, NPROC, Q, M, TEMP, ZG, YG, RHO, ESCZ, ESCY,
                   SURFACEMATRIX, POT, POTVAC, SLICEPROC, EDGECHARGE, T, SNUMBER, TOTALENERGY, ESCT, DPHIQN, PCHI, BX,
                   BY, BZ, NPARTPROC, NODIAGREG, DIAGHISTORIES, FVARRAYS, FVBIN, FVPERARRAYCOUNT, FVLIMITS, HISTLIMITS,
                   TIMEHISTORY_UPPER, TIMEHISTORY_LOWER, FLAGM, EQUIPOTM, FLAG, ITERTIME_UPPER, ITERTIME_LOWER, INJRATE,
                   OBJECTS, EDGES, DIAGM, OBJECTSENUM, OBJECTSCURRENTI, OBJECTSCURRENTE, OBJECTSCURRENTFLUXI,
                   OBJECTSCURRENTFLUXE, RHO1, SOLW01, SOLNS01, RHO2, SOLW02, SOLNS02, KSI, TAU, MU, ALPHAYZ, ALPHAXZ,
                   NC, NA, NP, IREL, FLOATCONSTANT}
_SPICE2_LABELS = {MKSN0, MKSTE, MKSB, MKSMAINIONM, MKSMAINIONQ, MKSPAR1, MKSPAR2, MKSPAR3}
_SPICE3_LABELS = ()

_GENERAL_CONV_TYPES = {
    DT: c.CONV_TIME,
    DZ: c.CONV_LENGTH,
    M: c.CONV_MASS,
    Q: c.CONV_CHARGE,
    TEMP: c.CONV_TEMPERATURE,
    T: c.CONV_TIME,
    OBJECTSCURRENTE: c.CONV_CURRENT,
    OBJECTSCURRENTI: c.CONV_CURRENT,
    OBJECTSCURRENTFLUXE: c.CONV_FLUX,
    OBJECTSCURRENTFLUXI: c.CONV_FLUX,
    POT: c.CONV_POTENTIAL,
    POTVAC: c.CONV_POTENTIAL
}
_SPICE2_CONV_TYPES = {
    MKSN0: c.CONV_DENSITY,
    MKSTE: c.CONV_TEMPERATURE,
    MKSMAINIONQ: c.CONV_CHARGE,
    MKSMAINIONM: c.CONV_MASS
}
DIAGNOSTIC_CONV_TYPES = {
    'Pot': c.CONV_POTENTIAL,
    'Hist': c.CONV_DIST_FUNCTION
}
DEFAULT_REDUCED_DATASET = set(_GENERAL_CONV_TYPES.keys()) | {c.DIAG_PROBE_POT, ALPHAXZ, ALPHAYZ, OBJECTSENUM, NPC, NZ,
                                                             NY}


class MatlabData(object):
    def __init__(self, tfile_dict):
        self.version = tfile_dict[VERSION]
        self.header = tfile_dict[HEADER]
        self.globals = tfile_dict[GLOBALS]


class SpiceTData(object):
    _ALL_LABELS = _GENERAL_LABELS
    _ALL_CONV_TYPES = _GENERAL_CONV_TYPES

    def __init__(self, t_filename, deallocate=False, converter=None, convert=False, variable_names=None, reduce=True):
        # TODO: (2019-04-11) This would be better implemented by storing everything in a dictionary, leaving all the
        #  member variables None, and overriding the __get_attribute__() method to read from the dictionary instead.
        # Read matlab filename into dictionary and then distribute contents into named variables
        self.t_filename = t_filename
        self.converter = converter
        self.has_converted = {label: False for label in self._ALL_CONV_TYPES}

        self.t_file_sizes = {label: size for label, size, dtype in sio.whosmat(t_filename)}
        self.t_file_labels = set(self.t_file_sizes.keys())

        self.diagnostic_labels = self.t_file_labels - self._ALL_LABELS - _MATLAB_LABELS

        self.t_dict = sio.loadmat(t_filename, variable_names=variable_names)

        # Fill in all variables in the file but not read allow member variables to be populated
        for label in self.t_file_labels:
            if label not in self.t_dict:
                self.t_dict[label] = None

        # Fill in all variables expected to be in the file but that weren't found (this can happen with different
        # versions of spice)
        for label in self._ALL_LABELS:
            if label not in self.t_dict:
                self.t_dict[label] = 'Not found in t-file'

        if self.t_dict[T] is not None and self.t_dict[DT] is not None and np.mean(self.t_dict[T]) == 0.0:
            print('WARNING: Encountered t-zeroing, creating an approximate t array')
            self.t_dict[T] = np.arange(1, len(self.t_dict[T]) + 1) * 2048.0 * self.t_dict[DT]

        self.matlab_data = MatlabData(self.t_dict)
        self.version = self.t_dict[VERSION_SPICE]

        # Convertible variables
        self.dt = self.t_dict[DT]
        self.dz = self.t_dict[DZ]
        self.t = self.t_dict[T]
        self.q = self.t_dict[Q]
        self.m = self.t_dict[M]
        self.temp = self.t_dict[TEMP]
        self.objectscurrenti = self.t_dict[OBJECTSCURRENTI]
        self.objectscurrente = self.t_dict[OBJECTSCURRENTE]
        self.objectspowerfluxi = self.t_dict[OBJECTSCURRENTFLUXI]
        self.objectspowerfluxe = self.t_dict[OBJECTSCURRENTFLUXE]
        self.pot = self.t_dict[POT]
        self.potvac = self.t_dict[POTVAC]

        # Other variables
        self.nz = self.t_dict[NZ]
        self.nzmax = self.t_dict[NZMAX]
        self.ny = self.t_dict[NY]
        self.count = self.t_dict[COUNT]
        self.hpos = self.t_dict[HPOS]
        self.deltah = self.t_dict[DELTAH]
        self.npc = self.t_dict[NPC]
        self.nproc = self.t_dict[NPROC]
        self.zg = self.t_dict[ZG]
        self.yg = self.t_dict[YG]
        self.rho = self.t_dict[RHO]
        self.escz = self.t_dict[ESCZ]
        self.escy = self.t_dict[ESCY]
        self.esct = self.t_dict[ESCT]
        self.surfacematrix = self.t_dict[SURFACEMATRIX]
        self.sliceproc = self.t_dict[SLICEPROC]
        self.edgecharge = self.t_dict[EDGECHARGE]
        self.snumber = self.t_dict[SNUMBER]
        self.totalenergy = self.t_dict[TOTALENERGY]
        self.dphiqn = self.t_dict[DPHIQN]
        self.pchi = self.t_dict[PCHI]
        self.bx = self.t_dict[BX]
        self.by = self.t_dict[BY]
        self.bz = self.t_dict[BZ]
        self.npartproc = self.t_dict[NPARTPROC]
        self.nodiagreg = self.t_dict[NODIAGREG]
        self.diaghistories = self.t_dict[DIAGHISTORIES]
        self.fvarrays = self.t_dict[FVARRAYS]
        self.fvbin = self.t_dict[FVBIN]
        self.fvperparraycount = self.t_dict[FVPERARRAYCOUNT]
        self.fvlimits = self.t_dict[FVLIMITS]
        self.histlimits = self.t_dict[HISTLIMITS]
        self.timehistory_upper = self.t_dict[TIMEHISTORY_UPPER]
        self.timehistory_lower = self.t_dict[TIMEHISTORY_LOWER]
        self.flagm = self.t_dict[FLAGM]
        self.equipotm = self.t_dict[EQUIPOTM]
        self.flag = self.t_dict[FLAG]
        self.itertime_upper = self.t_dict[ITERTIME_UPPER]
        self.itertime_lower = self.t_dict[ITERTIME_LOWER]
        self.injrate = self.t_dict[INJRATE]
        self.objects = self.t_dict[OBJECTS]
        self.edges = self.t_dict[EDGES]
        self.diagm = self.t_dict[DIAGM]
        self.objectsenum = self.t_dict[OBJECTSENUM]
        self.rho01 = self.t_dict[RHO1]
        self.solw01 = self.t_dict[SOLW01]
        self.solns01 = self.t_dict[SOLNS01]
        self.rho02 = self.t_dict[RHO2]
        self.solw02 = self.t_dict[SOLW02]
        self.solns02 = self.t_dict[SOLNS02]
        self.ksi = self.t_dict[KSI]
        self.tau = self.t_dict[TAU]
        self.mu = self.t_dict[MU]
        self.alphayz = self.t_dict[ALPHAYZ]
        self.alphaxz = self.t_dict[ALPHAXZ]
        self.nc = self.t_dict[NC]
        self.na = self.t_dict[NA]
        self.np = self.t_dict[NP]
        self.irel = self.t_dict[IREL]
        self.floatconstant = self.t_dict[FLOATCONSTANT]

        # Fix for if t has been zeroed


        self.diagnostics = {key: value for key, value in self.t_dict.items() if key in self.diagnostic_labels}
        for label in self.diagnostics:
            if any(marker in label for marker in DIAGNOSTIC_CONV_TYPES.keys()):
                self.has_converted[label] = False

        if deallocate:
            self.deallocate()

        if variable_names is not None and reduce:
            self.reduce(variable_names)

        if convert:
            self._convert(self.converter.__class__)

    def deallocate(self):
        # Deallocate the t_dict dictionary if deemed unnecessary to keep the data in dict form.
        del self.t_dict
        import gc
        gc.collect()

    def reduce(self, reduced_dataset):
        """
        Member function which allows for a reduced dataset to be stored in the
        object to reduce the memory costs running multiple instances of splopter
        at once for larger simulations.

        Works by going through a fully populated SpiceTData object and removing
        all the attributes which are not in the reduced_dataset list.

        :param reduced_dataset:     A list of strings corresponding to the
                                    member variable names that must be kept.

        """
        reduced_dataset = set(reduced_dataset)
        available_labels = self._ALL_LABELS | self.diagnostic_labels
        if not reduced_dataset.issubset(available_labels):
            unavailable_labels = reduced_dataset - available_labels
            raise ValueError('The reduced_dataset provided is not a list of labels which can be removed to reduce the '
                             'size of the TData object. List must be a subset of the available labels in the t-file.\n'
                             f'The following requested labels are not available: ({unavailable_labels})')

        for label in self._ALL_LABELS:
            if label not in reduced_dataset:
                delattr(self, label.lower())
        for diag_label in self.diagnostic_labels:
            if diag_label in self.diagnostics and diag_label not in reduced_dataset:
                self.diagnostics.pop(diag_label)

        import gc
        gc.collect()

    def set_converter(self, converter):
        self.converter = converter

    def _convert(self, converter_type):
        if self.converter and isinstance(self.converter, converter_type):
            for label, conv_type in self._ALL_CONV_TYPES.items():
                # TODO: This is messy and should be changed. Whole class should probably be a dictionary, but
                # TODO: this works for now and makes other bits of the code more pythonic
                if not self.has_converted[label]:
                    setattr(self, label.lower(), self.converter(getattr(self, label.lower()), conversion_type=conv_type))
                    self.has_converted[label] = True

            for diag_key, data in self.diagnostics.items():
                if diag_key in self.has_converted and not self.has_converted[diag_key]:
                    for marker, conv_type in DIAGNOSTIC_CONV_TYPES.items():
                        if marker in diag_key:
                            self.diagnostics[diag_key] = self.converter(self.diagnostics[diag_key], conv_type)
                            self.has_converted[diag_key] = True
        else:
            print('Not ready to be converted, no {} has been specified'.format(converter_type))

    def denormalise(self):
        self._convert(flopter.spice.normalise.Denormaliser)

    def normalise(self):
        self._convert(flopter.spice.normalise.Normaliser)


class Spice2TData(SpiceTData):
    _ALL_LABELS = _GENERAL_LABELS | _SPICE2_LABELS
    _ALL_CONV_TYPES = {**_GENERAL_CONV_TYPES, **_SPICE2_CONV_TYPES}

    def __init__(self, t_filename, deallocate=False, variable_names=None):
        super().__init__(t_filename, deallocate=False, variable_names=variable_names, reduce=False)

        self.mksn0 = self.t_dict[MKSN0]
        self.mkste = self.t_dict[MKSTE]
        self.mksb = self.t_dict[MKSB]
        self.mksmainionm = self.t_dict[MKSMAINIONM]
        self.mksmainionq = self.t_dict[MKSMAINIONQ]
        self.mkspar1 = self.t_dict[MKSPAR1]
        self.mkspar2 = self.t_dict[MKSPAR2]
        self.mkspar3 = self.t_dict[MKSPAR3]

        if deallocate:
            self.deallocate()

        if variable_names is not None:
            self.reduce(variable_names)


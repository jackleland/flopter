import scipy.io as spio

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


# Lists of labels specific to each version, those not in these lists are assumed to be diagnostic outputs.
_GENERAL_LABELS = (NZ,
    NZMAX,
    NY,
    COUNT,
    HPOS,
    DELTAH,
    NPC,
    DT,
    DZ,
    NPROC,
    Q,
    M,
    TEMP,
    ZG,
    YG,
    RHO,
    ESCZ,
    ESCY,
    SURFACEMATRIX,
    POT,
    POTVAC,
    SLICEPROC,
    EDGECHARGE,
    T,
    SNUMBER,
    TOTALENERGY,
    ESCT,
    DPHIQN,
    PCHI,
    BX,
    BY,
    BZ,
    NPARTPROC,
    NODIAGREG,
    DIAGHISTORIES,
    FVARRAYS,
    FVBIN,
    FVPERARRAYCOUNT,
    FVLIMITS,
    HISTLIMITS,
    TIMEHISTORY_UPPER,
    TIMEHISTORY_LOWER,
    FLAGM,
    EQUIPOTM,
    FLAG,
    ITERTIME_UPPER,
    ITERTIME_LOWER,
    INJRATE,
    OBJECTS,
    EDGES,
    DIAGM,
    OBJECTSENUM,
    OBJECTSCURRENTI,
    OBJECTSCURRENTE,
    OBJECTSCURRENTFLUXI,
    OBJECTSCURRENTFLUXE,
    RHO1,
    SOLW01,
    SOLNS01,
    RHO2,
    SOLW02,
    SOLNS02,
    KSI,
    TAU,
    MU,
    ALPHAYZ,
    ALPHAXZ,
    NC,
    NA,
    NP,
    IREL,
    FLOATCONSTANT)
_SPICE2_LABELS = (MKSB,
    MKSTE,
    MKSB,
    MKSMAINIONM,
    MKSMAINIONQ,
    MKSPAR1,
    MKSPAR2,
    MKSPAR3)
_SPICE3_LABELS = ()


class SpiceData(object):
    _ALL_LABELS = _GENERAL_LABELS

    def __init__(self, t_filename):
        t_data = spio.loadmat(t_filename)

        self.Nz = t_data[NZ]
        self.Nzmax = t_data[NZMAX]
        self.Ny = t_data[NY]
        self.count = t_data[COUNT]
        self.hpos = t_data[HPOS]
        self.deltah = t_data[DELTAH]
        self.Npc = t_data[NPC]
        self.dt = t_data[DT]
        self.dz = t_data[DZ]
        self.nproc = t_data[NPROC]
        self.q = t_data[Q]
        self.m = t_data[M]
        self.temp = t_data[TEMP]
        self.zg = t_data[ZG]
        self.yg = t_data[YG]
        self.rho = t_data[RHO]
        self.Escz = t_data[ESCZ]
        self.Escy = t_data[ESCY]
        self.surfacematrix = t_data[SURFACEMATRIX]
        self.pot = t_data[POT]
        self.potvac = t_data[POTVAC]
        self.sliceproc = t_data[SLICEPROC]
        self.edgecharge = t_data[EDGECHARGE]
        self.t = t_data[T]
        self.snumber = t_data[SNUMBER]
        self.totalenergy = t_data[TOTALENERGY]
        self.Esct = t_data[ESCT]
        self.dPHIqn = t_data[DPHIQN]
        self.pchi = t_data[PCHI]
        self.bx = t_data[BX]
        self.by = t_data[BY]
        self.bz = t_data[BZ]
        self.npartproc = t_data[NPARTPROC]
        self.nodiagreg = t_data[NODIAGREG]
        self.diaghistories = t_data[DIAGHISTORIES]
        self.fvarrays = t_data[FVARRAYS]
        self.fvbin = t_data[FVBIN]
        self.fvperparraycount = t_data[FVPERARRAYCOUNT]
        self.fvlimits = t_data[FVLIMITS]
        self.histlimits = t_data[HISTLIMITS]
        self.timehistory_upper = t_data[TIMEHISTORY_UPPER]
        self.timehistory_lower = t_data[TIMEHISTORY_LOWER]
        self.flagm = t_data[FLAGM]
        self.equipotm = t_data[EQUIPOTM]
        self.flag = t_data[FLAG]
        self.itertime_upper = t_data[ITERTIME_UPPER]
        self.itertime_lower = t_data[ITERTIME_LOWER]
        self.injrate = t_data[INJRATE]
        self.objects = t_data[OBJECTS]
        self.edges = t_data[EDGES]
        self.diagm = t_data[DIAGM]
        self.objectsenum = t_data[OBJECTSENUM]
        self.objectscurrenti = t_data[OBJECTSCURRENTI]
        self.objectscurrente = t_data[OBJECTSCURRENTE]
        self.objectspowerfluxi = t_data[OBJECTSCURRENTFLUXI]
        self.objectspowerfluxe = t_data[OBJECTSCURRENTFLUXE]
        self.rho01 = t_data[RHO1]
        self.solw01 = t_data[SOLW01]
        self.solns01 = t_data[SOLNS01]
        self.rho02 = t_data[RHO2]
        self.solw02 = t_data[SOLW02]
        self.solns02 = t_data[SOLNS02]
        self.ksi = t_data[KSI]
        self.tau = t_data[TAU]
        self.mu = t_data[MU]
        self.alphayz = t_data[ALPHAYZ]
        self.alphaxz = t_data[ALPHAXZ]
        self.Np = t_data[NC]
        self.Na = t_data[NA]
        self.Np = t_data[NP]
        self.irel = t_data[IREL]
        self.floatconstant = t_data[FLOATCONSTANT]

        self.diagnostics = {key: value for key, value in t_data.items() if key not in self._ALL_LABELS}


class Spice2Data(SpiceData):
    _ALL_LABELS = _GENERAL_LABELS + _SPICE2_LABELS

    def __init__(self, t_data):
        super().__init__(t_data)

        self.mksn0 = t_data[MKSN0]
        self.mksTe = t_data[MKSTE]
        self.mksB = t_data[MKSB]
        self.mksmainionm = t_data[MKSMAINIONM]
        self.mksmainionq = t_data[MKSMAINIONQ]
        self.mkspar1 = t_data[MKSPAR1]
        self.mkspar2 = t_data[MKSPAR2]
        self.mkspar3 = t_data[MKSPAR3]

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

    def __init__(self, t_filename, deallocate=True):
        # Read matlab filename into dictionary and then distribute contents into named variables
        self.t_filename = t_filename
        self.t_data = spio.loadmat(t_filename)

        self.Nz = self.t_data[NZ]
        self.Nzmax = self.t_data[NZMAX]
        self.Ny = self.t_data[NY]
        self.count = self.t_data[COUNT]
        self.hpos = self.t_data[HPOS]
        self.deltah = self.t_data[DELTAH]
        self.Npc = self.t_data[NPC]
        self.dt = self.t_data[DT]
        self.dz = self.t_data[DZ]
        self.nproc = self.t_data[NPROC]
        self.q = self.t_data[Q]
        self.m = self.t_data[M]
        self.temp = self.t_data[TEMP]
        self.zg = self.t_data[ZG]
        self.yg = self.t_data[YG]
        self.rho = self.t_data[RHO]
        self.Escz = self.t_data[ESCZ]
        self.Escy = self.t_data[ESCY]
        self.surfacematrix = self.t_data[SURFACEMATRIX]
        self.pot = self.t_data[POT]
        self.potvac = self.t_data[POTVAC]
        self.sliceproc = self.t_data[SLICEPROC]
        self.edgecharge = self.t_data[EDGECHARGE]
        self.t = self.t_data[T]
        self.snumber = self.t_data[SNUMBER]
        self.totalenergy = self.t_data[TOTALENERGY]
        self.Esct = self.t_data[ESCT]
        self.dPHIqn = self.t_data[DPHIQN]
        self.pchi = self.t_data[PCHI]
        self.bx = self.t_data[BX]
        self.by = self.t_data[BY]
        self.bz = self.t_data[BZ]
        self.npartproc = self.t_data[NPARTPROC]
        self.nodiagreg = self.t_data[NODIAGREG]
        self.diaghistories = self.t_data[DIAGHISTORIES]
        self.fvarrays = self.t_data[FVARRAYS]
        self.fvbin = self.t_data[FVBIN]
        self.fvperparraycount = self.t_data[FVPERARRAYCOUNT]
        self.fvlimits = self.t_data[FVLIMITS]
        self.histlimits = self.t_data[HISTLIMITS]
        self.timehistory_upper = self.t_data[TIMEHISTORY_UPPER]
        self.timehistory_lower = self.t_data[TIMEHISTORY_LOWER]
        self.flagm = self.t_data[FLAGM]
        self.equipotm = self.t_data[EQUIPOTM]
        self.flag = self.t_data[FLAG]
        self.itertime_upper = self.t_data[ITERTIME_UPPER]
        self.itertime_lower = self.t_data[ITERTIME_LOWER]
        self.injrate = self.t_data[INJRATE]
        self.objects = self.t_data[OBJECTS]
        self.edges = self.t_data[EDGES]
        self.diagm = self.t_data[DIAGM]
        self.objectsenum = self.t_data[OBJECTSENUM]
        self.objectscurrenti = self.t_data[OBJECTSCURRENTI]
        self.objectscurrente = self.t_data[OBJECTSCURRENTE]
        self.objectspowerfluxi = self.t_data[OBJECTSCURRENTFLUXI]
        self.objectspowerfluxe = self.t_data[OBJECTSCURRENTFLUXE]
        self.rho01 = self.t_data[RHO1]
        self.solw01 = self.t_data[SOLW01]
        self.solns01 = self.t_data[SOLNS01]
        self.rho02 = self.t_data[RHO2]
        self.solw02 = self.t_data[SOLW02]
        self.solns02 = self.t_data[SOLNS02]
        self.ksi = self.t_data[KSI]
        self.tau = self.t_data[TAU]
        self.mu = self.t_data[MU]
        self.alphayz = self.t_data[ALPHAYZ]
        self.alphaxz = self.t_data[ALPHAXZ]
        self.Np = self.t_data[NC]
        self.Na = self.t_data[NA]
        self.Np = self.t_data[NP]
        self.irel = self.t_data[IREL]
        self.floatconstant = self.t_data[FLOATCONSTANT]

        self.diagnostics = {key: value for key, value in self.t_data.items() if key not in self._ALL_LABELS}

        if deallocate:
            self.deallocate()

    def deallocate(self):
        # Deallocate the t_data dictionary as the data is duplicated into individual variables and therefore unnecessary
        del self.t_data
        import gc
        gc.collect()


class Spice2Data(SpiceData):
    _ALL_LABELS = _GENERAL_LABELS + _SPICE2_LABELS

    def __init__(self, t_filename, deallocate=True):
        super().__init__(t_filename, deallocate=False)

        self.mksn0 = self.t_data[MKSN0]
        self.mksTe = self.t_data[MKSTE]
        self.mksB = self.t_data[MKSB]
        self.mksmainionm = self.t_data[MKSMAINIONM]
        self.mksmainionq = self.t_data[MKSMAINIONQ]
        self.mkspar1 = self.t_data[MKSPAR1]
        self.mkspar2 = self.t_data[MKSPAR2]
        self.mkspar3 = self.t_data[MKSPAR3]

        if deallocate:
            self.deallocate()

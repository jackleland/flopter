from constants import POTENTIAL, CURRENT, TIME, ELEC_CURRENT, ION_CURRENT


class IVData(dict):
    """
        A dictionary which holds IV specific data, namely
            - Voltage
            - Current
            - Time
            - [Electron Current] (for simulations)
            - [Ion Current] (for simulations)
    """
    _DEFAULT_TRIM_BEG = 0.0
    _DEFAULT_TRIM_END = 1.0

    def __init__(self, voltage, total_current, time, e_current=None, i_current=None,
                 trim_beg=_DEFAULT_TRIM_BEG, trim_end=_DEFAULT_TRIM_END):
        super().__init__([
            (CURRENT, total_current),
            (POTENTIAL, voltage),
            (TIME, time)
        ])
        if e_current is not None:
            self[ELEC_CURRENT] = e_current
        if i_current is not None:
            self[ION_CURRENT] = i_current
        self.trim_beg = trim_beg
        self.trim_end = trim_end
        self.current = total_current
        self.voltage = voltage
        self.time = time

    def split(self):
        if ION_CURRENT in self.keys():
            return self[POTENTIAL], self[CURRENT], self[ION_CURRENT]
        else:
            return self[POTENTIAL], self[CURRENT]

    def trim(self, trim_beg=None, trim_end=None):
        if not trim_beg and not trim_end:
            if self.trim_beg == self._DEFAULT_TRIM_BEG and self.trim_end == self._DEFAULT_TRIM_END:
                print('WARNING: trim values unchanged from default, no trimming will take place')
                return
            else:
                trim_beg = self.trim_beg
                trim_end = self.trim_end

        full_length = len(self[CURRENT])
        start = int(full_length * trim_beg)
        stop = int(full_length * trim_end)
        for key, value in self.items():
            self[key] = value[start:stop]



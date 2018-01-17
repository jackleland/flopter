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
    def __init__(self, voltage, total_current, time, e_current=None, i_current=None):
        super().__init__([
            (CURRENT, total_current),
            (POTENTIAL, voltage),
            (TIME, time)
        ])
        if e_current is not None:
            self[ELEC_CURRENT] = e_current
        if i_current is not None:
            self[ION_CURRENT] = i_current

    def split(self):
        if ION_CURRENT in self.keys():
            return self[POTENTIAL], self[CURRENT], self[ION_CURRENT]
        else:
            return self[POTENTIAL], self[CURRENT]

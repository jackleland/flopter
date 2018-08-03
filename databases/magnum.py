from codac.datastore import client
import numpy as np
import external.magnumdbutils as ut
import external.readfastadc as adc
import re
import pandas as p

TIMES = 0
DATA = 1

TARGET_POS = 'TargetLinearCurPos'
TARGET_ROT = 'TargetRotationCurPos'
TARGET_TILT = 'TargetTiltingCurPos'
TARGET_VOLTAGE = 'TargetVoltage'
TARGET_VOLTAGE_PS = 'TargetVoltagePs'
PLASMA_STATE = 'PlasmaPlcState'
SOURCE_PUMP_SPEED = 'Rp1SoChSpeed'
HEATING_PUMP_SPEED = 'Rp3HeChSpeed'
TARGET_PUMP_SPEED = 'Rp5TaChSpeed'
TS_DENS_PROF = 'TsProfNe'
TS_DENS_PROF_D = 'TsProfNe_d'
TS_TEMP_PROF = 'TsProfTe'
TS_TEMP_PROF_D = 'TsProfTe_d'
TS_RAD_COORDS = 'TsRadCoords'
BEAM_DUMP_DOWN = 'BeamDumpDown'
BEAM_DUMP_UP = 'BeamDumpUp'
TRIGGER_START = 'MasterTrigStartFt'
TARGET_CHAMBER_PRESSURE = 'TaChPresBara'

DEFAULT_VARS = [
    TARGET_POS,
    TARGET_ROT,
    TARGET_TILT,
    TARGET_VOLTAGE,
    TARGET_VOLTAGE_PS,
    PLASMA_STATE,
    TS_DENS_PROF,
    TS_DENS_PROF_D,
    TS_TEMP_PROF,
    TS_TEMP_PROF_D,
    TS_RAD_COORDS,
    SOURCE_PUMP_SPEED,
    HEATING_PUMP_SPEED,
    TARGET_PUMP_SPEED,
    BEAM_DUMP_DOWN,
    BEAM_DUMP_UP,
    TRIGGER_START,
    TARGET_CHAMBER_PRESSURE
]
TS_VARS = [
    TS_DENS_PROF,
    TS_DENS_PROF_D,
    TS_TEMP_PROF,
    TS_TEMP_PROF_D,
    TS_RAD_COORDS
]

SLC_STATES = {
    4: 'At Standby',
    5: 'Ramp to Pulse A',
    6: 'At Pulse A',
    7: 'Ramp to Standby',
    16: 'Heating up',
    17: 'Ramp to Pulse B',
    18: 'At Pulse B'
}


class MagnumDB(object):
    TIME_RANGE_TEST = ut.make_time_range('2018-05-01 08:00:00', ut.get_hours(72))
    TIME_RANGE_MAIN = ut.make_time_range('2018-06-05 08:00:00', ut.get_hours(96))
    _REDUNDANCY_TIME = 10
    _MAX_DURATION = 30

    def __init__(self, time_range=None, time_stamp=None):
        self.db = client.connect('127.0.0.1')
        self.raw_data = self.db.root.chDir('_Raw')
        if time_range:
            self.time_range = time_range
            self.time_stamp = time_range.startTime
        elif time_stamp:
            self.time_stamp = time_stamp
            ts_start = self.time_stamp - ut.get_seconds(self._REDUNDANCY_TIME)
            ts_end = self.time_stamp + ut.get_seconds(self._REDUNDANCY_TIME + self._MAX_DURATION)
            self.time_range = client.TimeRange(ts_start, ts_end)
        else:
            self.time_range = self.TIME_RANGE_MAIN
            self.time_stamp = self.time_range.startTime

        print('Start Time: ', ut.human_time_str(self.time_stamp))
        print('End Time: ', ut.human_time_str(self.time_range.endTime))
        self.all_plasma_states = self.get_data(PLASMA_STATE, self.time_range)

        self.b_plasma_timeranges = {}
        for i, state in enumerate(self.all_plasma_states[DATA]):
            # Look only for states at pulse B
            if state != 18:
                continue
            start_time = self.all_plasma_states[TIMES][i]
            end_time = self.all_plasma_states[TIMES][i+1]
            self.b_plasma_timeranges[i] = client.TimeRange(start_time, end_time)
        self.b_times = [tr.startTime for tr in self.b_plasma_timeranges.values()]

    def get_data(self, search_name, time_range=None, numpify_fl=True, ref_time=None):
        """
        Wrapper method for extracting data from the magnum database.

        :param search_name:     Variable name to extract, in string form. Defaults are given at the top of this file,
                                more info on the differ intranet page.
        :param time_range:      [Optional] Time range to look between. By default uses the self.time_range specified or
                                found at initialisation.
        :param numpify_fl:      [Optional] Boolean - specifies whether to convert output to numpy arrays. Default true.
        :param ref_time:        [Optional] Reference timestamp for time conversion. If given all times will be
                                subtracted from this and converted into seconds.
        :return:                [time, data] of all data points of type search_name in the given time_range.
        """
        if not time_range:
            time_range = self.time_range

        db_var = self.db.findNode(search_name)[0]  # only take the first in list
        time, data = db_var.read(time_range, 0, 0, unit=client.SI_UNIT)
        if ref_time:
            time = [client.timetoposix(timestamp - ref_time) for timestamp in time]
        if numpify_fl and not isinstance(data, np.ndarray):
            # Check whether loaded data is a profile (e.g. TS) and convert each profile to a numpy array
            if isinstance(data, list):
                data = [np.array(data_list) for data_list in data]
            data = np.array(data)
        return [time, data]

    def get_data_dict(self, time_range=None, variables=None, numpify_fl=True, ref_time=None):
        if not variables:
            variables = DEFAULT_VARS

        if not time_range:
            time_range = self.time_range

        variable_vals = {}
        for var in variables:
            data = self.get_data(var, time_range, numpify_fl=numpify_fl, ref_time=ref_time)
            if len(data[0]) == 0:
                continue
            variable_vals[var] = data
        return variable_vals

    def get_ts_data(self, time_range=None, numpify_fl=True, ref_time=None):
        return self.get_data_dict(time_range=time_range, numpify_fl=numpify_fl, ref_time=ref_time,
                                  variables=[TS_TEMP_PROF, TS_TEMP_PROF_D, TS_DENS_PROF, TS_DENS_PROF_D, TS_RAD_COORDS])

    def pad_continuous_variable(self, data):
        t, d = data
        differ = np.diff(t)
        interval = np.min(differ)
        new_time = np.arange(t[0], t[-1] + interval, interval)
        new_data = np.zeros(np.shape(new_time))

        j = 0
        for i in range(len(new_time)):
            if self.is_roughly_equal(t[j], new_time[i]):
                new_data[i] = d[j]
                j += 1
            else:
                new_data[i] = d[j - 1]
        return new_time, new_data

    @staticmethod
    def is_roughly_equal(a, b, tolerance=.99):
        return (a * tolerance) <= b <= (a * (1 + (1 - tolerance)))

    @staticmethod
    def get_offset_times(data, offset):
        """
        Generate new time array for a magnum_db with a different offset, and convert into seconds.
        :param data:    MagnumDB data from the get_data method
        :param offset:  Offset timestamp to reposition data to. Must be a MagnumDB timestamp, i.e. a int64 value
        :return:        Offset MagnumDB data array
        """
        new_times = [client.timetoposix(time - offset) for time in data[0]]
        return [new_times, data[ut.DATA]]

    def get_shot_duration(self, start_time):
        # if start_time in self.all_plasma_states[TIMES]:
        i = self.all_plasma_states[TIMES].index(start_time)
        if i < len(self.all_plasma_states[TIMES]):
            end_time = self.all_plasma_states[TIMES][i+1]
            return end_time - start_time

    @classmethod
    def get_approx_time_range(cls, shot_number=None, filename=None):
        if not shot_number and filename is not None:
            shot_number = re.search(adc.FILE_TIMESTAMP_REGEX, filename).group(1)
        if not shot_number and not filename:
            raise ValueError('No valid shot number given, need one of shot_number or filename to be specified.')

        time_stamp = int(shot_number)

        ts_start = time_stamp - ut.get_seconds(cls._REDUNDANCY_TIME)
        ts_end = time_stamp + ut.get_seconds(cls._REDUNDANCY_TIME + cls._MAX_DURATION)
        time_range = client.TimeRange(ts_start, ts_end)
        return time_range

    def get_time_range(self, shot_number=None, filename=None):
        """
        Attempts to find the corresponding b-state plasma for a given shot_number. Loops through all b-plasma states and
        compares the given shot_number string to each. If no exact match found (unlikely due to precision) then it loops
        through the timestamps bit wise and returns the most precise time match.

        Note that this method is now unnecessary given the ability to define a new time range from a given shot
        timestamp.
        """
        if not shot_number and filename is not None:
            shot_number = re.search(adc.FILE_TIMESTAMP_REGEX, filename).group(1)
        if not shot_number and not filename:
            raise ValueError('No valid shot number given, need one of shot_number or filename to be specified.')

        possible_matches = {}
        for index in self.b_plasma_timeranges.keys():
            time_stamp = self.all_plasma_states[TIMES][index]

            if shot_number == str(time_stamp):
                return self.b_plasma_timeranges.get(index)
            for j in range(len(str(time_stamp)) - 8):
                for k in [-1, 0, 1]:
                    trunc_time_stamp = str(time_stamp + (k * (1*(10**(j + 1)))))[:-(j + 1)]
                    if trunc_time_stamp == shot_number[:-(j + 1)]:
                        possible_matches[index] = j
        if len(possible_matches) == 0:
            print('Shot number {} not found in time range {}. Returning None...'.format(shot_number, self.time_range))
            return None
        if len(possible_matches) == 1:
            index = list(possible_matches.keys())[0]
            return index, self.b_plasma_timeranges[index]
        else:
            return max(possible_matches.values())

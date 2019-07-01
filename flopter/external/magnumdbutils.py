import codac.datastore.client as client
import datetime as dt
import numpy as np

TIMES = 0
DATA = 1

TARGET_POS = 'TargetLinearCurPos'
TARGET_ROT = 'TargetRotationCurPos'
TARGET_TILT = 'TargetTiltingCurPos'
PLASMA_STATE = 'PlasmaPlcState'
SOURCE_PUMP_SPEED = 'Rp1SoChSpeed'
HEATING_PUMP_SPEED = 'Rp3HeChSpeed'
TARGET_PUMP_SPEED = 'Rp5TaChSpeed'
SLC_STATES = {
    4: 'At Standby',
    5: 'Ramp to Pulse A',
    6: 'At Pulse A',
    7: 'Ramp to Standby',
    16: 'Heating up',
    17: 'Ramp to Pulse B',
    18: 'At Pulse B'
}


def get_seconds(seconds):
    return seconds * (1 << 32)


def get_minutes(minutes):
    return get_seconds(minutes * 60)


def get_hours(hours):
    return get_minutes(hours * 60)


def make_time_range(start_str, duration):
    start_time = client.datetotime(dt.datetime.strptime(start_str, client.DATE_FORMAT))
    end_time = start_time + duration
    return client.TimeRange(start_time, end_time)


def make_time_range_from_dates(start_str, end_str):
    start_time = client.datetotime(dt.datetime.strptime(start_str, client.DATE_FORMAT))
    end_time = client.datetotime(dt.datetime.strptime(end_str, client.DATE_FORMAT))
    return client.TimeRange(start_time, end_time)


def get_data_si(db, search_name, time_range, numpify_fl=True):
    db_var = db.findNode(search_name)[0] # only take the first in list
    if numpify_fl:
        return np.array(db_var.read(time_range, 0, 0, unit=client.SI_UNIT))
    else:
        return db_var.read(time_range, 0, 0, unit=client.SI_UNIT)
    # Return 2 lists: timestamp list, data list


def human_time_str(db_time_stamp):
    return client.timetostring(int(db_time_stamp) & 0xffffffff00000000).split()[1]


def human_date_str(db_time_stamp):
    return client.timetostring(int(db_time_stamp) & 0xffffffff00000000).split()[0]


def human_datetime_str(db_time_stamp):
    return client.timetostring(int(db_time_stamp) & 0xffffffff00000000)


def human_time_ms_str(db_time_stamp):
    # split into date, time
    # take time part
    time_string = client.timetostring(db_time_stamp).split()[1]
    if "." in time_string:
        # remove "000" at the end
        return time_string[:-3]
    else:
        # add ".000" at the end
        return time_string + ".000"


# --------------------- Main program -----------------------------------

if __name__ == '__main__':
    db_1 = client.connect('localhost')
    time_range = make_time_range("2017-02-02 12:00:00", get_hours(8))
    plasma_plc_state = get_data_si(db_1, "PlasmaPlcState", time_range)

    num_data = len(plasma_plc_state[DATA])
    if num_data > 0:
        print("Average: " + str(np.mean(plasma_plc_state[DATA])))

    plasma_at_pulse_a = 0
    for i in range(num_data):
        if plasma_plc_state[DATA][i] == 6:
            plasma_at_pulse_a = plasma_plc_state[TIMES][i]
            print("PulseA start: ", human_time_str(plasma_plc_state[TIMES][i]))
        else:
            if plasma_at_pulse_a != 0:
                print("PulseA stop:  ", human_time_str(plasma_plc_state[TIMES][i]))
                print("Duration:     ", (plasma_plc_state[TIMES][i] - plasma_at_pulse_a) / get_seconds(1))
                plasma_at_pulse_a = 0
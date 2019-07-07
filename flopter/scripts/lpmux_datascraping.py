import urllib.request
import numpy as np
import os
import codac.datastore.client as client
import datetime



number = 1
allowNumbers = [1, 2]
REAL_IO = True
channels = {
    1: np.int16,
    2: np.int16,
    3: np.uint16,
    4: np.uint16,
    5: np.uint16,
    6: np.uint16,
    19: np.uint16
}
channel_names = (
    'voltage',
    'current',
    'vfloat_1',
    'vfloat_2',
    'vfloat_3',
    'vfloat_4',
    'active_probe'
)

if number == 1 and number in allowNumbers :
    _url = "http:/leland/data?CHANNEL={}"
    if REAL_IO:
        # The real code

        # Construct filenames and folders from current datetime and magnum-db timestamp
        timestamp = client.now()
        date_now = datetime.datetime.now()
        date_time_str = date_now.strftime("%Y_%m_%d_%Hh-%Mm-%Ss")
        date_iso_str = date_now.strftime("%Y_%m_%d")
        output_filename = 'lpmux_{}_{}.dat'.format(date_iso_str, timestamp)
        output_folder = '\\\\Rijnh\\Data\\Magnum\\ADC\\{}\\{}_Leland\\'.format(date_now.year, date_time_str)
        os.makedirs(output_folder, exist_ok=True)
        output_filepath = output_folder + output_filename
        file_size = 0

        # Attempt 1 (python write binary file and urllib)
        with open(output_filepath, 'wb') as output_file:
            for i in channels.keys():
                with urllib.request.urlopen(_url.format(i)) as _src:
                    _data = _src.read()
                    output_file.write(_data)
                file_size += len(_data)
            _errorMsg = ("Wrote {:d} bytes to {}".format(file_size, output_filepath))

        # Attempt 2 (linux command)
        os.chdir(output_folder)
        for i in channels.keys():
            os.system('curl http://lpmux01/data?CHANNEL={} >> {}'.format(i, output_filename))
        _errorMsg = ("Wrote {:d} bytes to {}".format(file_size, output_filepath))

        # Attempt 3 (numpy)
        data = []
        for ch, d_type in channels.items():
            with urllib.request.urlopen(_url.format(ch)) as _src:
                _data = _src.read()
                data.append(np.frombuffer(_data, dtype=d_type))
            file_size += len(_data)
        data = np.rec.fromarrays(data, channel_names)
        data.tofile(output_filename)
        _errorMsg = ("Wrote {:d} bytes to {}".format(file_size, output_filepath))



    else :
        # Fake/Test code
        with urllib.request.urlopen(_url) as _src:
            _data = _src.read()
        _errorMsg = ("1 Got %d bytes" % len(_data))
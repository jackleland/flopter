
# coding: utf-8
import traceback
import os, struct
import numpy as np
from os.path import join



def read_header(f):
    header = dict(zip(['freq', 'number', 'version', 'active'], 
                       struct.unpack(">diHH", f.read(16))))
    header['dsize'] = 2 * header['number']
    header['tsize'] = header['dsize'] * bin(header['active']).count('1')
    print('Frequency=', header['freq'])
    print('Nr samples=', header['number'])
    print('Version=', header['version'])
    print('Active Channels=', bin(header['active']))
    if header['version'] == 0:
        header['hsize'] = 160
        format = ">2d" + 8*"16s"
        # Version 0 is for OLD Pilot data
    elif header['version'] == 1:
        # We now use Version 1
        header['hsize'] = 272
        format = ">" + 2*"8d" + 8*"16s"
    data = struct.unpack(format, f.read(header['hsize'] - 16))
    for i in range(8):
        header[i] = dict(zip(['offset', 'sensitivity', 'name'], data[i::8]))
        print('Channel', i, 'Name =', header[i]['name'])
        print('Channel', i, 'Offset =', header[i]['offset'])
        print('Channel', i, 'Sensitivity =', header[i]['sensitivity'])
    return header


def print_data(ch, d):
    _size = len(d)
    print('Channel %i Got data %i values' % (ch, _size))
    _i = 100
    while _i < 1000 and _i < _size :
        print('\t', _i, d[_i])
        _i += 100


def process_adc_file(directory, filename):
    print('processing file: %s' % filename)
    with open(join(directory, filename), 'rb') as f:
        header = read_header(f)
        if header['dsize'] == 0:
            return
        for ch in (ch for ch in range(8) if ((header['active'] >> ch) & 1)):
            rawData = np.frombuffer(f.read(header['dsize']), dtype='>i2')
            if len(rawData) == 0:
                print("no data in file for adc %i??" % i)
                continue
            _offset      = header[ch]['offset']
            _sensitivity = header[ch]['sensitivity']
            #print("Offset=",     _offset)
            #print("Sensitivity", _sensitivity)
            physData = _offset + rawData * _sensitivity
            print_data(ch, physData)
    f.close()


if __name__ == '__main__':
    _dir  = r"\\Differ\data\Magnum\ADC\Test"
    _file = "2017-11-02 17h 06m 55s TT_06483849332130951678.adc"
    try :
        process_adc_file(_dir, _file)
    except :
        traceback.print_exc()

    _cmd = input("Ready, Press Enter: ")

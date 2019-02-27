import numpy as np
import matplotlib.pyplot as plt
import codac as cdc
from codac.datastore import client
import external.magnumdbutils as ut
from magopter import Magopter
import os
import glob

"""
A script for reading in the data from the spectrometer for 
"""

SPECTROSCOPY_VARS = [
    'AvantesCh1Calibration',
    'AvantesCh1Counts',
    'AvantesCh1DarkCounts',
    'AvantesCh1DelTime',
    'AvantesCh1Description',
    'AvantesCh1IntTime',
    'AvantesCh1NrOfAvg',
    'AvantesCh1SmoothPx',
    'AvantesCh2Calibration',
    'AvantesCh2Counts',
    'AvantesCh2DarkCounts',
    'AvantesCh2DelTime',
    'AvantesCh2Description',
    'AvantesCh2IntTime',
    'AvantesCh2NrOfAvg',
    'AvantesCh2SmoothPx',
    'AvantesCh3Counts',
    'AvantesCh3DarkCounts',
    'AvantesCh3DelTime',
    'AvantesCh3Description',
    'AvantesCh3IntTime',
    'AvantesCh3NrOfAvg',
    'AvantesCh3SmoothPx',
    'AvantesCh4Counts',
    'AvantesCh4DarkCounts',
    'AvantesCh4DelTime',
    'AvantesCh4Description',
    'AvantesCh4IntTime',
    'AvantesCh4NrOfAvg',
    'AvantesCh4SmoothPx',
    'AvantesCh5Counts',
    'AvantesCh5DarkCounts',
    'AvantesCh5DelTime',
    'AvantesCh5Description',
    'AvantesCh5IntTime',
    'AvantesCh5NrOfAvg',
    'AvantesCh5SmoothPx',
    'AvantesCh6Counts',
    'AvantesCh6DarkCounts',
    'AvantesCh6DelTime',
    'AvantesCh6Description',
    'AvantesCh6IntTime',
    'AvantesCh6NrOfAvg',
    'AvantesCh6SmoothPx',
    'AvantesTrigStartFt'
]

# Get all folders and files associated with Magnum probe data
data_path = Magopter.get_data_path()
folders = next(os.walk(data_path))[1]

files = []
file_folders = []
for folder1 in folders:
    os.chdir(Magopter.get_data_path() + folder1)
    files.extend(glob.glob('*.adc'))
    file_folders.extend([folder1] * len(glob.glob('*.adc')))

files.sort()
folder = file_folders[-2] + '/'

# Collect files of interest
# files_oi = files[285:297]
files_oi = files[285:286]

for i, file in enumerate(files_oi):
    print('{}:    {}'.format(i, file))

# Put data variables from the Magnum database into a dictionary - filenames given as the key
spectroscopy_data = {}
for f in files_oi:
    m = Magopter(folder, f)
    spectroscopy_data[f] = m.magnum_db.get_data_dict(variables=SPECTROSCOPY_VARS, ref_time=m.timestamp)

# Print the contents of the dictionary and their respective sizes.
for f, data_dict in spectroscopy_data.items():
    print(f)
    for spect_var, spect_data in data_dict.items():
        print('{}: Shape = {}'.format(spect_var, [len(spect_data), np.shape(spect_data[1])]))
    print()


# Messy bit of code for visualising the first channel spectrum and overlaying some of the Balmer lines on top.
data_dict = spectroscopy_data['2018-06-07 14h 38m 33s TT_06564321037878221702.adc']

print(data_dict.keys())
print(np.shape(data_dict['AvantesCh1Counts'][1]))

wavelength_ranges = []
wavelength_ranges.append(np.linspace(299, 451, 2048))
wavelength_ranges.append(np.linspace(448, 579, 2048))
wavelength_ranges.append(np.linspace(378, 476, 2048))
wavelength_ranges.append(np.linspace(485, 558, 2048))
wavelength_ranges.append(np.linspace(599, 643, 2048))
wavelength_ranges.append(np.linspace(600, 950, 2048))

# plt.figure()
for i in range(6):
    plt.figure()
    plt.semilogy(wavelength_ranges[i] + 6.8, data_dict['AvantesCh{}Counts'.format(i+1)][1][0], label='Channel {}'
                 .format(i))
    for h_line in [ 434.0, 410.2, 397.0]:
        plt.axvline(x=h_line, color='red')
    plt.legend()
plt.show()

# Output the first data dictionary for Daljeet to analyse (1st measurement from 1st shot) [1/205]
np.save('/home/jleland/Data/spectal_data_magnum18.bin', data_dict['AvantesCh1Counts'][1][0])

# TODO: Read relevant arrays into an xarray object and save as a netcdf

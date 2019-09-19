import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import xarray as xr 
import scipy.stats as stat
import sys
import os
import glob
import re
sys.path.append('/home/jleland/Coding/Projects/flopter')
import flopter.core.ivdata as iv
import flopter.core.lputils as lp
import flopter.magnum.database as ut
import flopter.core.fitters as fts


def preprocess_sel(ds, sweep=slice(0, 997)):
    return ds.sel(sweep=sweep)


def preprocess_average(ds):
    ds = ds.reset_index('time', drop=True)
    ds_avg = ds.mean('sweep')
    ds_avg = ds_avg.assign({'d_current': ds.std('sweep')['current']})
    return ds_avg


# Create analysed dataset metadata
path_to_datasets = '/home/jleland/Data/Magnum/adc_files/'
# path_to_analysed_datasets = 'analysed_2'
path_to_analysed_datasets = 'analysed_3'
# path_to_analysed_datasets = 'analysed_3_downsampled'
os.chdir(path_to_datasets)

analysed_dataset_fns = glob.glob(path_to_analysed_datasets + '/*.nc')
analysed_dataset_fns.sort()

analysed_infos = []

for i, anal_ds in enumerate(analysed_dataset_fns):
    match = re.search("\/a{1}([0-9]{3})_([0-9]{3})_([0-9]{19})\.nc", anal_ds)
    shot_index = int(match.group(1))
    adc_index = int(match.group(2))
    shot_timestamp = int(match.group(3))
    shot_time = ut.human_datetime_str(int(match.group(3)))
    
    ds = xr.open_dataset(anal_ds)
    time_len = len(ds['time'])
    sweep_len = len(ds['sweep'])
    
    analysed_infos.append({
        'adc_index':adc_index, 
        'shot_number':shot_index, 
        'shot_timestamp':shot_timestamp, 
        'shot_time':shot_time, 
        'filename':anal_ds,
        'time_len': time_len,
        'sweep_len': sweep_len
    })

analysed_infos_df = pd.DataFrame(analysed_infos).set_index('adc_index')


# Loading the metadata .csv file into a pandas dataframe
metadata_filename = f'{path_to_analysed_datasets}_xr_metadata.csv'
# analysed_infos_df = pd.read_csv(metadata_filename).set_index('adc_index')


# Saving the metadata.csv file from the constructed pandas dataframe
analysed_infos_df.to_csv(metadata_filename)

magnum_probes = lp.MagnumProbes()

# Open metadata for adc_files
os.chdir(path_to_datasets)
meta_data_ds = xr.open_dataset('all_meta_data.nc')
print(meta_data_ds)


files_oi = analysed_infos_df['filename'].values
shot_numbers = analysed_infos_df['shot_number'].values
shot_numbers_da = xr.DataArray(shot_numbers, dims=['shot_number'], name='shot_number')
min_sweep_len = analysed_infos_df['sweep_len'].min()
print(f'Min_seeep_len: {min_sweep_len}')

tilt_da = meta_data_ds[['shot_tilt', 'adc_4_probe', 'adc_5_probe']]
tilt_da['shot_tilt'] = tilt_da.shot_tilt.round()


combined_ds = xr.open_mfdataset(files_oi, concat_dim=shot_numbers_da, preprocess=preprocess_average, parallel=True)

combined_ds = xr.merge([combined_ds, meta_data_ds.rename({'shot_time': 'sweep_time'})])\
    .assign_coords(tilt=meta_data_ds['shot_tilt'].round()).swap_dims({'shot_number': 'tilt'})

combined_ds.to_netcdf('all_combined_ds.nc')

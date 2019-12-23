import pathlib
import glob
import re
import xarray as xr
import pandas as pd
import flopter.magnum.database as ut


def create_analysed_ds_metadata(path):
    analysed_dataset_fns = glob.glob(path + '/*.nc')
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
            'adc_index': adc_index,
            'shot_number': shot_index,
            'shot_timestamp': shot_timestamp,
            'shot_time': shot_time,
            'filename': anal_ds,
            'time_len': time_len,
            'sweep_len': sweep_len
        })

    return pd.DataFrame(analysed_infos).set_index('adc_index')
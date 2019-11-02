import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import scipy.io as sio
import sys
import os
import glob
import pathlib as pth
sys.path.append('/home/jleland/Coding/Projects/flopter')
import flopter.spice.splopter as spl
import flopter.spice.tdata as td
import flopter.core.ivdata as iv
import flopter.core.fitters as fts

# In[3]:


lowdens_dir = pth.Path('/home/jleland/Spice/spice2/bin/data_local_m2/lowdens_anglescan')
os.chdir(lowdens_dir)

# In[4]:


scans_searchstr = '*'
angles_search_str = '/*[!.{yml, inp}]/backup*'

all_run_dirs = {}
scans = glob.glob(scans_searchstr)
for scan in scans:
    all_run_dirs[scan] = glob.glob(scan + angles_search_str)
    print(f'{scan}: {all_run_dirs[scan]}\n')

# In[5]:


scan = scans[1]
filename = lowdens_dir / all_run_dirs[scan][0] / 't-alpha_yz_-9.0.mat'
t_file_info = sio.whosmat(filename)
# for info in t_file_info:
#     print(info)

# t_file_labels, t_file_sizes, t_file_types = zip(*t_file_info)
# t_file_labels, t_file_sizes, t_file_types = zip(*sio.whosmat(filename))
t_file_sizes = {label: size for label, size, dtype in sio.whosmat(filename)}
t_file_labels = set(t_file_sizes.keys())
# print(t_file_labels)

desired_variables = td.DEFAULT_REDUCED_DATASET | {'ProbePot'}

for var in desired_variables:
    if var in t_file_labels:
        print(f'{var}: {t_file_sizes[var]}')
    else:
        print(f'WARNING: {var}')

prel_vars = ['t', 'dt']
prel_tfile = sio.loadmat(filename, variable_names=prel_vars)
print([prel_tfile[var] for var in prel_vars])
print(np.mean(prel_tfile['t']))
print(len(prel_tfile['t']))
synth_t = np.arange(1, len(prel_tfile['t']) + 1) * 2048.0 * prel_tfile['dt']
print(synth_t - np.squeeze(prel_tfile['t']))


# fig, axes = plt.subplots(2)
splopters = {}

for i in range(len(scans)):
    #     ax = axes[i]
    scan = scans[i]

    for angle_dir in all_run_dirs[scan]:
        print(f'Creating splopter for {scan}, {angle_dir}')
        splopter = spl.Splopter(lowdens_dir / angle_dir, reduce=desired_variables)
        splopter.prepare(denormaliser_fl=False, homogenise_fl=True, find_se_temp_fl=False)
        splopters[angle_dir] = splopter
#         ax.plot(splopter.iv_data['V'][:320], splopter.iv_data['I'][:320])

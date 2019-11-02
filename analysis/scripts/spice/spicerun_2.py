import flopter.spice.splopter as spl
import flopter.spice.tdata as td
import os
import pathlib as pth
import glob


lowdens_dir = pth.Path('/home/jleland/Data/external/spice/')
os.chdir(lowdens_dir)

skippable_scans = {
    'marconi/lowdens_anglescan/inpartest_as',
    'marconi/lowdens_anglescan/rearwall_tiltscan',
    'marconi/lowdens_anglescan/recessed_tiltscan'
}
skippable = {
    'marconi/lowdens_anglescan/angled_tiltscan/alpha_yz_-1.0',
    'marconi/lowdens_anglescan/angled_tiltscan/alpha_yz_-2.0',
    'marconi/lowdens_anglescan/angled_tiltscan/alpha_yz_-3.0',
    'marconi/lowdens_anglescan/inpartest_as_restart/alpha_yz_-1.0',
    'marconi/lowdens_anglescan/inpartest_as_restart/alpha_yz_-2.0',
    'marconi/lowdens_anglescan/inpartest_as_restart/alpha_yz_-3.0',
    'cumulus/lowdens_anglescan/angled_tiltscan_redo/alpha_yz_-1.0',
    'cumulus/lowdens_anglescan/angled_tiltscan_redo/alpha_yz_-1.0_restart',
    'cumulus/lowdens_anglescan/angled_tiltscan_redo/alpha_yz_-90.0',
    'cumulus/lowdens_anglescan/flush_tiltscan_redo/alpha_yz_-1.0',
    'cumulus/lowdens_anglescan/angled_tiltscan_ext/alpha_yz_-1.0',
    'marconi/lowdens_anglescan/inpartest_as/alpha_yz_-1.0',
    'marconi/lowdens_anglescan/inpartest_as/alpha_yz_-2.0',
    'marconi/lowdens_anglescan/inpartest_as/alpha_yz_-3.0',
    'marconi/lowdens_anglescan/inpartest_as/alpha_yz_-4.0',
    'marconi/lowdens_anglescan/inpartest_as/alpha_yz_-5.0',
    'marconi/lowdens_anglescan/inpartest_as/alpha_yz_-6.0',
    'marconi/lowdens_anglescan/inpartest_as/alpha_yz_-7.0',
    'marconi/lowdens_anglescan/inpartest_as/alpha_yz_-8.0',
    'marconi/lowdens_anglescan/inpartest_as/alpha_yz_-9.0',
    # 'cumulus/lowdens_anglescan/flush_tiltscan_ext/alpha_yz_-1.0',
    # 'cumulus/lowdens_anglescan/angled_tiltscan_ext/alpha_yz_-1.0'
}

scans_searchstr = '*/lowdens_anglescan/*'
# angles_search_str = '/*[!.{yml, inp}]/backup*'
angles_search_str = '/*[!.{yml, inp}]'

desired_variables = td.DEFAULT_REDUCED_DATASET | {'ProbePot'}

all_run_dirs = {}
scans = glob.glob(scans_searchstr)
scans = list(set(scans) - skippable_scans)
for scan in scans:
    if scan in skippable_scans:
        print(f'Skipping {scan}...\n\n')
        continue
    all_run_dirs[scan] = glob.glob(scan + angles_search_str)

    print(f'{scan}: {all_run_dirs[scan]}\n')

# For adding a new run to the list of splopters
angle_dir = all_run_dirs[scans[6]][0]
print(angle_dir)

splopters = {}

if angle_dir in splopters:
    print(f'Removing {angle_dir} from existing list of splopters')
    splopters.pop(angle_dir)
splopter = spl.Splopter(lowdens_dir / angle_dir, reduce=desired_variables)
splopter.prepare(denormaliser_fl=True, homogenise_fl=True, find_se_temp_fl=False)
splopter.denormalise()
splopters[angle_dir] = splopter

print('Created splopter')
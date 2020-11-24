import flopter.spice.splopter as spl
import pathlib as pth
import flopter.spice.tdata as td

spl_path = pth.Path("/home/jleland/data/external_big/spice/marconi/spice2/sheath_exp/angled_1/alpha_yz_-1.0")

non_standard_variables = {'t', 'ProbePot', 'npartproc', 'Nz', 'Nzmax', 'Ny', 'count', 'Npc', 'snumber', 'nproc'}
desired_variables = td.DEFAULT_REDUCED_DATASET | non_standard_variables

splopter = spl.Splopter(spl_path, reduce=desired_variables, ignore_tzero_fl=True)

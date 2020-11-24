import flopter.core.lputils as lpu
from flopter.spice.inputfactory import SimulationParameters, Simulation2DGeometry
import numpy as np
import matplotlib.pyplot as plt


flush_probe = lpu.AngledTipProbe(a=5e-3, b=5e-3, L=5e-3, g=1e-3, d_perp=0.0, theta_f=0., theta_p=np.radians(0.0))
angled_probe = lpu.AngledTipProbe(a=5e-3, b=5e-3, L=5e-3, g=1e-3, d_perp=0.0, theta_f=0., theta_p=np.radians(10.0))
recessed_probe = lpu.AngledTipProbe(a=5e-3, b=5e-3, L=5e-3, g=1e-3, d_perp=3e-4, theta_f=0., theta_p=np.radians(10.0))
r_probe = lpu.MagnumProbes().probe_r
s_probe = lpu.MagnumProbes().probe_s
l_probe = lpu.MagnumProbes().probe_l    # rearwall probe

simulation_parameters = SimulationParameters(1e18, 7.5, 1836, 1, 0.8, np.radians(8))
simulation_parameters.calculate_plasma_params(flush_probe, print_fl=True)

fig, axes = plt.subplots(2, sharex=True, sharey=True)

l_probe_sim = Simulation2DGeometry(l_probe, simulation_parameters, padded_fl=True, rearwall=True)
l_probe_sim.plot(axes[0], probe_colour='b', line_colour='k')
l_probe_sim.print_objects_sizes()

s_probe_sim = Simulation2DGeometry(s_probe, simulation_parameters, padded_fl=True, rearwall=True)
s_probe_sim.plot(axes[1], probe_colour='r', line_colour='k')
simulation_parameters.calculate_plasma_params(s_probe, print_fl=True)
s_probe_sim.print_objects_sizes()

plt.show()
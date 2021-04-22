import numpy as np
import matplotlib.pyplot as plt
import flopter.core.lputils as lpu
from flopter.spice.inputfactory import SimulationParameters, Simulation2DGeometry, Simulation3DGeometry


if __name__ == '__main__':
    flush_probe = lpu.AngledTipProbe(a=5e-3, b=5e-3, L=5e-3, g=1e-3, d_perp=0.0, theta_f=0., theta_p=np.radians(0.0))

    pad = False
    sim_params_lst = []

    fig, ax = plt.subplots(1, 4, figsize=[12, 6])
    print(f'temp \t sim y \t sim z \t fw y \t fw z \t pr y \t\t pr z \t rw y \t\t rw z \n')

    for i, temp in enumerate([5, 10, 20, 50]):
        simulation_parameters = SimulationParameters(1e18, temp, 1836, 1, 0.8, np.radians(3))

        simulation_parameters.calculate_plasma_params(flush_probe, print_fl=False)
        # simulation_parameters.plot_angular_dependence(flush_probe, ax=ax[:, i], label=temp)

        flush_probe_sim = Simulation2DGeometry(flush_probe, simulation_parameters, padded_fl=pad, rearwall=False,
                                               rearwall_shadow_fl=True)
        flush_probe_sim.plot(ax[i], probe_colour='b', line_colour='k')
        # flush_probe_sim.print_objects_sizes()

        probe_y_1 = flush_probe_sim.wall_width + flush_probe_sim.g_hat
        probe_y_2 = flush_probe_sim.wall_width + flush_probe_sim.g_hat + flush_probe_sim.L_hat

        rw_y_1 = flush_probe_sim.sim_width - flush_probe_sim.rearwall_w
        rw_y_2 = flush_probe_sim.sim_width

        print(f'{temp} \t'
              f'\t {flush_probe_sim.sim_width} \t {flush_probe_sim.sim_height} '
              f'\t {0},{flush_probe_sim.wall_width} \t {flush_probe_sim.wall_height} '
              f'\t {probe_y_1},{probe_y_2} \t {flush_probe_sim.leading_edge_height} '
              f'\t {rw_y_1},{rw_y_2} \t {flush_probe_sim.rearwall_h} \n')

    fig.tight_layout()
    plt.show()



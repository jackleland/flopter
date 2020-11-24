import numpy as np
import matplotlib.pyplot as plt
import flopter.core.lputils as lpu
from flopter.spice.inputfactory import SimulationParameters, Simulation2DGeometry, Simulation3DGeometry


if __name__ == '__main__':
    flush_probe = lpu.AngledTipProbe(a=5e-3, b=5e-3, L=5e-3, g=1e-3, d_perp=0.0, theta_f=0., theta_p=np.radians(0.0))
    angled_probe = lpu.AngledTipProbe(a=5e-3, b=5e-3, L=5e-3, g=1e-3, d_perp=0.0, theta_f=0., theta_p=np.radians(10.0))
    recessed_probe = lpu.AngledTipProbe(a=5e-3, b=5e-3, L=5e-3, g=1e-3, d_perp=3e-4, theta_f=0., theta_p=np.radians(10.0))
    r_probe = lpu.MagnumProbes().probe_r
    s_probe = lpu.MagnumProbes().probe_s
    l_probe = lpu.MagnumProbes().probe_l  # rearwall probe

    half_flush_probe = lpu.AngledTipProbe(a=2.5e-3, b=2.5e-3, L=2.5e-3, g=5e-4, d_perp=0.0, theta_f=0.,
                                          theta_p=np.radians(0.0))
    half_angled_probe = lpu.AngledTipProbe(a=2.5e-3, b=2.5e-3, L=2.5e-3, g=5e-4, d_perp=0.0, theta_f=0.,
                                           theta_p=np.radians(10.0))
    half_recessed_probe = lpu.AngledTipProbe(a=2.5e-3, b=2.5e-3, L=2.5e-3, g=5e-4, d_perp=1.5e-4, theta_f=0.,
                                             theta_p=np.radians(10.0))

    # simulation_parameters = SimulationParameters(1e17, 5, 1836, 1, 0.8, np.radians(1))
    # simulation_parameters = SimulationParameters(1e17, 5, 1836, 1, 0.8, np.radians(1))
    pad = True
    simulation_parameters = SimulationParameters(1e18, 5, 1836, 1, 1.0, np.radians(4.5))

    simulation_parameters.calculate_plasma_params(flush_probe, print_fl=True)
    simulation_parameters.calculate_plasma_params(half_flush_probe, print_fl=True)
    simulation_parameters.calculate_plasma_params(half_angled_probe, print_fl=True)
    simulation_parameters.calculate_plasma_params(half_recessed_probe, print_fl=True)

    fig, ax = plt.subplots(2, 3, sharex=True, figsize=[10, 6])
    simulation_parameters.plot_angular_dependence(flush_probe, ax=ax[:, 0], label='flush')
    simulation_parameters.plot_angular_dependence(half_flush_probe, ax=ax[:, 0], label='half-flush')
    simulation_parameters.plot_angular_dependence(angled_probe, ax=ax[:, 1], label='angled')
    simulation_parameters.plot_angular_dependence(half_angled_probe, ax=ax[:, 1], label='half-angled')
    simulation_parameters.plot_angular_dependence(recessed_probe, ax=ax[:, 2], label='recessed')
    simulation_parameters.plot_angular_dependence(half_recessed_probe, ax=ax[:, 2], label='half-recessed')
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[0, 2].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[1, 2].legend()

    # simulation_parameters = SimulationParameters(5e18, 5, 1836, 1, 0.8, np.radians(3))
    # simulation_parameters = SimulationParameters(1e19, 5, 1836, 1, 0.8, np.radians(1))
    # simulation_parameters = SimulationParameters(1e20, 5, 1836, 1, 0.8, np.radians(1))

    # fig, axes = plt.subplots()
    # angled_probe_sim = Simulation2DGeometry(angled_probe, simulation_parameters, padded_fl=False, rearwall=False)
    # angled_probe_sim.plot(axes, plot_arrows_fl=False)
    # angled_probe_sim.print_objects_sizes()

    # noinspection PyTypeChecker
    # fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    #
    # flush_probe_sim = Simulation2DGeometry(flush_probe, simulation_parameters, padded_fl=pad, rearwall=False)
    # flush_probe_sim.plot(ax[0][0])
    # print('Flush Probe: ')
    # flush_probe_sim.print_objects_sizes()
    #
    # angled_probe_sim = Simulation2DGeometry(angled_probe, simulation_parameters, padded_fl=pad, rearwall=False)
    # print('Angled Probe: ')
    # angled_probe_sim.plot(ax[0][1])
    # angled_probe_sim.print_objects_sizes()
    #
    # recessed_probe_sim = Simulation2DGeometry(recessed_probe, simulation_parameters, padded_fl=pad, rearwall=True)
    # recessed_probe_sim.plot(ax[1][0])
    # print('Recessed Probe: ')
    # recessed_probe_sim.print_objects_sizes()
    #
    # sprobe_probe_sim = Simulation2DGeometry(s_probe, simulation_parameters, padded_fl=pad, rearwall=True)
    # sprobe_probe_sim.plot(ax[1][1])
    # print('S Probe: ')
    # sprobe_probe_sim.print_objects_sizes()

    fig, ax = plt.subplots(2, 3, sharex=False, sharey=False, figsize=[10, 6])

    flush_probe_sim = Simulation2DGeometry(flush_probe, simulation_parameters, padded_fl=pad, rearwall=False,
                                           rearwall_shadow_fl=True)
    flush_probe_sim.plot(ax[0, 0], probe_colour='b', line_colour='k')
    flush_probe_sim.print_objects_sizes()

    half_flush_probe_sim = Simulation2DGeometry(half_flush_probe, simulation_parameters, padded_fl=pad, rearwall=False,
                                                rearwall_shadow_fl=True)
    half_flush_probe_sim.plot(ax[1, 0], probe_colour='b', line_colour='k')
    half_flush_probe_sim.print_objects_sizes()

    angle_probe_sim = Simulation2DGeometry(angled_probe, simulation_parameters, padded_fl=pad, rearwall=False,
                                           rearwall_shadow_fl=True)
    angle_probe_sim.plot(ax[0, 1], probe_colour='r', line_colour='k')
    angle_probe_sim.print_objects_sizes()

    half_angled_probe_sim = Simulation2DGeometry(half_angled_probe, simulation_parameters, padded_fl=pad, rearwall=False,
                                                 rearwall_shadow_fl=True)
    half_angled_probe_sim.plot(ax[1, 1], probe_colour='r', line_colour='k')
    half_angled_probe_sim.print_objects_sizes()

    recessed_probe_sim = Simulation2DGeometry(recessed_probe, simulation_parameters, padded_fl=pad, rearwall=False,
                                              rearwall_shadow_fl=True)
    recessed_probe_sim.plot(ax[0, 2], probe_colour='r', line_colour='k')
    recessed_probe_sim.print_objects_sizes()

    half_recessed_probe_sim = Simulation2DGeometry(half_recessed_probe, simulation_parameters, padded_fl=pad,
                                                   rearwall=False,
                                                   rearwall_shadow_fl=True)
    half_recessed_probe_sim.plot(ax[1, 2], probe_colour='r', line_colour='k')
    half_recessed_probe_sim.print_objects_sizes()

    fig.tight_layout()

    plt.show()
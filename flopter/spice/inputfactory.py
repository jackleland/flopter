import flopter.spice.inputparser as inp
import flopter.spice.normalise as nrm
import flopter.core.constants as c
import flopter.spice.utils as su
from flopter.core.decorators import printmethod, plotmethod
from flopter.core.lputils import LangmuirProbe, AngledTipProbe
import flopter.core.lputils as lpu
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_REARWALL_RECESSION = 1e-3   # mm


class InputParserFactory:
    def __init__(self, plasma_params, simulation_params, shapes, species):
        self.plasma_params = plasma_params
        self.simulation_params = simulation_params
        self.shapes = shapes
        self.species = species


@printmethod
def generate_2d_input(probe, density, temperature, mu, tau, b_field, alpha, ion_charge=1, padded_fl=True, rearwall=0.0,
                      wall_height=None, rearwall_shadow_fl=False, ax=None):
    """
    Function for the auto-generation of a 2D SPICE input file from desired input
    parameters. This returns an InputParser object which can then be either
    manipulated or printed directly to a file.


    :param probe:       [LangmuirProbe] Object describing the tip geometry of
                        the simulation.
    :param density:     Desired electron density (in m^-3).
    :param temperature: Desired electron temperature (in eV).
    :param mu:          Ion-to-electron mass ratio.
    :param tau:         Ion-to-electron temperature ratio.
    :param b_field:     Magnetic field strength (in T).
    :param alpha:       Magnetic field incidence angle to the tile (in radians).
    :param ion_charge:  Charge (atomic number) of the ion species.
    :param padded_fl:   [boolean] Boolean flag to control whether the simulation
                        is padded in the x-direction.
    :param rearwall:    Controls whether the rearwall of the probe-wall
                        configuration is recessed relative to the forewall to
                        shadow its leading edge. This can either be specified as
                        a boolean flag, in which case the standard MAST-U
                        recession of 1mm is used; or as a float, which is taken
                        to be the recession height in mm.
    :param wall_height:   Controls the minimum depth of the gap either side of the
                        probe, with the default being 1.5mm.
    :param ax:          [matplotlib axes object] For controlling plotting, if
                        given the function will attempt to plot a diagram of the
                        simulation window onto the given axis. Will not plot if
                        left None, which is the default behaviour.
    :return:            [InputParser] InputParser object with sections filled
                        out in accordance with SPICE-2 input file requirements.

    """
    if not isinstance(probe, LangmuirProbe):
        raise ValueError('Argument "probe" needs to be an instance of LangmuirProbe')

    angled_fl = probe.is_angled()

    # Calculate ion properties in SI units
    ion_mass = c.ELECTRON_MASS * mu
    ion_temp = temperature / tau

    print(f'Ion mass: {ion_mass}kg')
    print(f'Ion temperature: {ion_temp}eV\n\n')

    # Calculate plasma properties
    debye = su.get_lambda_d(density, temperature)
    larmor = su.get_larmor_r(temperature, b_field, ion_mass, ion_charge)
    ksi = su.get_ksi(density, temperature, ion_charge, ion_mass, b_field)
    probe_length = probe.get_2d_probe_length()
    exposed_d, exposed_h = probe.calc_exposed_lengths(alpha)

    print(f'Ksi: {ksi}')
    print(f'Lambda_D: {debye}')
    print(f'Larmor: {larmor * 1000:.3g} mm')
    print(f'Mass Ratio: {mu}\n')
    print(f'r_Li/L: \t{larmor / probe_length}')
    print(f'r_Li/lambda_D: \t{larmor / debye}')
    print(f'L/lambda_D: \t{probe_length / debye}\n\n')

    # Normalise all values for ease of simulation use
    pre_sheath = 5 * larmor

    larmor_hat = normalise(larmor, debye)
    L_hat = normalise(probe.get_2d_probe_length(), debye)
    g_hat = normalise(probe.g, debye)
    d_perp_hat = normalise(probe.d_perp, debye)
    psheath_hat = normalise(pre_sheath, debye)
    exp_hat = normalise(exposed_h, debye)

    wedge_h_hat = normalise(probe.get_2d_probe_height(), debye)
    if rearwall is True:
        rearwall = DEFAULT_REARWALL_RECESSION
    rearwall_recess = normalise(rearwall, debye)

    if wall_height is None:
        wall_height = max(exp_hat + d_perp_hat + (3 * larmor_hat), rearwall_recess + (2 * larmor_hat))
    else:
        wall_height = normalise(wall_height, debye)

    print(f'Probe length: \t{L_hat}')
    print(f'Gap length: \t{g_hat}')
    print(f'Minimum gap height: \t{wall_height}')
    print(f'Drop length: \t{d_perp_hat}')
    print(f'Pre-sheath size: \t{psheath_hat}')
    print(f'Exposed length: {exp_hat}')
    if angled_fl:
        print(f'Wedge height: \t{wedge_h_hat}')

    # Height calculations
    if wall_height < rearwall_recess + (2 * larmor_hat):
        raise ValueError('Wall height given does not allow for the rearwall to be recessed as requested.')
    if wall_height < exp_hat + d_perp_hat + (3 * larmor_hat):
        print('WARNING: Wall height given does not allow the leading edge shadowing to be properly captured. \n'
              'Setting wall height to it\'s minimum allowed value')
        wall_height = exp_hat + d_perp_hat + (3 * larmor_hat)

    leading_edge_height = wall_height - d_perp_hat
    rearwall_h = wall_height - rearwall_recess

    # Width calculations
    wall_width = max(4 * larmor_hat, wall_height)
    if rearwall_shadow_fl:
        larmor_smear = int(larmor_hat / np.sin(alpha))
        rearwall_top_shadow = int(((rearwall_recess + wedge_h_hat - d_perp_hat) / np.tan(alpha)) - g_hat) + larmor_smear
        rearwall_w = max(wall_width, rearwall_top_shadow)
    else:
        rearwall_w = wall_width

    min_height = psheath_hat + leading_edge_height + max(wedge_h_hat, d_perp_hat)
    min_width = L_hat + (2 * g_hat) + wall_width + rearwall_w
    sim_height = find_next_power(min_height)
    sim_width = find_next_power(min_width)

    spare_width = sim_width - min_width
    if padded_fl:
        padding_width = int(spare_width / 2)
        wall_width += padding_width
        rearwall_w += padding_width
    else:
        sim_height = min_height
        sim_width = min_width

    tip_points = None
    if angled_fl:
        tip_points = [
            [
                wall_width + g_hat,
                leading_edge_height
            ],
            [
                wall_width + g_hat + L_hat,
                leading_edge_height
            ],
            [
                wall_width + g_hat + L_hat,
                leading_edge_height + wedge_h_hat
            ]
        ]

    print('\nMinimum sim height: \t {}\t({})'.format(min_height, sim_height))
    print('Minimum sim width: \t {}\t({}) \n'.format(min_width, sim_width))

    if ax is not None:
        sim_objects = []

        sim_window = plt.Rectangle((0, 0), sim_width, sim_height, fc='w', ec='k', zorder=-2)
        sim_objects.append(sim_window)

        fore_wall = plt.Rectangle((0, 0), wall_width, wall_height, fc='gray', ec='k')
        sim_objects.append(fore_wall)

        rear_wall = plt.Rectangle((sim_width - rearwall_w, 0), rearwall_w, rearwall_h,
                                  fc='gray', ec='k')
        sim_objects.append(rear_wall)

        probe_body = plt.Rectangle((wall_width + g_hat, 0), L_hat,
                                   leading_edge_height, fc='b', ec='b', zorder=5)
        sim_objects.append(probe_body)

        if tip_points is not None:
            probe_tip = plt.Polygon(tip_points, fc='b', ec='b', zorder=5)
            sim_objects.append(probe_tip)

        # Draw the lower
        ww_arrow = wall_width
        ax.arrow(ww_arrow, wall_height, L_hat, -L_hat * np.tan(alpha), color='r', zorder=-1)
        ax.arrow(ww_arrow, wall_height, -ww_arrow, ww_arrow * np.tan(alpha), color='r', zorder=-1)

        trailing_edge_width = wall_width + g_hat + L_hat
        trailing_edge_height = leading_edge_height + wedge_h_hat
        te_to_wall = sim_width - trailing_edge_width
        ax.arrow(trailing_edge_width, trailing_edge_height, te_to_wall, -te_to_wall * np.tan(alpha),
                 color='r', zorder=-1)
        ax.arrow(trailing_edge_width, trailing_edge_height, -trailing_edge_width, trailing_edge_width * np.tan(alpha),
                 color='r', zorder=-1)

        for so in sim_objects:
            ax.add_patch(so)

        ax.axis('scaled')
        ax.autoscale()

    print('Sim width: {}'.format(sim_width))
    print('Sim height: {}\n'.format(sim_height))

    print('Forewall y: [{},{}]'.format(0, wall_width))
    print('Forewall x: [{},{}]\n'.format(0, wall_height))

    print('Rearwall y: [{},{}]'.format(sim_width - rearwall_w, sim_width))
    print('Rearwall x: [{},{}]\n'.format(0, rearwall_h))

    print('Probe body y: [{},{}]'.format(wall_width + g_hat, wall_width + g_hat + L_hat))
    print('Probe body x: [{},{}]\n'.format(0, leading_edge_height))

    if angled_fl:
        print('Probe tip y: [{},{},{}]'.format(tip_points[0][0], tip_points[1][0], tip_points[2][0]))
        print('Probe tip x: [{},{},{}]\n'.format(tip_points[0][1], tip_points[1][1], tip_points[2][1]))
    print('\n')


def normalise(value, debye_length):
    return int(value/debye_length)


def find_next_power(length):
    x = 1
    while x < length:
        x *= 2
    return x


@printmethod
def test_printing_function():
    print('Printing is working')


if __name__ == "__main__":
    flush_probe = lpu.AngledTipProbe(5e-3, 5e-3, 5e-3, 1e-3, 0., 0., np.radians(10.0))
    # flush_probe = lpu.MagnumProbes().probe_r

    fig, ax = plt.subplots()
    generate_2d_input(flush_probe, 1e18, 5, 1836, 1, 0.8, np.radians(1), padded_fl=False, ax=ax, print_fl=True,
                      rearwall=0.0, rearwall_shadow_fl=False)
    ax.autoscale()

    # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    # generate_2d_input(flush_probe, 1e18, 5, 1836, 1, 0.8, np.radians(2), padded_fl=False,
    #                   ax=ax[0], print_fl=True, rearwall=1e-3, rearwall_shadow_fl=True)
    # generate_2d_input(flush_probe, 1e18, 5, 1836, 1, 0.8, np.radians(2), padded_fl=True,
    #                   ax=ax[1], print_fl=True, rearwall=1e-3)

    # ax[0].autoscale()
    # ax[1].autoscale()

    plt.show()


import flopter.spice.inputparser as inp
import flopter.spice.normalise as nrm
import flopter.core.constants as c
import flopter.spice.utils as su
from flopter.core.decorators import printmethod, plotmethod
from flopter.core.lputils import LangmuirProbe, AngledTipProbe
import flopter.core.lputils as lpu
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_REARWALL_RECESSION = 1e-3   # in m (1 mm)


class InputParserFactory:
    def __init__(self, plasma_params, simulation_params, shapes, species):
        self.plasma_params = plasma_params
        self.simulation_params = simulation_params
        self.shapes = shapes
        self.species = species


class SimulationParameters:
    def __init__(self, density, temperature, mu, tau, b_field, alpha_pitch, alpha_yaw=0.0, ion_charge=1):
        """
        Class for holding a set of plasma and simulation values of interest.
        There are also methods for calculating information about the simulation
        from these parameters. This is intended to encapsulate the values needed
        to be passed to the 2 and 3d input file generators.


        :param density:     Desired electron density (in m^-3).
        :param temperature: Desired electron temperature (in eV).
        :param mu:          Ion-to-electron mass ratio.
        :param tau:         Ion-to-electron temperature ratio.
        :param b_field:     Magnetic field strength (in T).
        :param alpha_pitch: Pitch component of the magnetic field incidence
                            angle to the tile (in radians). Pitch is taken as
                            the pitch along the vector from foretile to
                            reartile.
        :param alpha_yaw:   Yaw component of the magnetic field incidence angle
                            to the tile (in radians). Yaw is taken as the yaw
                            along the vector from foretile to reartile.
        :param ion_charge:  Charge (atomic number) of the ion species.

        """
        self.density = density
        self.temperature = temperature
        self.mu = mu
        self.tau = tau
        self.b_field = b_field
        self.alpha_yz = alpha_pitch
        self.alpha_xy = alpha_yaw
        self.ion_charge = ion_charge

    def calculate_plasma_params(self, probe, print_fl=False):
        # Calculate ion properties in SI units
        ion_mass = c.ELECTRON_MASS * self.mu
        ion_temp = self.temperature / self.tau

        # Calculate plasma properties
        debye = su.get_lambda_d(self.density, self.temperature)
        larmor = su.get_larmor_r(self.temperature, self.b_field, ion_mass, self.ion_charge)
        exposed_d, exposed_h = probe.calc_exposed_lengths(self.alpha_yz)

        if print_fl:
            ksi = su.get_ksi(self.density, self.temperature, self.ion_charge, ion_mass, self.b_field)
            probe_length = probe.get_2d_probe_length()

            print(f'Ion mass: {ion_mass}kg')
            print(f'Ion temperature: {ion_temp}eV\n\n')
            print(f'Ksi: {ksi}')
            print(f'Lambda_D: {debye}')
            print(f'Larmor: {larmor * 1000:.3g} mm')
            print(f'Mass Ratio: {self.mu}\n')
            print(f'r_Li/L: \t{larmor / probe_length}')
            print(f'r_Li/lambda_D: \t{larmor / debye}')
            print(f'L/lambda_D: \t{probe_length / debye}\n\n')

        pre_sheath = 5 * larmor
        return debye, larmor, exposed_h, pre_sheath


class Simulation2DGeometry:
    """
    Class for storing geometric information about the wall, probe and rearwall
    in a SPICE-2 simulation of an FMP.
    """

    def __init__(self, probe, sim_params, padded_fl=True, rearwall=0.0, wall_height=None, rearwall_shadow_fl=False):
        """
        Initialisation auto-calculates the geometry of a 2D SPICE simulation
        from desired input parameters.


        :param probe:       [LangmuirProbe] Object describing the tip geometry
                            of the simulation.
        :param sim_params:  [SimulationParameters] Object describing the plasma
                            setup of the simulation.
        :param padded_fl:   [boolean] Boolean flag to control whether the
                            simulation domain is padded to the nearest power of
                            2.
        :param rearwall:    Controls whether the rearwall of the probe-wall
                            configuration is recessed relative to the forewall
                            to shadow its leading edge. This can either be
                            specified as a boolean flag, in which case the
                            standard MAST-U recession of 1mm is used; or as a
                            float, which is taken to be the recession height in
                            mm.
        :param wall_height: Controls the minimum depth of the gap either side of
                            the probe, with the default being 1.5mm.
        :param rearwall_shadow_fl:
                            [boolean] Boolean flag to control whether the
                            simulation extends the simulation domain to
                            encapsulate the entire shadow cast by the probe on
                            the rear-wall.
        :return:            [InputParser] InputParser object with sections filled
                            out in accordance with SPICE-2 input file requirements.

        """
        if not isinstance(sim_params, SimulationParameters):
            raise ValueError('Argument "sim_params" needs to be an instance of SimulationParameters')
        if not isinstance(probe, LangmuirProbe):
            raise ValueError('Argument "probe" needs to be an instance of LangmuirProbe')

        self.probe = probe
        self.sim_params = simulation_parameters
        self.angled_fl = probe.is_angled()
        self.debye, self.larmor, exposed_h, pre_sheath = sim_params.calculate_plasma_params(probe)

        # Normalise all values for ease of simulation use
        larmor_hat = self.normalise_length(self.larmor)
        psheath_hat = self.normalise_length(pre_sheath)

        # Normalised probe params
        self.L_hat = self.normalise_length(probe.get_2d_probe_length())
        self.g_hat = self.normalise_length(probe.g)
        self.d_perp_hat = self.normalise_length(probe.d_perp)
        exposed_h_hat = self.normalise_length(exposed_h)
        wedge_h_hat = self.normalise_length(probe.get_2d_probe_height())

        if rearwall is True:
            rearwall = DEFAULT_REARWALL_RECESSION
        rearwall_recess_hat = self.normalise_length(rearwall)

        if wall_height is None:
            self.wall_height = max(exposed_h_hat + self.d_perp_hat + (3 * larmor_hat),
                                   rearwall_recess_hat + (2 * larmor_hat))
        else:
            self.wall_height = self.normalise_length(wall_height)

        # Wall height calculations
        if self.wall_height < rearwall_recess_hat + (2 * larmor_hat):
            raise ValueError('Wall height given does not allow for the rearwall to be recessed as requested.')
        if self.wall_height < exposed_h_hat + self.d_perp_hat + (3 * larmor_hat):
            print('WARNING: Wall height given does not allow the leading edge shadowing to be properly captured. \n'
                  'Setting wall height to it\'s minimum allowed value')
            self.wall_height = exposed_h_hat + self.d_perp_hat + (3 * larmor_hat)

        self.leading_edge_height = self.wall_height - self.d_perp_hat
        self.rearwall_h = self.wall_height - rearwall_recess_hat

        # Width calculations
        self.wall_width = max(4 * larmor_hat, self.wall_height)
        if rearwall_shadow_fl:
            larmor_smear = int(larmor_hat / np.sin(sim_params.alpha_yz))
            rearwall_top_shadow = int(((rearwall_recess_hat + wedge_h_hat - self.d_perp_hat)
                                       / np.tan(sim_params.alpha_yz)) - self.g_hat) + larmor_smear
            self.rearwall_w = max(self.wall_width, rearwall_top_shadow)
        else:
            self.rearwall_w = self.wall_width

        self.min_height = psheath_hat + self.leading_edge_height + max(wedge_h_hat, self.d_perp_hat)
        self.min_width = self.L_hat + (2 * self.g_hat) + self.wall_width + self.rearwall_w
        self.max_height = find_next_power(self.min_height)
        self.max_width = find_next_power(self.min_width)

        spare_width = self.max_width - self.min_width
        if padded_fl:
            padding_width = int(spare_width / 2)
            self.wall_width += padding_width
            self.rearwall_w += padding_width
            self.sim_height = self.max_height
            self.sim_width = self.max_width
        else:
            self.sim_height = self.min_height
            self.sim_width = self.min_width

        if self.angled_fl:
            self.tip_points = [
                [
                    self.wall_width + self.g_hat,
                    self.leading_edge_height
                ],
                [
                    self.wall_width + self.g_hat + self.L_hat,
                    self.leading_edge_height
                ],
                [
                    self.wall_width + self.g_hat + self.L_hat,
                    self.leading_edge_height + wedge_h_hat
                ]
            ]
        else:
            self.tip_points = None

    def plot(self, ax=None, plot_arrows_fl=True, probe_colour='b', line_colour='r'):
        if ax is None:
            _, ax = plt.subplots()

        sim_objects = []

        sim_window = plt.Rectangle((0, 0), self.sim_width, self.sim_height, fc='w', ec='k', zorder=-2)
        sim_objects.append(sim_window)

        fore_wall = plt.Rectangle((0, 0), self.wall_width, self.wall_height, fc='gray', ec='k')
        sim_objects.append(fore_wall)

        rear_wall = plt.Rectangle((self.sim_width - self.rearwall_w, 0), self.rearwall_w, self.rearwall_h,
                                  fc='gray', ec='k')
        sim_objects.append(rear_wall)

        probe_body = plt.Rectangle((self.wall_width + self.g_hat, 0), self.L_hat,
                                   self.leading_edge_height, fc=probe_colour, ec=probe_colour, zorder=5)
        sim_objects.append(probe_body)

        if self.tip_points is not None:
            probe_tip = plt.Polygon(self.tip_points, fc=probe_colour, ec=probe_colour, zorder=5)
            sim_objects.append(probe_tip)

        # Draw the magnetic field lines for the leading and trailing edges
        if plot_arrows_fl:
            ww_arrow = self.wall_width
            ax.arrow(ww_arrow, self.wall_height, self.L_hat, -self.L_hat * np.tan(self.sim_params.alpha_yz),
                     color=line_colour, zorder=-1)
            ax.arrow(ww_arrow, self.wall_height, -ww_arrow, ww_arrow * np.tan(self.sim_params.alpha_yz),
                     color=line_colour, zorder=-1)

            trailing_edge_width = self.wall_width + self.g_hat + self.L_hat
            te_to_wall = self.sim_width - trailing_edge_width
            if self.tip_points:
                # Get max height along correct axis
                trailing_edge_height = np.max(self.tip_points, axis=0)[1]
            else:
                trailing_edge_height = self.leading_edge_height

            ax.arrow(trailing_edge_width, trailing_edge_height, te_to_wall,
                     -te_to_wall * np.tan(self.sim_params.alpha_yz), color=line_colour, zorder=-1)
            ax.arrow(trailing_edge_width, trailing_edge_height, -trailing_edge_width,
                     trailing_edge_width * np.tan(self.sim_params.alpha_yz), color=line_colour, zorder=-1)

        for so in sim_objects:
            ax.add_patch(so)
        ax.axis('scaled')
        ax.autoscale()

        return ax, sim_objects

    def normalise_length(self, value, scalar=None):
        if scalar is None:
            scalar = self.debye
        return int(value / scalar)

    def print_normalised_lengths(self):
        print(f'Probe length: \t{self.L_hat}')
        print(f'Gap length: \t{self.g_hat}')
        print(f'Minimum gap height: \t{self.wall_height}')
        print(f'Drop length: \t{self.d_perp_hat}')
        print(f'Pre-sheath size: \t{5}')
        if self.angled_fl:
            print(f'Wedge height: \t{self.normalise_length(self.probe.get_2d_probe_height())}')

    def print_objects_sizes(self):
        print(f'Sim width: {self.sim_width}')
        print(f'Sim height: {self.sim_height}\n')

        print(f'Forewall y: [{0},{self.wall_width}]')
        print(f'Forewall z: [{0},{self.wall_height}]\n')

        print(f'Rearwall y: [{self.sim_width - self.rearwall_w},{self.sim_width}]')
        print(f'Rearwall z: [{0},{self.rearwall_h}]\n')

        print(f'Probe body y: [{self.wall_width + self.g_hat},{self.wall_width + self.g_hat + self.L_hat}]')
        print(f'Probe body z: [{0},{self.leading_edge_height}]\n')

        if self.angled_fl:
            print(f'Probe tip y: [{self.tip_points[0][0]},{self.tip_points[1][0]},{self.tip_points[2][0]}]')
            print(f'Probe tip z: [{self.tip_points[0][1]},{self.tip_points[1][1]},{self.tip_points[2][1]}]\n')
        print('\n')

    def print(self):
        self.print_normalised_lengths()
        self.print_objects_sizes()
        return self

    def generate_input(self):
        pass


class Simulation3DGeometry(Simulation2DGeometry):
    """
    Extension of Simulation2DGeometry, this is a class for storing geometric
    information about the wall, probe and rearwall in a SPICE-3 simulation of an
    FMP.

    It is padded by default as SPICE-3 requires dimensions to be in powers of 2.
    """

    def __init__(self, probe, sim_params, rearwall=0.0, wall_height=None, rearwall_shadow_fl=False,
                 continuous_wall_fl=False):
        """
        Initialisation auto-calculates the geometry of a 2D SPICE simulation
        from desired input parameters. The simulation domain must be padded for
        a SPICE-3 simulation.


        :param probe:       [LangmuirProbe] Object describing the tip geometry
                            of the simulation.
        :param sim_params:  [SimulationParameters] Object describing the plasma
                            setup of the simulation.
        :param rearwall:    Controls whether the rearwall of the probe-wall
                            configuration is recessed relative to the forewall
                            to shadow its leading edge. This can either be
                            specified as a boolean flag, in which case the
                            standard MAST-U recession of 1mm is used; or as a
                            float, which is taken to be the recession height in
                            mm.
        :param wall_height: Controls the minimum depth of the gap either side of
                            the probe, with the default being 1.5mm.
        :param rearwall_shadow_fl:
                            [boolean] Boolean flag to control whether the
                            simulation extends the simulation domain to
                            encapsulate the entire shadow cast by the probe on
                            the rear-wall.
        :param continuous_wall_fl:
                            [boolean] Boolean flag to control whether the front
                            and rear walls are joined continuously. A gap of
                            g_hat will be inserted between the left- and
                            right-most points of the probe tip. If this is on
                            along with a specification of recession for the
                            rearwall, a step will be included at the halfway
                            point between the two walls.
        :return:            [InputParser] InputParser object with sections filled
                            out in accordance with SPICE-2 input file requirements.

        """
        super().__init__(probe, sim_params, padded_fl=True, rearwall=rearwall, wall_height=wall_height,
                         rearwall_shadow_fl=rearwall_shadow_fl)

        if continuous_wall_fl:
            raise NotImplementedError('This feature has not been implemented yet')

        self.probe_depth = self.normalise_length(self.probe.get_3d_probe_depth())

        self.min_depth = self.probe_depth + (2 * self.g_hat) + (2 * self.wall_width)
        self.max_depth = find_next_power(self.probe_depth)
        self.sim_depth = self.max_depth
        wedge_height = self.normalise_length(probe.get_2d_probe_height())

        self.tip_points_3d = [
            [
                self.wall_width + self.g_hat,
                self.leading_edge_height
            ],
            [
                self.wall_width + self.g_hat + self.L_hat,
                self.leading_edge_height
            ],
            [
                self.wall_width + self.g_hat + self.L_hat,
                self.leading_edge_height + wedge_height
            ]
        ]


def find_next_power(length):
    x = 1
    while x < length:
        x *= 2
    return x


if __name__ == "__main__":
    flush_probe = lpu.AngledTipProbe(a=5e-3, b=5e-3, L=5e-3, g=1e-3, d_perp=0.0, theta_f=0., theta_p=np.radians(0.0))
    angled_probe = lpu.AngledTipProbe(a=5e-3, b=5e-3, L=5e-3, g=1e-3, d_perp=0.0, theta_f=0., theta_p=np.radians(10.0))
    recessed_probe = lpu.AngledTipProbe(a=5e-3, b=5e-3, L=5e-3, g=1e-3, d_perp=3e-4, theta_f=0., theta_p=np.radians(10.0))
    r_probe = lpu.MagnumProbes().probe_r
    s_probe = lpu.MagnumProbes().probe_s
    l_probe = lpu.MagnumProbes().probe_l    # rearwall probe

    half_flush_probe = lpu.AngledTipProbe(a=2.5e-3, b=2.5e-3, L=5e-4, g=5e-4, d_perp=0.0, theta_f=0.,
                                          theta_p=np.radians(0.0))

    # simulation_parameters = SimulationParameters(1e17, 5, 1836, 1, 0.8, np.radians(1))
    # simulation_parameters = SimulationParameters(1e17, 5, 1836, 1, 0.8, np.radians(1))
    pad = True
    simulation_parameters = SimulationParameters(1e18, 7.5, 1836, 1, 0.8, np.radians(8))
    simulation_parameters.calculate_plasma_params(flush_probe, print_fl=True)

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

    fig, axes = plt.subplots(2, sharex=True, sharey=True)

    l_probe_sim = Simulation2DGeometry(l_probe, simulation_parameters, padded_fl=True, rearwall=True)
    l_probe_sim.plot(axes[0], probe_colour='b', line_colour='k')
    l_probe_sim.print_objects_sizes()

    s_probe_sim = Simulation2DGeometry(s_probe, simulation_parameters, padded_fl=True, rearwall=True)
    s_probe_sim.plot(axes[1], probe_colour='r', line_colour='k')
    simulation_parameters.calculate_plasma_params(s_probe, print_fl=True)
    s_probe_sim.print_objects_sizes()

    plt.show()


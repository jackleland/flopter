import numpy as np
import pathlib as pth
import flopter.spice.inputparser as inp
import flopter.core.constants as c


def get_ksi(density, temp, q, m_i, B):
    debye = get_lambda_d(density, temp)
    larmor = get_larmor_r(temp, B, m_i, q)
    return larmor / debye


def get_lambda_d(density, temp):
    return np.sqrt((c.EPSILON_0 * temp) / (c.ELEM_CHARGE * density))


def get_larmor_r(temp, B, m_i, q):
    return np.sqrt((c.ELEM_CHARGE * temp * m_i)) / (q * c.ELEM_CHARGE * B)


def get_mu(m_i, m_e):
    return m_i / m_e


def is_code_output_dir(directory):
    # TODO: (2019-09-12) -  This could be split into two functions, one which returns output_files, particle files
    #  etc. and another which runs the test. Then a single robust method can be used for this and get_ta_filenames()
    if not isinstance(directory, pth.Path) and isinstance(directory, str):
        directory = pth.Path(directory)
    if directory.is_dir():
        output_files = set(directory.glob('t-*.mat'))
        particle_files = set(directory.glob('t-*[0-9][0-9].mat'))
        a_file = set(directory.glob('[!t-]*[!.2d].mat'))
        t_file = output_files - particle_files

        # Directory is deemed to be a spice directory if it has a t-file and particle_files
        return len(t_file) == 1 and len(particle_files) > 1
    else:
        print(f'{directory} is not a directory')
        return False


def plot_2d_sim_window(input_file, ax=None, colour='red'):
    import matplotlib.pyplot as plt

    return_ax = False
    if ax is None:
        fig, ax = plt.subplots()
        return_ax = True

    inputparser = inp.InputParser()
    with open(input_file, 'r') as inpf:
        inputparser.read_file(inpf)

    shape_section_labels = {'rectangle', 'triangle', 'circle'}
    shape_sections = {}
    for shape in shape_section_labels:
        shape_section = [inputparser[section] for section in inputparser.sections() if shape in section]
        shape_sections[shape] = shape_section
    print(shape_sections)

    geometry = inputparser['geom']
    sim_objects = [plt.Rectangle((0, 0), int(geometry['Ly']), int(geometry['Lz']), fc='w', ec='gray', zorder=-2)]

    for shape_sec in shape_sections['rectangle']:
        rect_colour = colour if 'probe' in shape_sec['name'].lower() else 'gray'
        rect_width = int(shape_sec['yhigh']) - int(shape_sec['ylow'])
        rect_height = int(shape_sec['zhigh']) - int(shape_sec['zlow'])

        sim_objects.append(
            plt.Rectangle((int(shape_sec['ylow']), int(shape_sec['zlow'])), rect_width, rect_height, fc=rect_colour,
                          ec='k'))

    for shape_sec in shape_sections['triangle']:
        triangle_colour = colour if 'probe' in shape_sec['name'].lower() else 'gray'
        triangle_points = [[shape_sec['ya'], shape_sec['za']], [shape_sec['yb'], shape_sec['zb']],
                           [shape_sec['yc'], shape_sec['zc']]]
        triangle_points = [(int(tp[0]), int(tp[1])) for tp in triangle_points]

        sim_objects.append(plt.Polygon(triangle_points, fc=triangle_colour, ec='k'))

    for so in sim_objects:
        ax.add_patch(so)

    ax.axis('scaled')
    ax.set_xlabel(r'$z$ [$\lambda_D$]')
    ax.set_ylabel(r'$y$ [$\lambda_D$]')
    ax.autoscale()

    if return_ax:
        return ax

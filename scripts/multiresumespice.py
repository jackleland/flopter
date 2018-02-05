import os
import shutil
import subprocess
import glob

# constants
_inputs_path = 'inputs/'
_template_path = _inputs_path + 'disttemplate_ng_hg_sbm.inp'
_script_path = 'scripts/masala/masala_restart_template.sh'
_prep_run_directory = 'data/tests/distpreptest_halfnogap1'
_group_directory = 'data/script_test/'
_old_filename = 'distpreptest'


def create_input_file(new_inp_filename, inp_replacements):
    with open(_template_path, 'rt') as ftemplate:
        with open(new_inp_filename, 'wt') as finput:
            template_contents = ftemplate.read()
            for name, replacement in inp_replacements.items():
                template_contents.replace(name, replacement)
            finput.write(template_contents)


def edit_masala_restart(masala_replacements):
    with open(_script_path, 'r') as fmasala:
        masala = fmasala.read()
        for name, replacement in masala_replacements.items():
            masala.replace(name, replacement)
    with open(_script_path, 'w') as fmasala:
        fmasala.write(masala)


def create_run_directory(run_directory, new_filename):
    shutil.copytree(_prep_run_directory, run_directory)

    cwd = os.getcwd()
    print('Changing directory: {} -> {}'.format(cwd, run_directory))
    os.chdir(run_directory)

    subprocess.call('rename_files {} {}'.format(_old_filename, new_filename))
    for inp_f in glob.glob('*.inp'):
        os.remove(inp_f)
    os.remove('prepare.txt')
    os.remove('runscript.pbs')

    print('Changing directory: {} -> {}'.format(run_directory, cwd))
    os.chdir(cwd)


if __name__ == '__main__':
    spice2_directory = '~/Spice/spice2/bin/'
    os.chdir(spice2_directory)

    run_basename = 'distruns'
    run_suffix = '_halfnogap'
    input_suffix = '_ng_hg_sbmdr'

    zhigh = 375
    zlow = 85
    zdist = zhigh - zlow
    n_runs = 9
    diag_z_height = int(zdist/n_runs)

    for i in range(n_runs):
        diag_low = int((i*diag_z_height)+zlow)
        diag_high = int(((i+1)*32)+(zlow-1))
        print(diag_low, diag_high)

        new_input_filename = run_basename + str(i + 1) + input_suffix
        new_input_filepath = _inputs_path + new_input_filename
        replacements_i = {'[z_low]': diag_low, '[z_high]': diag_high}
        print(replacements_i)
        # create_input_file(new_input_filepath, replacements_i)

        run_name = run_basename + str(i + 1) + run_suffix
        replacements_m = {'[run_name]': run_name, '[input_file]': new_input_filename}
        print(replacements_m)
        # edit_masala_restart(replacements_m)

        run_directory = _group_directory + run_name
        new_filename_short = run_basename + str(i + 1)
        create_run_directory(run_directory, new_filename_short)

        # subprocess.call(_script_path + ' -r')






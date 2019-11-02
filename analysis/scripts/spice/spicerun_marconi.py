from flopter.spice import splopter as spl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    spl_t0_old = spl.Splopter('bin/data_local_m/', 'magnum/', 'fetail_T0_S1/', reduce_fl=True)
    spl_t0 = spl.Splopter('bin/data_local_m/', 'magnum/', 'fetail_T0_S/', reduce_fl=True)
    spl_t2 = spl.Splopter('bin/data_local_m/', 'magnum/', 'fetail_T2-5_S1/', reduce_fl=True)
    spl_t10 = spl.Splopter('bin/data_local_m/', 'magnum/', 'fetail_T10_S1/', reduce_fl=True)

    splopters = [spl_t0_old, spl_t0, spl_t2, spl_t10]
    for splopter in splopters:
        splopter.prepare(homogenise_fl=True, denormaliser_fl=False)
        plt.figure()
        splopter.plot_raw()

    plt.show()

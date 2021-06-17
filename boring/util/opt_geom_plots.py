import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.optimize import curve_fit
import more_itertools as mit
import numpy as np


def opt_geom_plots(filename, x):

    l = len(filename)
    nrg_list = np.zeros((l,x))
    opt_mass = np.zeros((l,x))
    opt_ratio = np.zeros((l,x))
    opt_spacing = np.zeros((l,x))
    opt_t_ratio = np.zeros((l,x))
    opt_res = np.zeros((l,x))
    density = np.zeros((l,x))
    opt_success = np.zeros((l,x))
    opt_side = np.zeros((l,x))
    opt_temp = np.zeros((l,x))
    cell_dens = np.zeros((l,x))
    pack_dens = np.zeros((l,x))

    slope = np.zeros(l)
    intercept = np.zeros(l)
    r_value = np.zeros(l)
    slope2 = np.zeros(l)
    intercept2 = np.zeros(l)
    r_value2 = np.zeros(l)
    s_label = 'Wh/kg cell'
    popt = np.zeros((l,4))
    

    fig, ax = plt.subplots(2,1)

    for z,f in enumerate(filename):
        df = pd.read_csv(f)
        opt_mass[z] = df['mass'].to_numpy()
        opt_temp[z] = df['temp'].to_numpy()
        opt_side[z] = df['side'].to_numpy()
        nrg_list[z] = df['energy'].to_numpy()
        opt_spacing[z] = df['spacing'].to_numpy()
        opt_ratio[z] = df['ratio'].to_numpy()
        opt_t_ratio[z] = df['t_ratio'].to_numpy()
        cell_dens[z] = df['cell_dens'].to_numpy()
        pack_dens[z] = df['pack_dens'].to_numpy()
        opt_success[z] = df['success'].to_numpy()


        # cell_dens = 225*((nrg_list*2/3)/12)
        # pack_dens = (16*nrg_list*2/3)/(.048*16 + opt_mass)
        overhead = 100*opt_mass/(opt_mass + .048*16*(18/nrg_list))

        # slope[z], intercept[z], r_value[z], p_value, std_err = stats.linregress(cell_dens[z],pack_dens[z])
        # slope2[z], intercept2[z], r_value2[z], p_value2, std_err2 = stats.linregress(cell_dens[z],overhead[z])

        def func(x, a, b, c, d):
            return a*x**3 + b*x**2 + c*x + d

        popt[z],pcov = curve_fit(func, cell_dens[z],pack_dens[z])
        print(popt)
        indices = [i for i in range(len(opt_success[z])) if opt_success[z][i] > 3]  # record indices where 32,41
        def find_ranges(iterable):  #     Yield range of consecutive numbers
            for group in mit.consecutive_groups(iterable):
                group = list(group)
                if len(group) == 1:
                    yield group[0]
                else:
                    yield group[0], group[-1]

        zones = list(find_ranges(indices))
        #print(zones)
        labels = ['Al. grid','PCM grid', 'Al. honeycomb']

        ax[1].plot(cell_dens[z],pack_dens[z], label=labels[z]) # *0.412+80
        ax[1].set_ylabel('Wh/kg pack')
        ax[1].set_xlabel('Wh/kg cell')

        ax[0].plot(cell_dens[z],overhead[z])
        ax[0].set_ylabel('overhead %')
        # ax[0].xaxis.set_visible(False)
        ax[1].legend(loc="lower right")

    plt.show()

if __name__ == '__main__':

    opt_geom_plots(['../metamodel/al_opt.csv','../metamodel/pcm_opt.csv','../metamodel/hny_opt2.csv'],30)


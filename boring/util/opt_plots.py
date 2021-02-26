import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
import more_itertools as mit
import numpy as np


def opt_plots(filename, x):

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
    s_label = 'Wh/kg cell'

    fig, ax = plt.subplots(3,3)

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

        slope[z], intercept[z], r_value[z], p_value, std_err = stats.linregress(cell_dens[z],pack_dens[z])

        indices = [i for i in range(len(opt_success[z])) if opt_success[z][i] > 3]  # record indices where 32,41
        def find_ranges(iterable):  #     Yield range of consecutive numbers
            for group in mit.consecutive_groups(iterable):
                group = list(group)
                if len(group) == 1:
                    yield group[0]
                else:
                    yield group[0], group[-1]

        zones = list(find_ranges(indices))
        print(zones)


        ax[0,2].plot(nrg_list[z],opt_temp[z])
        ax[0,2].set_ylabel('Neighbor Temp (K)')
        ax[1,2].plot(nrg_list[z],opt_side[z])
        ax[1,2].set_ylabel('side (mm)')
        ax[2,2].plot(cell_dens[z],pack_dens[z])
        ax[2,2].plot(cell_dens[z],cell_dens[z]*slope[z]+intercept[z]) # *0.412+80
        ax[2,2].set_ylabel('Wh/kg pack')
        s_label = s_label + '\n pack = {:.2f}*cell + {:2.1f} (R = {:.2f})'.format(slope[z],intercept[z],r_value[z])
        ax[2,2].set_xlabel(s_label)
        ax[0,0].plot(nrg_list[z],opt_mass[z])
        ax[0,0].set_ylabel('optimal mass (kg)')
        ax[1,0].plot(nrg_list[z],opt_spacing[z])
        ax[1,0].set_ylabel('optimal spacing')
        ax[2,0].plot(nrg_list[z],opt_ratio[z])
        ax[2,0].set_ylabel('optimal hole ratio')
        ax[2,0].set_xlabel('energy (kJ)')
        ax[0,1].plot(nrg_list[z],overhead[z])
        ax[0,1].set_ylabel('overhead %')
        ax[1,1].plot(nrg_list[z],opt_t_ratio[z])
        ax[1,1].set_ylabel('temp ratio')
        ax[2,1].plot(nrg_list[z],opt_success[z])
        for zone in zones:  # plot vertical red zones on all subplots (except the last plot)
            if type(zone) is tuple: #it's a range
                (minz,maxz) = zone
                [ax2.axvspan(nrg_list[z][minz], nrg_list[z][maxz], alpha=0.5, color='red') for ax2 in ax.flatten()[:-1]]
            elif (zone == x-1):
                [ax2.axvspan(nrg_list[z][zone-1], nrg_list[z][zone], alpha=0.5, color='red') for ax2 in ax.flatten()[:-1]]
            else: # it's just one point
                [ax2.axvspan(nrg_list[z][zone-1], nrg_list[z][zone+1], alpha=0.5, color='red') for ax2 in ax.flatten()[:-1]]
        # https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list

        ax[2,1].set_ylabel('opt_success')
        ax[2,1].set_xlabel('energy (kJ)')
    plt.show()

if __name__ == '__main__':
    
    opt_plots(['../metamodel/pcm_opt.csv', '../metamodel/al_opt.csv'],30)

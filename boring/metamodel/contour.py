from boring.metamodel.metaOptGroup import MetaOptimize

import openmdao.api as om
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import pandas as pd
import time
import more_itertools as mit

'''
Plot Design Space with Contour Maps

set gif = True to create a gif across a range of energies.

Author: Jeff Chin
'''


p = om.Problem()
model = p.model
nn = 1

p.model.add_subsystem(name='meta_optimize',
                      subsys=MetaOptimize(num_nodes=nn),
                      promotes_inputs=['*'],
                      promotes_outputs=['*'])

p.setup()

xx = 16
yy = 11

extra_bp = np.linspace(1.01,1.41,xx)
ratio_bp = np.linspace(0.1,0.9,yy)

MASS = np.zeros((xx,yy))
SIDE = np.zeros((xx,yy))
TEMP = np.zeros((xx,yy))
RATIO = np.zeros((xx,yy))

filenames = []

gif = True
nrg_list = np.linspace(16,32,9)

for nrg in nrg_list:
    filename = f'{nrg}.png'
    filenames.append(filename)

    for i,spacing in enumerate(extra_bp):
        for j,ratio in enumerate(ratio_bp): 

            p.set_val('extra', spacing)
            p.set_val('ratio', ratio)
            # p.set_val('resistance', 0.006)
            p.set_val('energy', nrg)

            p.run_driver()
            
            MASS[i][j] = p.get_val('mass')
            SIDE[i][j] = p.get_val('side')
            TEMP[i][j] = p.get_val('temp2_data')
            try:
                RATIO[i][j] = p.get_val('temp2_data')/p.get_val('temp3_data')
            except:
                RATIO[i][j] = 1.


    # print(TEMP)
    # print(RATIO)
    df = pd.read_csv('test_250.csv')
    spacing = df['spacing'].to_numpy()
    ratio = df['ratio'].to_numpy()
    error = df['success'].to_numpy()
    # print(error)

    side_range = np.linspace(10,120,20)
    temp_range = np.linspace(300,330,20)
    ratio_range = np.linspace(1.0,1.1,20)


    # # Plotting
    # fig, ax = plt.subplots(2,3)
    # fig.suptitle('Energy = {} kJ'.format(nrg), fontsize=16)
    # ax[0,1].contour(ratio_bp, extra_bp, MASS, 20, cmap='Greens');
    # ax[0,1].set_title('Mass')
    # ax[0,1].set_ylabel('Spacing')

    # # ax[1,1].contour(ratio_bp, extra_bp, SIDE, levels=side_range, cmap='Greys');
    # ax[1,1].contour(ratio_bp, extra_bp, MASS, 20, cmap='Greens');
    # ax[1,1].contour(ratio_bp, extra_bp, TEMP, levels=temp_range, cmap='Reds');
    # ax[1,1].contour(ratio_bp, extra_bp, RATIO, levels=ratio_range, cmap='Blues');
    # # ax[1,1].set_title('Combined')
    # ax[1,1].set_xlabel('Hole Ratio')
    # # ax[1,1].plot(ratio[error<3],spacing[error<3],"x")  # converged cases
    # # ax[1,1].plot(ratio[error>3],spacing[error>3],"o")  # failed cases
    # # ax[2,1].contour(ratio_bp, extra_bp, SIDE, levels=side_range, cmap='Greys');
    # # ax[2,1].set_title('side')
    # t = ax[1,2].contour(ratio_bp, extra_bp, TEMP, levels=temp_range, cmap='Reds');
    # #ax[1,2].clabel(t, inline=True, fontsize=10)
    # ax[1,2].set_title('Temp')
    # ax[1,0].contour(ratio_bp, extra_bp, RATIO, levels=ratio_range, cmap='Blues');
    # ax[1,0].set_title('Temp Ratio')
    # ax[1,0].set_ylabel('Spacing')

    # # plt.colorbar()
    # fig.delaxes(ax[0][0])
    # fig.delaxes(ax[0][2])
    # # fig.delaxes(ax[2][0])
    # # fig.delaxes(ax[2][2])


    plt.title('Energy = {} kJ'.format(nrg), fontsize=16)
    plt.ylabel('Spacing')

    plt.contour(ratio_bp, extra_bp, MASS, 20, cmap='Greens');
    plt.contour(ratio_bp, extra_bp, TEMP, levels=temp_range, cmap='Reds');
    plt.contour(ratio_bp, extra_bp, RATIO, levels=ratio_range, cmap='Blues');
    plt.xlabel('Hole Ratio')


    if gif:
        plt.savefig(filename)
        plt.close()


if gif:
    # build gif
    with imageio.get_writer('mygif_pres.gif', mode='I',duration=0.75) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
else:
    plt.show()
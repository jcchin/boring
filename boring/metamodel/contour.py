from boring.metamodel.metaOptGroup import MetaOptimize

import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import more_itertools as mit


p = om.Problem()
model = p.model
nn = 1

p.model.add_subsystem(name='meta_optimize',
                      subsys=MetaOptimize(num_nodes=nn),
                      promotes_inputs=['*'],
                      promotes_outputs=['*'])

p.setup()

extra_bp = np.linspace(1.,1.5,6)
ratio_bp = np.linspace(1.,2,5)

MASS = np.zeros((6,5))
DIAG = np.zeros((6,5))
TEMP = np.zeros((6,5))
RATIO = np.zeros((6,5))

for i,spacing in enumerate(extra_bp):
    for j,ratio in enumerate(ratio_bp): 

        p.set_val('extra', spacing)
        p.set_val('ratio', ratio)
        p.set_val('resistance', 0.009)
        p.set_val('energy', 28.13793103)

        p.run_driver()
        
        MASS[i][j] = p.get_val('mass')
        DIAG[i][j] = p.get_val('diagonal')
        TEMP[i][j] = p.get_val('temp2_data')
        RATIO[i][j] = p.get_val('temp2_data')/p.get_val('temp3_data')



df = pd.read_csv('opt_out.csv')
spacing = df['spacing'].to_numpy()
ratio = df['ratio'].to_numpy()


# Plotting
fig, ax = plt.subplots(3,3)
ax[0,1].contour(ratio_bp, extra_bp, MASS, 20, cmap='Greens');
ax[0,1].set_title('Mass')

ax[1,1].contour(ratio_bp, extra_bp, DIAG, 20, cmap='Greys');
ax[1,1].contour(ratio_bp, extra_bp, MASS, 20, cmap='Greens');
ax[1,1].contour(ratio_bp, extra_bp, TEMP, 20, cmap='Reds');
ax[1,1].contour(ratio_bp, extra_bp, RATIO, 20, cmap='Blues');
ax[1,1].plot(ratio,spacing,"x")
ax[2,1].contour(ratio_bp, extra_bp, DIAG, 20, cmap='Greys');
ax[2,1].set_title('Diagonal')
ax[1,2].contour(ratio_bp, extra_bp, TEMP, 20, cmap='Reds');
ax[1,2].set_title('Temp')
ax[1,0].contour(ratio_bp, extra_bp, RATIO, 20, cmap='Blues');
ax[1,0].set_title('Ratio')

# plt.colorbar()
fig.delaxes(ax[0][0])
fig.delaxes(ax[0][2])
fig.delaxes(ax[2][0])
fig.delaxes(ax[2][2])
plt.show()


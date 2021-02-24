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

xx = 16
yy = 11

extra_bp = np.linspace(1.,2.0,xx)
ratio_bp = np.linspace(0.25,1.0,yy)

MASS = np.zeros((xx,yy))
SIDE = np.zeros((xx,yy))
TEMP = np.zeros((xx,yy))
RATIO = np.zeros((xx,yy))

for i,spacing in enumerate(extra_bp):
    for j,ratio in enumerate(ratio_bp): 

        p.set_val('extra', spacing)
        p.set_val('ratio', ratio)
        # p.set_val('resistance', 0.006)
        p.set_val('energy', 32)

        p.run_driver()
        
        MASS[i][j] = p.get_val('mass')
        SIDE[i][j] = p.get_val('side')
        TEMP[i][j] = p.get_val('temp2_data')
        try:
            RATIO[i][j] = p.get_val('temp2_data')/p.get_val('temp3_data')
        except:
            RATIO[i][j] = 1.


# print(TEMP)
print(RATIO)
df = pd.read_csv('opt_out.csv')
spacing = df['spacing'].to_numpy()
ratio = df['ratio'].to_numpy()
error = df['success'].to_numpy()
print(error)

side_range = np.linspace(10,120,20)
temp_range = np.linspace(300,360,20)
ratio_range = np.linspace(1,1.2,20)


# Plotting
fig, ax = plt.subplots(3,3)
ax[0,1].contour(ratio_bp, extra_bp, MASS, 20, cmap='Greens');
ax[0,1].set_title('Mass')

ax[1,1].contour(ratio_bp, extra_bp, SIDE, levels=side_range, cmap='Greys');
ax[1,1].contour(ratio_bp, extra_bp, MASS, 20, cmap='Greens');
ax[1,1].contour(ratio_bp, extra_bp, TEMP, levels=temp_range, cmap='Reds');
ax[1,1].contour(ratio_bp, extra_bp, RATIO, levels=ratio_range, cmap='Blues');
ax[1,1].plot(ratio[error<3],spacing[error<3],"x")  # converged cases
ax[1,1].plot(ratio[error>3],spacing[error>3],"o")  # failed cases
ax[2,1].contour(ratio_bp, extra_bp, SIDE, levels=side_range, cmap='Greys');
ax[2,1].set_title('side')
t = ax[1,2].contour(ratio_bp, extra_bp, TEMP, levels=temp_range, cmap='Reds');
#ax[1,2].clabel(t, inline=True, fontsize=10)
ax[1,2].set_title('Temp')
ax[1,0].contour(ratio_bp, extra_bp, RATIO, levels=ratio_range, cmap='Blues');
ax[1,0].set_title('Ratio')

# plt.colorbar()
fig.delaxes(ax[0][0])
fig.delaxes(ax[0][2])
fig.delaxes(ax[2][0])
fig.delaxes(ax[2][2])
plt.show()


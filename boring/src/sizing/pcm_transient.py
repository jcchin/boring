"""
Run a heat pipe transient, where the final temperature of the condensor is specified,
dymos will determine the duration of the simulation

Authors: Sydney Schnulo, Jeff Chin
"""

import openmdao.api as om

import numpy as np
import dymos as dm

from boring.src.sizing.heatpipe_transient import get_hp_phase  # import the ODE
from boring.util.save_csv import save_csv

from boring.util.load_inputs import load_inputs

import time
start = time.time()

traj=dm.Trajectory()
p = om.Problem(model=traj)
p.driver = om.ScipyOptimizeDriver()
p.driver = om.pyOptSparseDriver(optimizer='SNOPT')

p.driver.declare_coloring()

num_cells = 2  # number of battery cells in the array

# construct the Dymos phase
# db=(min,max) duration of the simulation in seconds
# num_segments, minimum of 3, # of polynomials the simulation is fit to
# pcm = True, use Phase Change material pad (connected in thermal_network.py)
phase = get_hp_phase(num_cells=num_cells, db=(60, 60), num_segments=5, geom='round', pcm=True)
phase.add_timeseries_output('T_rate_pcm_1.cp_bulk', output_name='cp_bulk', units='kJ/kg/degK')

traj.add_phase('phase', phase)
# minimize final time, somewhat meaningless for a fixed time IVP,
# but a necessary dummy objective
phase.add_objective('time', loc='final', ref=1)   

p.model.linear_solver = om.DirectSolver()
p.setup(force_alloc_complex=True)

#p.model.list_inputs(prom_name=True)
p.model.list_outputs(prom_name=True)

# om.n2(p)
# quit()

p['phase.t_initial'] = 0.0
p['phase.t_duration'] = 60.

# set intial temperature profile for all cells
for cell in np.arange(num_cells):  
    p['phase.states:T_cell_{}'.format(cell)] = phase.interpolate(ys=[293.15, 350.0], nodes='state_input')

# Override cell 2 to be initialized hot
p['phase.states:T_cell_0'] = phase.interpolate(ys=[373.15, 333.15], nodes='state_input')

p.run_driver()
om.view_connections(p)

# Plot temperature results
import matplotlib.pyplot as plt

time_opt = p.get_val('phase.timeseries.time', units='s')

fig, ax = plt.subplots(2,1, sharex=True)


for j in np.arange(num_cells):

    T_cell = p.get_val('phase.timeseries.states:T_cell_{}'.format(j), units='K')
    Cp_pcm = p.get_val('phase.timeseries.cp_bulk')

    ax[1].plot(time_opt, T_cell, label='cell {}'.format(j))
    ax[0].plot(time_opt, Cp_pcm, label='cp {}'.format(j))

ax[1].set_xlabel('time, s')
ax[1].set_ylabel('T_cell, K')
#ax[1].legend()
ax[1].axhline(y=333, color='r', linestyle='-')
ax[1].axhline(y=338, color='r', linestyle='-')
print("--- elapsed time: %s seconds ---" % (time.time() - start))

plt.show()

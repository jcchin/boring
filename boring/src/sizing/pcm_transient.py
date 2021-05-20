"""
Run a heat pipe transient, where the final temperature of the condensor is specified,
dymos will determine the duration of the simulation

Authors: Jeff Chin, Sydney Schnulo
"""

import openmdao.api as om

import numpy as np
import dymos as dm

from boring.src.sizing.heatpipe_transient import get_hp_phase  # import the ODE
from boring.util.save_csv import save_csv

from boring.util.load_inputs import load_inputs

from pathlib import Path
import time
start = time.time()

case_name = 'cases.sql'

traj=dm.Trajectory()
p = om.Problem(model=traj)
p.driver = om.ScipyOptimizeDriver()
p.driver = om.pyOptSparseDriver(optimizer='SNOPT')
p.driver.declare_coloring()

# case recording
recorder = om.SqliteRecorder(case_name)
p.recording_options['includes'] = ['*'] # save everything when prob.record_iteration is called
p.recording_options['record_constraints'] = True 
p.recording_options['record_desvars'] = True 
p.recording_options['record_objectives'] = True 
p.add_recorder(recorder)

num_cells = 2  # number of battery cells in the array

# construct the Dymos phase
# db=(min,max) duration of the simulation in seconds
# num_segments, minimum of 3, # of polynomials the simulation is fit to
# pcm = True, use Phase Change material pad (connected in thermal_network.py)
phase = get_hp_phase(num_cells=num_cells, db=(300, 300), num_segments=20, transcription_order=3, geom='flat', pcm=True, solve_segments=False)
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
p['phase.t_duration'] = 300.

# set intial temperature profile for all cells
for cell in np.arange(num_cells):  
    p['phase.states:T_cell_{}'.format(cell)] = phase.interpolate(ys=[293.0, 350.0], nodes='state_input')

# Override cell 2 to be initialized hot
p['phase.states:T_cell_0'] = phase.interpolate(ys=[373, 330], nodes='state_input')

p.run_driver()
# om.view_connections(p)


# move cases.sql up and over to the output folder
pth = Path('./cases.sql').absolute()
pth.rename(pth.parents[2]/'output'/pth.name)


# Plot temperature results
import matplotlib.pyplot as plt

time_opt = p.get_val('phase.timeseries.time', units='s')

fig, ax = plt.subplots(3,1, sharex=True)

T_cell_0 = p.get_val('phase.timeseries.states:T_cell_0', units='K')
T_cell_1 = p.get_val('phase.timeseries.states:T_cell_1', units='K')
Cp_pcm = p.get_val('phase.timeseries.cp_bulk')

ax[1].plot(time_opt, T_cell_0, label='cell {}')
ax[2].plot(time_opt, T_cell_1, label='cell {}')
ax[0].plot(time_opt, Cp_pcm, label='cp {}')

ax[2].set_xlabel('time, s')
ax[1].set_ylabel('T Runaway, K')
ax[2].set_ylabel('T Neighbor, K')
ax[0].set_ylabel('Bulk c_p, kJ/kg/K')
#ax[1].legend()
ax[2].axhline(y=333, color='r', linestyle='-')
ax[2].axhline(y=338, color='r', linestyle='-')
print("--- elapsed time: %s seconds ---" % (time.time() - start))

plt.show()


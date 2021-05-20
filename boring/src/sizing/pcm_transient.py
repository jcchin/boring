"""
Run a heat pipe transient, where the final temperature of the condensor is specified,
dymos will determine the duration of the simulation

Authors: Jeff Chin, Sydney Schnulo
"""

import openmdao.api as om

import numpy as np
import dymos as dm

from boring.src.sizing.heatpipe_transient import get_hp_phase  # import the ODE
from boring.src.sizing.geometry.hp_geom import HPgeom
from boring.util.save_csv import save_csv

from boring.util.load_inputs import load_inputs

from pathlib import Path
import time
start = time.time()

case_name = 'cases.sql'

p = om.Problem()
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

nn = 5
num_cells = 2  # number of battery cells in the array

# p.model.add_subsystem(name = 'size',
#               subsys = HPgeom(num_nodes=nn, geom='round'),
#               promotes_inputs=['LW:L_flux', 'LW:L_adiabatic', 'XS:t_w', 'XS:t_wk', 'XS:D_v'],
#               promotes_outputs=['XS:D_od','XS:r_i', 'XS:A_w', 'XS:A_wk', 'LW:A_flux', 'LW:A_inter', 'LW:L_eff']) 
p.model.add_subsystem(name = 'size',
              subsys = HPgeom(num_nodes=nn, geom='flat'),
              promotes_inputs=['LW:L_flux', 'LW:L_adiabatic', 'XS:t_w', 'XS:t_wk', 'XS:W_v','XS:H_v'],
              promotes_outputs=['XS:W_hp','XS:H_hp', 'XS:A_w', 'XS:A_wk', 'LW:A_flux', 'LW:A_inter', 'LW:L_eff','XS:r_h']) 


# construct the Dymos phase
# db=(min,max) duration of the simulation in seconds
# num_segments, minimum of 3, # of polynomials the simulation is fit to
# pcm = True, use Phase Change material pad (connected in thermal_network.py)
traj=dm.Trajectory()
# phase = get_hp_phase(num_cells=num_cells, db=(10, 10), num_segments=1, geom='round', pcm=True)
phase = get_hp_phase(num_cells=num_cells, db=(10, 10), num_segments=1, geom='flat', pcm=True)
traj.add_phase('phase', phase)
p.model.add_subsystem(name = 'traj', subsys=traj,
                      promotes_inputs=['*'],
                      promotes_outputs=['*'])

phase.add_timeseries_output('T_rate_pcm_1.cp_bulk', output_name='cp_bulk', units='kJ/kg/degK')

traj.add_phase('phase', phase)

# connect sizing outputs to dymos parameters
# phase.add_parameter('XS:D_od', targets='XS:D_od', units='mm')  # round
# phase.add_parameter('XS:r_i', targets='XS:r_i', units='mm')  # round
# phase.add_parameter('XS:W_hp', targets='XS:W_hp', units='mm')  # flat
# phase.add_parameter('XS:H_hp', targets='XS:H_hp', units='mm')  # flat
phase.add_parameter('XS:A_w', targets='XS:A_w', units='mm**2')
phase.add_parameter('XS:A_wk', targets='XS:A_wk', units='mm**2')
# phase.add_parameter('LW:A_flux', targets='LW:A_flux', units='m**2')
phase.add_parameter('LW:A_inter', targets='LW:A_inter', units='mm**2')
phase.add_parameter('LW:L_eff', targets='LW:L_eff', units='mm')
phase.add_parameter('XS:r_h', targets='XS:r_h', units='m')

# p.model.connect('XS:D_od', 'phase.parameters:XS:D_od')  # round
# p.model.connect('XS:r_i', 'phase.parameters:XS:r_i')  # round
# p.model.connect('XS:W_hp', 'phase.parameters:XS:W_hp')  # flat
# p.model.connect('XS:H_hp', 'phase.parameters:XS:H_hp')  # flat
p.model.connect('XS:A_w', 'phase.parameters:XS:A_w')
p.model.connect('XS:A_wk', 'phase.parameters:XS:A_wk')
# p.model.connect('LW:A_flux', 'phase.parameters:LW:A_flux')
p.model.connect('LW:A_inter', 'phase.parameters:LW:A_inter')
p.model.connect('LW:L_eff', 'phase.parameters:LW:L_eff')
p.model.connect('XS:r_h', 'phase.parameters:XS:r_h')


# constants = p.model.add_subsystem('constants', om.IndepVarComp(), promotes_outputs=['*'])
# constants.add_output('c_p', val=1.5, units='kJ/(kg*K)', desc='cell specific heat')
# constants.add_output('mass', val=0.0316, units='kg', desc='cell mass')

# # connect battery properties
# phase.add_parameter('c_p', targets='c_p', units='kJ/(kg*K)')
# p.model.connect('c_p', 'phase.parameters:c_p')
# phase.add_parameter('mass', targets='mass', units='kg')
# p.model.connect('mass', 'phase.parameters:mass')



# minimize final time, somewhat meaningless for a fixed time IVP,
# but a necessary dummy objective
phase.add_objective('time', loc='final', ref=1)   

p.model.linear_solver = om.DirectSolver()
p.setup(force_alloc_complex=True)
p.final_setup()


p['phase.t_initial'] = 0.0
p['phase.t_duration'] = 10.

# set intial temperature profile for all cells
for cell in np.arange(num_cells):  
    p['phase.states:T_cell_{}'.format(cell)] = phase.interpolate(ys=[293.15, 350.0], nodes='state_input')

# Override cell 2 to be initialized hot
p['phase.states:T_cell_0'] = phase.interpolate(ys=[500, 330], nodes='state_input')

p.run_driver()
p.model.list_inputs(prom_name=True)
p.model.list_outputs(prom_name=True)
om.view_connections(p)
#om.n2(p)

# move cases.sql up and over to the output folder
pth = Path('./cases.sql').absolute()
pth.rename(pth.parents[2]/'output'/pth.name)


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

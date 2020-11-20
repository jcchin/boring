import openmdao.api as om

import numpy as np
import dymos as dm

from boring.src.sizing.heatpipe_run import HeatPipeRun 

p = om.Problem(model=om.Group())
model = p.model
nn = 1
p.driver = om.ScipyOptimizeDriver()
p.driver = om.pyOptSparseDriver(optimizer='SNOPT')

p.driver.declare_coloring()

traj = p.model.add_subsystem('traj', dm.Trajectory())

phase = traj.add_phase('phase',
                       dm.Phase(ode_class=HeatPipeRun,
                                transcription=dm.GaussLobatto(num_segments=5, order=3, compressed=False)))

phase.set_time_options(fix_initial=True, fix_duration=False, duration_bounds=(1., 800.))

phase.add_state('T_cond', rate_source='Tdot_cond', targets='cond.Rex.T_in', units='K',# ref=333.15, defect_ref=333.15,
                    fix_initial=True, fix_final=False, solve_segments=False)

phase.add_parameter('T_evap', targets='evap.Rex.T_in', units='K',
                    dynamic=True, opt=False)

phase.add_boundary_constraint('T_cond', loc='final', equals=370)

phase.add_objective('time', loc='final', ref=1)

p.model.linear_solver = om.DirectSolver()
p.setup(force_alloc_complex=True)

p['traj.phase.t_initial'] = 0.0
p['traj.phase.t_duration'] = 195.
p['traj.phase.states:T_cond'] = phase.interpolate(ys=[293.15, 333.15], nodes='state_input')
p['traj.phase.parameters:T_evap'] = 373

p.run_model()

opt = p.run_driver()
sim = traj.simulate(times_per_seg=10)

import matplotlib.pyplot as plt 

time = sim.get_val('traj.phase.timeseries.time', units='s')
time_opt = p.get_val('traj.phase.timeseries.time', units='s')
T_cond = sim.get_val('traj.phase.timeseries.states:T_cond', units='K')
T_cond_opt = p.get_val('traj.phase.timeseries.states:T_cond', units='K')

plt.plot(time_opt, T_cond_opt, 'o')
plt.plot(time, T_cond)

plt.xlabel('time, s')
plt.ylabel('T_cond, K')

plt.show()



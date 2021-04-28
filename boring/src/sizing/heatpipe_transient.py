"""
Create a dymos phase that instantiates a heatpipe group,
then adds and conencts states for "n" cells/pcm pads.


Authors: Sydney Schnulo, Jeff Chin
"""

import openmdao.api as om

import numpy as np
import dymos as dm

from boring.src.sizing.heatpipe_group import HeatPipeGroup  # import the ODE
from boring.util.save_csv import save_csv

from boring.util.load_inputs import load_inputs

import matplotlib.pyplot as plt

# compressed = False requires more variables, but can provide more robust convergence
# solve_segments = <False, forward, backard> method for converging states

def get_hp_phase(transcription='gauss-radau', num_segments=5,
                 transcription_order=7, compressed=False,
                 solve_segments='forward', num_cells=3, db=(1, 300),
                 pcm=False, geom='round'):

    phase = dm.Phase(ode_class=HeatPipeGroup,
                                    ode_init_kwargs= {'num_cells': num_cells,
                                                      'pcm_bool': pcm,
                                                      'geom': geom},
                                    transcription=dm.GaussLobatto(num_segments=num_segments, order=transcription_order,
                                                                  compressed=compressed))

    phase.set_time_options(fix_initial=True, fix_duration=False, duration_bounds=db)

    for i in np.arange(num_cells):

        if pcm:
            # create cell temp states, connect the pcm temp state to the heatpipe
            # phase.add_state('T_cell_{}'.format(i), rate_source='T_rate_cell_{}.Tdot'.format(i), targets='cell_{}.Rex.T_in'.format(i), units='K',
            #                 lower=250, upper=400, fix_initial=True, fix_final=False, solve_segments=solve_segments)
            # create pcm temp states, connect T_in to external resistance Rex
            phase.add_state('T_cell_{}'.format(i), rate_source='T_rate_pcm_{}.Tdot'.format(i), targets='cell_{}.Rex.T_in'.format(i), units='K',
                            lower=250, upper=400, fix_initial=True, fix_final=False, solve_segments=solve_segments)

            phase.add_parameter('cell_{}.LW:L_flux'.format(i), val=0.02, units='m', targets='cell_{}.LW:L_flux'.format(i), include_timeseries=False, opt=False)
            phase.add_parameter('cell_{}.R'.format(i), val=0.0001, units='K/W', targets='cell_{}.Rex.R'.format(i), include_timeseries=False, opt=False)
        
        else:
            # since there is no pcm, connect cell T_in directly to external resistance Rex
            phase.add_state('T_cell_{}'.format(i), rate_source='T_rate_cell_{}.Tdot'.format(i), targets='cell_{}.Rex.T_in'.format(i), units='K',
                            lower=250, upper=400, fix_initial=True, fix_final=False, solve_segments=solve_segments)

            phase.add_parameter('cell_{}.LW:L_flux'.format(i), val=0.02, units='m', targets='cell_{}.LW:L_flux'.format(i), include_timeseries=False, opt=False)
            phase.add_parameter('cell_{}.R'.format(i), val=0.0001, units='K/W', targets='cell_{}.Rex.R'.format(i), include_timeseries=False, opt=False)

    return phase

if __name__ == '__main__':
    import time

    start = time.time()

    traj=dm.Trajectory()
    p = om.Problem(model=traj)
    p.driver = om.ScipyOptimizeDriver()
    p.driver = om.pyOptSparseDriver(optimizer='SNOPT')

    p.driver.declare_coloring()

    num_cells = [5, 10, 15]
    color = ('C0', 'C1', 'C2', 'C3', 'C4')

    i = 0

    cells = 2


    phase = get_hp_phase(num_cells=cells, db=(10, 10), num_segments=10, solve_segments=False, geom='round')

    traj.add_phase('phase', phase)

    phase.add_objective('time', loc='final', ref=1)

    p.model.options['assembled_jac_type'] = 'csc'
    p.model.linear_solver = om.DirectSolver(assemble_jac=True)
    p.setup()

    p['phase.t_initial'] = 0.0
    p['phase.t_duration'] = 10.

    for cell in np.arange(cells):
        p['phase.states:T_cell_{}'.format(cell)] = phase.interpolate(ys=[293.15, 333.15], nodes='state_input')

    p['phase.states:T_cell_1'] = phase.interpolate(ys=[373.15, 333.15], nodes='state_input')

    p.run_driver()
    p.model.list_inputs(prom_name=True)
    p.model.list_outputs(prom_name=True)
    time_opt = p.get_val('phase.timeseries.time', units='s')

    for j in np.arange(cells):

        T_cell = p.get_val('phase.timeseries.states:T_cell_{}'.format(j), units='K')

        plt.plot(time_opt, T_cell, label='cell {}'.format(j))


    plt.xlabel('time, s')
    plt.ylabel('T_cell, K')
    plt.legend()

    plt.show()





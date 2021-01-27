"""
Run a heat pipe transient, where the final temperature of the condensor is specified,
dymos will determine the duration of the simulation

Authors: Sydney Schnulo, Jeff Chin
"""

import openmdao.api as om

import numpy as np
import dymos as dm

from boring.src.sizing.heatpipe_run import HeatPipeGroup  # import the ODE
from boring.util.save_csv import save_csv

from boring.util.load_inputs import load_inputs


def get_hp_phase(transcription='gauss-lobatto', num_segments=5,
                 transcription_order=3, compressed=False,
                 solve_segments=False, num_cells=3, db=(1, 300),
                 pcm=False):

    phase = traj.add_phase('phase',
                           dm.Phase(ode_class=HeatPipeGroup,
                                    ode_init_kwargs= {'num_cells': num_cells,
                                                      'pcm_bool': pcm},
                                    transcription=dm.GaussLobatto(num_segments=num_segments, order=transcription_order,
                                                                  compressed=compressed)))

    phase.set_time_options(fix_initial=True, fix_duration=False, duration_bounds=db)

    for i in np.arange(num_cells):

        phase.add_state('T_cell_{}'.format(i), rate_source='T_rate_cell_{}.Tdot'.format(i), targets='cell_{}.Rex.T_in'.format(i), units='K',
                        lower=250, upper=400,
                        # ref=333.15, defect_ref=333.15,
                        fix_initial=True, fix_final=False, solve_segments=solve_segments)

    return phase

if __name__ == '__main__':
    import time

    start = time.time()

    p = om.Problem(model=om.Group())
    model = p.model
    p.driver = om.ScipyOptimizeDriver()
    p.driver = om.pyOptSparseDriver(optimizer='SNOPT')

    p.driver.declare_coloring()

    num_cells = 5

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    phase = get_hp_phase(num_cells=num_cells, db=(100, 100), num_segments=10)

    phase.add_objective('time', loc='final', ref=1)

    p.model.linear_solver = om.DirectSolver()
    p.setup(force_alloc_complex=True)

    p['traj.phase.t_initial'] = 0.0
    p['traj.phase.t_duration'] = 100.

    for cell in np.arange(num_cells):
        p['traj.phase.states:T_cell_{}'.format(cell)] = phase.interpolate(ys=[293.15, 333.15], nodes='state_input')

    p['traj.phase.states:T_cell_2'] = phase.interpolate(ys=[373.15, 333.15], nodes='state_input')

    p.run_driver()

    import matplotlib.pyplot as plt

    time_opt = p.get_val('traj.phase.timeseries.time', units='s')

    for j in np.arange(num_cells):

        T_cell = p.get_val('traj.phase.timeseries.states:T_cell_{}'.format(j), units='K')

        plt.plot(time_opt, T_cell, label='cell {}'.format(j))

    plt.xlabel('time, s')
    plt.ylabel('T_cell, K')
    plt.legend()

    plt.show()





import openmdao.api as om

import numpy as np
import dymos as dm

# from boring.src.sizing.mass.mass import packMass
# from boring.src.sizing.pack_design import packSize, pcmSize, SizingGroup
from boring.src.sizing.geometry.insulation_props import tempODE
import openmdao.api as om

"""
Author(s): Jeff Chin
"""


def get_cond_phase():
    """
    Use Dymos to minimize the thickness of the insulator,
    while still maintaining a temperature below 100degC after a 45 second transient
    """


    phase = dm.Phase(ode_class=tempODE,
                    transcription=dm.GaussLobatto(num_segments=20, order=3, 
                                                  solve_segments='forward', 
                                                  compressed=False))

    phase.set_time_options(fix_initial=True, fix_duration=True)

    phase.add_state('T', rate_source='Tdot', units='K', ref=333.15, defect_ref=333.15,
                    fix_initial=True, fix_final=False, solve_segments=False)

    phase.add_boundary_constraint('T', loc='final', units='K', upper=333.15, lower=293.15, shape=(1,))
    phase.add_parameter('d', opt=True, lower=0.001, upper=0.5, val=0.001, units='m', ref0=0, ref=1)
    
    return phase


if __name__ == "__main__":

    p = om.Problem(model=om.Group())
    model = p.model
    nn = 1
    p.driver = om.ScipyOptimizeDriver()
    p.driver = om.pyOptSparseDriver(optimizer='SLSQP')

    p.driver.declare_coloring()

    traj = dm.Trajectory()
    phase = get_cond_phase()
    traj.add_phase('phase0', phase)
    p.model.add_subsystem(name='traj', subsys=traj,
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

    phase.add_objective('d', loc='final', ref=1)

    p.model.linear_solver = om.DirectSolver()
    p.setup(force_alloc_complex=True)

    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = 45
    p['traj.phase0.states:T'] = phase.interpolate(ys=[293.15, 333.15], nodes='state_input')
    p['traj.phase0.parameters:d'] = 0.001

    p.run_driver()
    # cpd = p.check_partials(method='cs', compact_print=True) #check partial derivatives
    # assert_check_partials(cpd)
    # quit()

    # dm.run_problem(p)

    # p.record('final') #trigger problem record (or call run_driver if it's attached to the driver)
    # p.run_driver()
    # p.cleanup()

    model.list_inputs(prom_name=True)
    model.list_outputs(prom_name=True)
    # p.check_partials(compact_print=True, method='cs')

    print("thickness: ", p['traj.phase0.parameters:d'], " (m)")

    # print("num of cells: ", p['n_cells'])
    # print("flux: ", p['flux'])
    # print("PCM mass: ", p['PCM_tot_mass'])
    # print("PCM thickness (mm): ", p['t_PCM'])
    # print("OHP mass: ", p['mass_OHP'])
    # print("packaging mass: ", p['p_mass'])
    # print("total mass: ", p['tot_mass'])
    # print("package mass fraction: ", p['mass_frac'])
    # print("pack energy density: ", p['energy'] / (p['tot_mass']))
    # print("cell energy density: ", (p['q_max'] * p['v_n_c']) / (p['cell_mass']))
    # print("pack energy (kWh): ", p['energy'] / 1000.)
    # print("pack cost ($K): ", p['n_cells'] * 0.4)

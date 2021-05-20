
import unittest

import numpy as np
import dymos as dm
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.src.sizing.heatpipe_transient import get_hp_phase


class TestHPtransient(unittest.TestCase):

    def run_asserts(self, p):
        # check duration
        tf = p.get_val('traj.phase.timeseries.time')[-1]
        assert_near_equal(tf, 10., tolerance=1.E-5)

        # check final values
        Tf = p.get_val('traj.phase.timeseries.states:T_cell_2')[-1]
        assert_near_equal(Tf, 324.32626399, tolerance=1.E-5)

    def test_get_hp_phase(self):

        p = om.Problem()
        p.driver = om.ScipyOptimizeDriver()
        p.driver = om.pyOptSparseDriver(optimizer='SLSQP')
        p.driver.declare_coloring()

        traj=dm.Trajectory()
        p.model.add_subsystem('traj', traj)

        num_cells = [5, 10, 15]
        color = ('C0', 'C1', 'C2', 'C3', 'C4')

        i = 0

        cells = 3

        # construct the phase, add all the cell states
        phase = get_hp_phase(num_cells=cells, db=(10, 10), num_segments=10, solve_segments=False, geom='round')

        traj.add_phase('phase', phase)


        phase.add_objective('time', loc='final', ref=1)

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        p.setup()

        p['traj.phase.t_initial'] = 0.0
        p['traj.phase.t_duration'] = 10.

        for cell in np.arange(cells):
            p['traj.phase.states:T_cell_{}'.format(cell)] = phase.interpolate(ys=[293.15, 333.15], nodes='state_input')

        p['traj.phase.states:T_cell_2'] = phase.interpolate(ys=[373.15, 333.15], nodes='state_input')

        p.run_driver()
        self.run_asserts(p)


if __name__ == '__main__':
    unittest.main()

from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, n2, view_connections
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.util.spec_test import assert_match_spec
from boring.src.sizing.heatpipe_transient import hp_transient


class TestHPtransient(unittest.TestCase):

    def run_asserts(self, p):
        # check duration
        tf = p.get_val('traj.phase.timeseries.time')[-1]
        assert_near_equal(tf, 12.64626495, tolerance=1.E-5)

        # check final values
        Tf = p.get_val('traj.phase.timeseries.states:T_cond')[-1]
        assert_near_equal(Tf, 300., tolerance=1.E-5)

    def test_hp_transient(self):
        p = hp_transient(show_plots=False, Tf_final=300)
        self.run_asserts(p)


if __name__ == '__main__':
    unittest.main()

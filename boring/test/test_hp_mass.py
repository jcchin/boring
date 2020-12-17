from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.util.spec_test import assert_match_spec
from boring.src.sizing.mass.mass import heatPipeMass


class TestHeatPipeMass(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('hp_mass', subsys=heatPipeMass(num_nodes=1))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_hp_mass(self):
        self.prob['hp_mass.D_od'] = 0.03
        self.prob['hp_mass.D_v'] = 0.03
        self.prob['hp_mass.L_heatpipe'] = 0.3
        self.prob['hp_mass.t_w'] = .0005
        self.prob['hp_mass.t_wk'] = 0.0005
        self.prob['hp_mass.cu_density'] = 8960
        self.prob['hp_mass.fill_wk'] = 0.10
        self.prob['hp_mass.liq_density'] = 1000
        self.prob['hp_mass.fill_liq'] = 0.7

        self.prob.run_model()

        assert_near_equal(self.prob.get_val('hp_mass.mass_heatpipe'), 0.12455787, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('hp_mass.mass_liquid'), 0.14844025, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('hp_mass.mass_wick'), 0.01287802, tolerance=1.0E-5)

    def test_partials(self):
        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()

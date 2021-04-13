from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.src.sizing.mass.insulation_mass import insulationMass


class TestInsulationMass(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('ins_mass', subsys=insulationMass(num_nodes=1))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_ins_mass(self):
        self.prob['ins_mass.num_cells'] = 4
        self.prob['ins_mass.num_stacks'] = 1
        self.prob['ins_mass.batt_l'] = 106.0
        self.prob['ins_mass.L_flux'] = 50.0
        self.prob['ins_mass.batt_h'] = 6.4
        self.prob['ins_mass.ins_density'] = 1.6e-7
        self.prob['ins_mass.ins_thickness'] = 2
        self.prob['ins_mass.batt_side_sep'] = 2
        self.prob['ins_mass.batt_end_sep'] = 2

        self.prob.run_model()

        assert_near_equal(self.prob.get_val('ins_mass.ins_backing_area'), 21214., tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('ins_mass.ins_side_sep_area'), 3392., tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('ins_mass.ins_end_sep_area'), 2688., tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('ins_mass.ins_volume'), 54588., tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('ins_mass.ins_mass'), 0.00873408, tolerance=1.0E-5)

    def test_partials(self):
        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()

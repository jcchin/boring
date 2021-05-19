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
        self.prob['ins_mass.batt_l'] = .10599
        self.prob['ins_mass.L_flux'] = 0.04902
        self.prob['ins_mass.batt_cutout_w'] = 0.050038
        self.prob['ins_mass.batt_h'] = 0.00635
        self.prob['ins_mass.ins_density'] = 100
        self.prob['ins_mass.LW:L_adiabatic'] = 0.002
        self.prob['ins_mass.A_pad'] = 0.00259781
        self.prob['ins_mass.ins_pcm_layer_t'] = 0.002
        self.prob['ins_mass.LW:L_flux_flat'] = 0.025
        self.prob['ins_mass.XS:H_hp'] = 0.005

        self.prob.run_model()

        assert_near_equal(self.prob.get_val('ins_mass.ins_cell_tray_mass'), 0.00582975, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('ins_mass.ins_pcm_layer_mass'), 0.00254468, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('ins_mass.ins_hp_layer_mass'), 0.0044652, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('ins_mass.ins_tot_mass'), 0.01283963, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('ins_mass.ins_tot_volume'), 0.0001284, tolerance=2.91E-5)

    def test_partials(self):
        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()

from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.util.spec_test import assert_match_spec
from boring.src.sizing.thermal_resistance.axial_thermal_resistance import AxialThermalResistance


class TestAxialResistance(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('axial_thermal', subsys=AxialThermalResistance(num_nodes=1))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_axial_outputs(self):
        self.prob['axial_thermal.k_w'] = 11.4
        self.prob['axial_thermal.k_wk'] = 6.471327242361979
        self.prob['axial_thermal.LW:L_eff'] = 0.05
        self.prob['axial_thermal.XS:A_w'] = 8.63937979737193e-06
        self.prob['axial_thermal.XS:A_wk'] = 9.342782392510687e-06 

        self.prob.run_model()

        assert_near_equal(self.prob.get_val('axial_thermal.R_aw'), 507.67126983060723, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('axial_thermal.R_awk'), 826.99028728, tolerance=1.0E-5)

    def test_partials(self):  # derivative check

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)

    # def test_io_spec(self): 

    #     subsystem = AxialThermalResistance(num_nodes=1)
    #     assert_match_spec(subsystem, 'Design_specs/struct.json')


if __name__ == '__main__':
    unittest.main()

from __future__ import print_function, division, absolute_import

import unittest
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from boring.util.spec_test import assert_match_spec
from boring.src.sizing.thermal_resistance.vapor_thermal_resistance import VaporThermalResistance

class TestVaporResistance(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('vapor_thermal', subsys=VaporThermalResistance(num_nodes=1))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_vapor_outputs(self):
        self.prob['vapor_thermal.D_v'] = 10.1
        self.prob['vapor_thermal.R_g'] = 0.2
        self.prob['vapor_thermal.mu_v'] = 0.03
        self.prob['vapor_thermal.T_hp'] = 300
        self.prob['vapor_thermal.h_fg'] = 100
        self.prob['vapor_thermal.P_v'] = 1000
        self.prob['vapor_thermal.rho_v'] = 100
        self.prob['vapor_thermal.L_eff'] = 0.5
        
        assert_near_equal(self.prob.get_val('vapor_thermal.r_h'), 0.05, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('vapor_thermal.R_v'), 0.1100079, tolerance=1.0E-5)

    def test_partials(self): # derivative check

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)

    # def test_io_spec(self): 

    #     subsystem = VaporThermalResistance(num_nodes=1)
    #     assert_match_spec(subsystem, 'Design_specs/struct.json')


if __name__ =='__main__':
    unittest.main()
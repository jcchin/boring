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

        self.prob['axial_thermal.epsilon'] = 1 
        self.prob['axial_thermal.k_w'] = 2
        self.prob['axial_thermal.k_l'] = 3
        self.prob['axial_thermal.L_adiabatic'] = 4
        self.prob['axial_thermal.A_w'] = 5
        self.prob['axial_thermal.A_wk'] = 6
        assert_near_equal(self.prob.get_val('axial_thermal.k_wk'), 3., tolerance=1.0E-5) 
        assert_near_equal(self.prob.get_val('axial_thermal.R_aw'), 0.4, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('axial_thermal.R_awk'), 0.22222222, tolerance=1.0E-5)

    def test_partials(self): # derivative check

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)

    # def test_io_spec(self): 

    #     subsystem = AxialThermalResistance(num_nodes=1)
    #     assert_match_spec(subsystem, 'Design_specs/struct.json')


if __name__ =='__main__':
    unittest.main()
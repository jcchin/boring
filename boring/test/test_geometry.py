from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.util.spec_test import assert_match_spec
# from boring.src.sizing.heatpipe_core_geometries import CoreGeometries
from boring.src.sizing.geometry.hp_geometry import HeatPipeSizeGroup

class TestSize(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('sizing', subsys=HeatPipeSizeGroup(num_nodes=1))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

 
    def test_geometry_outputs(self): # calculation regression test

        self.prob['sizing.D_od'] = 2 
        self.prob['sizing.t_w'] = 0.01
        self.prob['sizing.D_v'] = 0.5
        self.prob['sizing.L_flux'] = 6
        self.prob['sizing.L_adiabatic'] = 0.01

        self.prob.run_model()

        assert_near_equal(self.prob.get_val('sizing.A_w'), 0.06251769, tolerance=1.0E-5) 
        assert_near_equal(self.prob.get_val('sizing.A_wk'), 2.88272542, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('sizing.A_inter'), 9.42477796, tolerance=1.0E-5)

        assert_near_equal(self.prob.get_val('sizing.r_i'), 0.99, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('sizing.A_flux'), 37.69911184, tolerance=1.0E-5)
        #assert_near_equal(self.prob.get_val('sizing.L_eff'), 5.51, tolerance=1.0E-5) #this calc will no longer be an output of this component


    def test_partials(self): # derivative check

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)

    # def test_io_spec(self): 

    #     subsystem = AxialThermalResistance(num_nodes=1)
    #     assert_match_spec(subsystem, 'Design_specs/struct.json')


if __name__ =='__main__':
    unittest.main()
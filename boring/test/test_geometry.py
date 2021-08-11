from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.util.spec_test import assert_match_spec
from boring.src.sizing.geometry.hp_geom import HPgeom


class TestSize(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('round', subsys=HPgeom(num_nodes=1, geom='round'), 
                            promotes_inputs=['*'])
        p1.model.add_subsystem('flat', subsys=HPgeom(num_nodes=1, geom='flat'), 
                            promotes_inputs=['*'])

        p1.setup(force_alloc_complex=True)


        self.prob['LW:L_flux'] = 50.8
        self.prob['LW:L_adiabatic'] = 3.
        self.prob['XS:t_w'] = 0.5
        self.prob['XS:t_wk'] = 0.69

        # round inputs
        self.prob['XS:D_v'] = 0.56
        # flat inputs
        self.prob['XS:W_v'] = 0.56
        self.prob['XS:H_v'] = 0.56

        p1.run_model()
        # p1.model.list_outputs(values=True, prom_name=True)


    def test_round_geometry_outputs(self): # calculation regression test

        assert_near_equal(self.prob.get_val('round.XS:A_w'), 3.83274304, tolerance=1.0E-5) 
        assert_near_equal(self.prob.get_val('round.XS:A_wk'), 2.70962366, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('round.LW:A_inter'), 89.37202781, tolerance=1.0E-5)

        assert_near_equal(self.prob.get_val('round.XS:r_i'), 0.97, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('round.LW:A_flux'), 469.203146, tolerance=1.0E-5)
        #assert_near_equal(self.prob.get_val('round.L_eff'), 5.51, tolerance=1.0E-5) #this calc will no longer be an output of this component

    def test_flat_geometry_outputs(self): # calculation regression test


        assert_near_equal(self.prob.get_val('flat.XS:A_w'), 4.88, tolerance=1.0E-5) 
        assert_near_equal(self.prob.get_val('flat.XS:A_wk'), 3.45, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('flat.LW:A_inter'), 28.448, tolerance=1.0E-5)

        assert_near_equal(self.prob.get_val('flat.LW:A_flux'), 149.352, tolerance=1.0E-5)
        #assert_near_equal(self.prob.get_val('flat.L_eff'), 5.51, tolerance=1.0E-5) #this calc will no longer be an output of this component

    def test_partials(self): # derivative check

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)

    # def test_io_spec(self): 

    #     subsystem = AxialThermalResistance(num_nodes=1)
    #     assert_match_spec(subsystem, 'Design_specs/struct.json')


if __name__ =='__main__':
    unittest.main()
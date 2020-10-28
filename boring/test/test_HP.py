from __future__ import print_function, division, absolute_import

import unittest


import numpy as np
from openmdao.api import Problem, Group, IndepVarComp, n2
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.util.spec_test import assert_match_spec
from boring.src.sizing.heat_pipe import OHP, FHP


class TestHP(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('ohp', subsys=OHP(num_nodes=1))
        p1.model.add_subsystem('fhp', subsys=FHP(num_nodes=1), promotes_inputs=[
                        ('d_init','d_{init}'),('tot_len','L_{pack}'),('rho_FHP','rho_{HP}')
                        ])

        p1.setup(force_alloc_complex=True)
        p1.run_model()
        p1.model.list_outputs(values=True, prom_name=True)

 
    def test_OHP(self):

        assert_near_equal(self.prob.get_val('ohp.mass_OHP'), 385.90453402, tolerance=1.0E-5)

    def test_FHP(self):

        assert_near_equal(self.prob.get_val('fhp.fhp_mass'), 21256733.87035246, tolerance=1.0E-5)

    # def test_partials(self):

    #     data = self.prob.check_partials(out_stream=None, method='cs')
    #     assert_check_partials(data, atol=1e-10, rtol=1e-10)

    def test_FHP_io_spec(self): 

        p1 = self.prob = Problem(model=Group())

        self.prob.model.set_input_defaults('ref_len', 240.)
        self.prob.model.set_input_defaults('req_flux', 50.)
        p1.model.add_subsystem('fhp', subsys=FHP(num_nodes=1), promotes_inputs=['ref_len','req_flux',
                        ('d_init','d_{init}'),('tot_len','L_{pack}'),('rho_FHP','rho_{HP}')
                        ],
                        promotes_outputs=[('fhp_mass','mass_{HP}'),('t_hp','t_{HP}')])
        p1.setup()
        #p1.model.list_inputs(prom_name=True)
        assert_match_spec(p1.model, 'Design_specs/heat_pipe.json')


if __name__ =='__main__':
    unittest.main()
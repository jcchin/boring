from __future__ import print_function, division, absolute_import

import unittest
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from boring.util.spec_test import assert_match_spec
from boring.src.sizing.material_properties.pcm_properties import PCM_props


class TestPCMProps(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('props', subsys=PCM_props(num_nodes=1))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_bulk_calc(self):
        self.prob['props.porosity'] = 1.
        self.prob['props.k_pcm'] = 12.
        self.prob.run_model()
        assert_near_equal(self.prob.get_val('props.k_bulk'), 12., tolerance=1.0E-5)

        self.prob['props.porosity'] = 0.
        self.prob['props.k_foam'] = 21.
        self.prob.run_model()
        assert_near_equal(self.prob.get_val('props.k_bulk'), 21.,
                          tolerance=1.0E-5)  # check the same exact output, make sure it changes

    def test_partials(self):  # derivative check

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)

    # def test_io_spec(self): 

    #     subsystem = PCM_props(num_nodes=1)
    #     assert_match_spec(subsystem, 'Design_specs/struct.json')


if __name__ == '__main__':
    unittest.main()

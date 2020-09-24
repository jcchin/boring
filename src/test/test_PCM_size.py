from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from src.util.spec_test import assert_match_spec
from src.PCM_size import SizingGroup


class TestPCM(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('pcm', subsys=SizingGroup(num_nodes=1))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

 
    def test_num_cells(self):

        assert_near_equal(self.prob.get_val('pcm.n_cells'), 2941.17647059, tolerance=1.0E-5)

    # def test_fuselage_partials(self):

    #     data = self.prob.check_partials(out_stream=None, method='cs')
    #     assert_check_partials(data, atol=1e-10, rtol=1e-10)

    def test_io_spec(self): 

        subsystem = SizingGroup(num_nodes=1)
        assert_match_spec(subsystem, 'PCM_size.json')


if __name__ =='__main__':
    unittest.main()
from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.src.sizing.geometry.hp_geometry import SizeComp


class TestSizeComp(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('size_comp', subsys=SizeComp(num_nodes=1, geom='FLAT'))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_size_comp(self):
        self.prob['size_comp.L_flux'] = 0.02
        self.prob['size_comp.L_adiabatic'] = 0.03
        self.prob['size_comp.t_w'] = 0.0005
        self.prob['size_comp.t_wk'] = 0.00069
        self.prob['size_comp.num_cells'] = 1
        self.prob['size_comp.W'] = 0.02


        self.prob.run_model()

        assert_near_equal(self.prob.get_val('size_comp.A_flux'), 0.0004, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('size_comp.L_eff'), 0.08, tolerance=1.0E-5)


    def test_partials(self):
        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()

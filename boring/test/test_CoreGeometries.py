from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.src.sizing.geometry.hp_geometry import CoreGeometries


class TestCoreGeometries(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('core_geometries', subsys=CoreGeometries(num_nodes=1, geom='flat'))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_size_comp(self):
        self.prob['core_geometries.t_wk'] = 0.056
        self.prob['core_geometries.W'] = 0.056
        self.prob['core_geometries.t_w'] = 0.056
        self.prob['core_geometries.L_flux'] = 0.056


        self.prob.run_model()

        assert_near_equal(self.prob.get_val('core_geometries.A_w'), 0.003136, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('core_geometries.A_wk'), 0.003136, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('core_geometries.A_inter'), 0.003136, tolerance=1.0E-5)


    def test_partials(self):
        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()

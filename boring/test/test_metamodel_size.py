from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.util.spec_test import assert_match_spec
from boring.metamodel.sizing_component import MetaPackSizeComp


class TestSize(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('size', subsys=MetaPackSizeComp(num_nodes=1))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_tot_size(self):  # calculation regression test

        self.prob['size.cell_rad'] = 9
        self.prob['size.extra'] = 1
        self.prob['size.ratio'] = 1
        self.prob['size.length'] = 65
        self.prob['size.al_density'] = 2.7e-6
        self.prob['size.n'] = 4

        self.prob.run_model()

        assert_near_equal(self.prob.get_val('size.hole_r'), 3.72792206, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('size.side'), 72., tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('size.solid_area'), 5184., tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('size.cell_cutout_area'), 4071.50407905, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('size.air_cutout_area'), 698.55966145, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('size.area'), 413.9362595, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('size.volume'), 26905.8568673, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('size.mass'), 0.07264581, tolerance=1.0E-5)

    def test_partials(self):  # derivative check

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-9, rtol=1e-10)  # volume wrt extra  |  1.1641532182693481e-10


    # def test_io_spec(self): 

    #     subsystem = MetaPackSizeComp(num_nodes=1)
    #     assert_match_spec(subsystem, 'Design_specs/struct.json')


if __name__ == '__main__':
    unittest.main()

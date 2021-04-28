from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.util.spec_test import assert_match_spec
from boring.src.sizing.mass.misc_mass import packMass


class TestMass(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('mass', subsys=packMass(num_nodes=1))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_tot_mass(self):  # calculation regression test

        self.prob['mass.n_cells'] = 100  # set input vals

        self.prob.run_model()

        assert_near_equal(self.prob.get_val('mass.tot_mass'), 109.32,
                          tolerance=1.0E-5)  # check you get the expected output

    def test_partials(self):  # derivative check

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)

    # def test_io_spec(self): 

    #     subsystem = packMass(num_nodes=1)
    #     assert_match_spec(subsystem, 'Design_specs/struct.json')


if __name__ == '__main__':
    unittest.main()

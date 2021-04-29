from __future__ import print_function, division, absolute_import

import unittest
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from boring.util.spec_test import assert_match_spec
from boring.src.sizing.material_properties.pcm_properties import PCM_props
from boring.src.sizing.material_properties.pcm_ps import PCM_PS
from boring.src.sizing.material_properties.cp_func import PCM_Cp
from boring.src.sizing.material_properties.pcm_group import PCM_Group


class TestPCMcp(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('cp', subsys=PCM_Cp(num_nodes=1))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_bulk_calc(self):
        # go out of bounds
        self.prob['cp.T'] = 80.
        self.prob.run_model()
        assert_near_equal(self.prob.get_val('cp.cp_pcm'), 1.5, tolerance=1.0E-5)

        # go in bounds
        self.prob['cp.T'] = 336.
        self.prob.run_model()
        assert_near_equal(self.prob.get_val('cp.cp_pcm'), 50., tolerance=1.0E-5)  # check the same exact output, make sure it changes

        # go in bounds
        self.prob['cp.T'] = 333.
        self.prob.run_model()
        assert_near_equal(self.prob.get_val('cp.cp_pcm'), 25.75, tolerance=1.0E-5)  # check the same exact output, make sure it changes


    def test_partials(self):  # derivative check

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)



class TestPCMps(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('ps', subsys=PCM_PS(num_nodes=1))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_bulk_calc(self):
        self.prob['ps.T_lo'] = 20.
        self.prob['ps.T_hi'] = 100.
        self.prob['ps.T'] = 80.
        self.prob.run_model()
        assert_near_equal(self.prob.get_val('ps.PS'), 0.25, tolerance=1.0E-5)

        # go out of bounds
        self.prob['ps.T'] = 0.
        self.prob.run_model()
        assert_near_equal(self.prob.get_val('ps.PS'), 1.25, tolerance=1.0E-5)  # check the same exact output, make sure it changes

    def test_partials(self):  # derivative check

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)


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


class TestPCMgroup(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=PCM_Group())
        p1.setup(force_alloc_complex=True)
        p1.model.list_outputs()

    def test_group_calc(self):
        self.prob['T'] = 334
        self.prob.run_model()
        assert_near_equal(self.prob.get_val('rate.c_p'), 49880.07777508, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('bulk.cp_bulk'), 49.88007777508, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('rate.Tdot'), -2.00480842e-05, tolerance=1.0E-5)

        self.prob['T'] = 340
        self.prob.run_model()
        assert_near_equal(self.prob.get_val('rate.c_p'), 1500.29799247, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('rate.Tdot'), -0.00066653, tolerance=1.0E-5)


if __name__ == '__main__':
    unittest.main()

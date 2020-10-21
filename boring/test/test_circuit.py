from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from lcapy import R

from src.util.spec_test import assert_match_spec
from src.sizing.circuit import Circuit


class TestCircuit(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('circ', subsys=Circuit())

        p1.setup(force_alloc_complex=True)
        p1.run_model()
        p1.model.list_outputs(values=True, prom_name=True)

 
    def test_resistance(self):

        Rtot = (R(1) + (R(2) + R(2)| R(3) + (R(4)|R(5)+R(5)+R(5)+R(5))))

        print(Rtot.simplify())

        #assert_near_equal(self.prob.get_val('ohp.mass_OHP'), 385.90453402, tolerance=1.0E-5)

        draw = False # plot the thermal network
        if draw:
            Rtot.draw('Thermal_Network.pdf')

if __name__ =='__main__':
    unittest.main()

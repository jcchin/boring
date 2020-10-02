from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from boring.util.spec_test import assert_match_spec, assert_match_vals
from boring.src.model.battery_group import BatteryGroup
from boring.src.model.battery_statics import BatteryStatics

class TestBatteryGroup(unittest.TestCase):

    def test_amprius_group(self):
        p1 = Problem(model=Group())
        p1.model.add_subsystem('battery', subsys=BatteryGroup(num_nodes=1, cell_type='AMPRIUS'))
        p1.setup()

        p1['battery.Q_{max}'] = 3.5
        p1['battery.cell.SOC'] = 0.2
        p1['battery.cell.U_Th'] = 0.0
        p1['battery.interp_group.interp_comp.SOC'] = 0.2
        p1['battery.interp_group.interp_comp.T_batt'] = 30

        p1.run_model()

        print('dXdt:SOC: ', p1['battery.dXdt:SOC'])
        print('V_thev: ', p1['battery.dXdt:V_{thev}'])
        print('battery pack voltage: ', p1['battery.V_{batt,actual}'])
        print('Open Circuit voltage: ', p1['battery.U_oc'])
        print('battery cell resistance: ', p1['battery.R_0'])
        print('battery cell Th resistance: ', p1['battery.R_Th'])
 
        assert_near_equal(p1.get_val('battery.dXdt:SOC', '1/s'), -6.4484127e-06, tolerance=1.0E-5)
        assert_near_equal(p1.get_val('battery.dXdt:V_{thev}','V/s'), 2.70833333e-05, tolerance=1.0E-5)
        assert_near_equal(p1.get_val('battery.U_oc','V'), 3.45166005, tolerance=1.0E-5)
        assert_near_equal(p1.get_val('battery.V_{batt,actual}','V'), 441.25088655, tolerance=1.0E-5)
        #check map lookup
        assert_near_equal(p1.get_val('battery.R_0','ohm'), 0.054, tolerance=1.0E-5)
        assert_near_equal(p1.get_val('battery.R_Th','ohm'), 0.05686946, tolerance=1.0E-5)

    def test_18650_group(self):
        p2 = Problem(model=Group())
        p2.model.add_subsystem('battery', subsys=BatteryGroup(num_nodes=1, cell_type='18650'))
        p2.setup()

        p2['battery.Q_{max}'] = 3.0
        p2['battery.cell.SOC'] = 0.2
        p2['battery.cell.U_Th'] = 0.0
        p2['battery.interp_group.interp_comp.SOC'] = 0.2
        p2['battery.interp_group.interp_comp.T_batt'] = 30

        p2.run_model()

        print('dXdt:SOC: ', p2['battery.dXdt:SOC'])
        print('V_thev: ', p2['battery.dXdt:V_{thev}'])
        print('battery pack voltage: ', p2['battery.V_{batt,actual}'])
        print('Open Circuit voltage: ', p2['battery.U_oc'])
        print('battery cell resistance: ', p2['battery.R_0'])
        print('battery cell Th resistance: ', p2['battery.R_Th'])
 
        assert_near_equal(p2.get_val('battery.dXdt:SOC', '1/s'), -7.52314815e-06, tolerance=1.0E-5)
        assert_near_equal(p2.get_val('battery.dXdt:V_{thev}','V/s'), 4.0625e-05, tolerance=1.0E-5)
        assert_near_equal(p2.get_val('battery.U_oc','V'), 3.37185148, tolerance=1.0E-5)
        assert_near_equal(p2.get_val('battery.V_{batt,actual}','V'), 431.33698944, tolerance=1.0E-5)
        #check map lookup
        assert_near_equal(p2.get_val('battery.R_0','ohm'), 0.025, tolerance=1.0E-5)
        assert_near_equal(p2.get_val('battery.R_Th','ohm'), 0.04207528, tolerance=1.0E-5)


    def test_dynamic_io_spec(self): 

        p3 = Problem(model=Group())
        subsystem = p3.model.add_subsystem('battery', subsys=BatteryGroup(num_nodes=1, cell_type='AMPRIUS'))
        p3.setup()
        p3.run_model()
        #subsystem = BatteryGroup(num_nodes=1)
        subsystem.list_inputs(prom_name=True)
        assert_match_spec(subsystem, 'ODE_specs/battery_comp.json')

if __name__ =='__main__':
    unittest.main()
from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.util.spec_test import assert_match_spec, assert_match_vals
from boring.src.model.battery_statics import BatteryStatics, BatteryStaticsGroup


class TestBatteryStatics(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())

        p1.model.add_subsystem('battery', subsys=BatteryStaticsGroup())

        p1.setup(force_alloc_complex=True)

        p1['battery.mass_{cell}'] = 0.045

        p1.run_model()

        mass = p1['battery.mass_{battery}']
        print('pack mass: ', mass)
        print('n_s: ', p1['battery.n_{series}'])
        print('n_p: ', p1['battery.n_{parallel}'])
        print('n_p2: ', p1['battery.n_{parallel}2'])
        print('total cells: ', p1['battery.n_{series}']*max(p1['battery.n_{parallel}'],p1['battery.n_{parallel}2']))
        print('vol pack (sqft): ', p1['battery.volume_{pack}']/12./12.)

    def test_batt_statics_calc(self):
 
        assert_near_equal(self.prob.get_val('battery.mass_{battery}','kg'), 249.754, tolerance=1.0E-3)

    def test_batt_statics_partials(self):
        data = self.prob.check_partials( method='cs', compact_print=True)
        # dirty hack to get around problem with the assert_check_partials method 
        # force it to use absolute error cause analytic is 0 (OM's check isn't smart enough to handle this right now)
        # TODO: remove this when OM is updated
        from openmdao.core.problem import ErrorTuple
        analytic_val = data['battery.BatteryStatics']['n_{parallel}2','I_{batt}']['J_fwd']
        check_val = data['battery.BatteryStatics']['n_{parallel}2','I_{batt}']['J_fd']
        et = ErrorTuple(forward=analytic_val-check_val, reverse=analytic_val-check_val, forward_reverse=0.0)
        data['battery.BatteryStatics']['n_{parallel}2','I_{batt}']['rel error'] = et
        
        assert_check_partials(data, atol=1.e-10, rtol=1.e-10)

    def test_batt_statics_io_spec(self): 

        subsystem = BatteryStaticsGroup()

        assert_match_spec(subsystem, 'Design_specs/battery_comp.json')


if __name__ =='__main__':
    unittest.main()
from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from lcapy import R

from boring.util.spec_test import assert_match_spec
from boring.src.sizing.circuit import Circuit, Evaporator, Condensor, thermal_link


class TestCircuit(unittest.TestCase):



    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('circ', subsys=Circuit())

        p1.setup(force_alloc_complex=True)

        Rexe = 0.0000001
        Rexc = 0.0000001
        Rwe = 0.2545383947014702
        Rwke = 0.7943030881649811
        Rv = 8.852701208752846e-06
        Rintere = 0.00034794562965549745
        Rinterc = 0.00017397281482774872
        Rwkc = 0.39715154408249054
        Rwka = 744.3007160198263
        Rwa = 456.90414284754644
        Rwc = 0.1272691973507351
        self.prob['circ.Rex_e.R'] = Rexe
        self.prob['circ.Rex_c.R'] = Rexc
        self.prob['circ.Rwe.R'] = Rwe
        self.prob['circ.Rwke.R'] = Rwke
        self.prob['circ.Rv.R'] = Rv
        self.prob['circ.Rinter_e.R'] = Rintere
        self.prob['circ.Rinter_c.R'] = Rinterc
        self.prob['circ.Rwkc.R'] = Rwkc
        self.prob['circ.Rwka.R'] = Rwka
        self.prob['circ.Rwa.R'] = Rwa
        self.prob['circ.Rwc.R'] = Rwc
        self.prob['circ.Rex_e.T_in'] = 100
        self.prob['circ.Rex_c.T_out'] = 20

        # test thermal link function



        p1.run_model()
        p1.model.list_outputs(values=True, prom_name=True)

 
    def test_resistance(self):

        Rexe = 0.0000001
        Rexc = 0.0000001
        Rwe = 0.2545383947014702
        Rwke = 0.7943030881649811
        Rv = 8.852701208752846e-06
        Rintere = 0.00034794562965549745
        Rinterc = 0.00017397281482774872
        Rwkc = 0.39715154408249054
        Rwka = 744.3007160198263
        Rwa = 456.90414284754644
        Rwc = 0.1272691973507351

        Rtot = (R(Rexe) + (R(Rwa) | R(Rwe) + (R(Rwka)|R(Rwke)+R(Rintere)+R(Rv)+R(Rinterc)+R(Rwkc))+ R(Rwc))+ R(Rexc))

        print(Rtot.simplify())
        ans = 16731692103737332239244353077427184638278095509511778941./10680954190791611228174081719413008273307025000000000000.

        Rtot2 = (self.prob.get_val('circ.n1.T')-self.prob.get_val('circ.n8.T'))/self.prob.get_val('circ.Rex_c.q')

        assert_near_equal(Rtot2, ans, tolerance=1.0E-5)

        draw = True # plot the thermal network
        if draw:
            Rtot.draw('Thermal_Network.pdf')

    def test_link(self):

        p2 = self.prob2 = Problem(model=Group())
        p2.model.add_subsystem('evap', Evaporator(links=1))
        p2.model.add_subsystem('cond', Condensor(links=1))

        thermal_link(p2.model,'evap','cond')

        p2.setup(force_alloc_complex=True)

        Rexe = 0.0000001
        Rexc = 0.0000001
        Rwe = 0.2545383947014702
        Rwke = 0.7943030881649811
        Rv = 8.852701208752846e-06
        Rintere = 0.00034794562965549745
        Rinterc = 0.00017397281482774872
        Rwkc = 0.39715154408249054
        Rwka = 744.3007160198263
        Rwa = 456.90414284754644
        Rwc = 0.1272691973507351
        self.prob2['evap.Rex_e.R'] = Rexe
        self.prob2['evap.Rwe.R'] = Rwe
        self.prob2['evap.Rwke.R'] = Rwke
        self.prob2['evap.Rinter_e.R'] = Rintere
        self.prob2['cond.Rinter_c.R'] = Rinterc
        self.prob2['cond.Rwkc.R'] = Rwkc
        self.prob2['cond.Rwc.R'] = Rwc
        self.prob2['cond.Rex_c.R'] = Rexc

        self.prob2['evap_bridge.Rv.R'] = Rv
        self.prob2['evap_bridge.Rwka.R'] = Rwka
        self.prob2['evap_bridge.Rwa.R'] = Rwa
        self.prob2['evap.Rex_e.T_in'] = 100
        self.prob2['cond.Rex_c.T_out'] = 20

        p2.run_model()

        Rtot3 = (self.prob2.get_val('evap.n1.T')-self.prob2.get_val('cond.n1.T'))/self.prob2.get_val('cond.Rex_c.q')

        ans = 16731692103737332239244353077427184638278095509511778941./10680954190791611228174081719413008273307025000000000000.
        assert_near_equal(Rtot3, ans, tolerance=1.0E-5)

if __name__ =='__main__':
    unittest.main()

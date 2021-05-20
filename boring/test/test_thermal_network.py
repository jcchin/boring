from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp, n2, view_connections
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

#from lcapy import R, LSection, Series

from boring.util.spec_test import assert_match_spec
from boring.util.load_inputs import load_inputs

from boring.src.sizing.thermal_network import Circuit, Radial_Stack, thermal_link
from boring.src.sizing.geometry.hp_geom import HPgeom
from boring.src.sizing.heatpipe_group import HeatPipeGroup


from PySpice.Spice.Netlist import Circuit as PyCircuit
from PySpice.Unit import *

class TestCircuit(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('circ', subsys=Circuit())

        p1.setup()

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

        p1.run_model()
        # p1.model.list_inputs(values=True, prom_name=True)
        # p1.model.list_outputs(values=True, prom_name=True)

    def skip_test_resistance(self): # this test works, it just doesn't run on Travis yet.
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

        circuit = PyCircuit('HeatPipe')

        circuit.V('input', 1, circuit.gnd, 80@u_V)
        circuit.R('Reex', 1, 2, 0.0000001@u_kΩ)
        circuit.R('Rew', 2, 3, 0.2545383947014702@u_kΩ)
        circuit.R('Rewk', 3, 4, 0.7943030881649811@u_kΩ)
        circuit.R('Rintere', 4, 5, 0.00034794562965549745@u_kΩ)
        circuit.R('Raw', 2, 9, 456.90414284754644@u_kΩ)
        circuit.R('Rawk', 3, 8, 744.3007160198263@u_kΩ)
        circuit.R('Rv', 5, 6, 8.852701208752846e-06@u_kΩ)
        circuit.R('Rinterc', 6, 7, 0.00017397281482774872@u_kΩ)
        circuit.R('Rcwk', 7, 8, 0.39715154408249054@u_kΩ)
        circuit.R('Rcw', 8, 9, 0.1272691973507351@u_kΩ)
        circuit.R('Rcex', 9, circuit.gnd, 0.0000001@u_kΩ)


        for resistance in (circuit.RReex, circuit.RRcex):
            resistance.minus.add_current_probe(circuit) # to get positive value

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.operating_point()

        for node in analysis.nodes.values():
            print('Node {}: {:5.2f} V'.format(str(node), float(node))) # Fixme: format value + unit

        #Pyspice treates ground as 0, need to offset by base temperature (circ.Rex_c.T_out = 20degC)

        assert_near_equal(self.prob.get_val('circ.n3.T'),float(analysis.nodes['4'])+20.,tolerance=1.0E-5)


    def test_link(self):
        nn = 1
        num_cells_tot = 2
        p2 = self.prob2 = Problem(model=Group())

        p2.model.add_subsystem(name = 'size',
                          subsys = HPgeom(num_nodes=nn, geom='round'),
                          promotes_inputs=['LW:L_flux', 'LW:L_adiabatic', 'XS:t_w', 'XS:t_wk', 'XS:D_v'],
                          promotes_outputs=['XS:D_od','XS:r_i', 'XS:A_w', 'XS:A_wk', 'LW:A_flux', 'LW:A_inter']) 


        p2.model.add_subsystem(name='hp',
                              subsys=HeatPipeGroup(num_nodes=nn, num_cells=num_cells_tot, pcm_bool=False, geom='round'),
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])


        p2.setup(force_alloc_complex=True)

        self.prob2['cell_0.Rex.R'] = 0.0000001
        self.prob2['cell_1.Rex.R'] = 0.0000001
        self.prob2['cell_0.Rex.T_in'] = 100
        self.prob2['cell_1.Rex.T_in'] = 20

        p2.run_model()
        p2.model.list_inputs(values=True, prom_name=True)
        p2.model.list_outputs(values=True, prom_name=True)
        # n2(p2)
        # view_connections(p2)

        Rtot3 = (self.prob2.get_val('cell_0.n1.T') - self.prob2.get_val('cell_1.n1.T')) / np.abs(
            self.prob2.get_val('cell_1.Rex.q'))

        ans = 1.5791057
        #assert_near_equal(Rtot3, ans, tolerance=5.0E-3)
        assert_near_equal(self.prob2.get_val('cell_0.Rex.R'), 0.0000001, tolerance=5.0E-3)
        assert_near_equal(self.prob2.get_val('cell_1.Rex.R'), 0.0000001, tolerance=5.0E-3)
        assert_near_equal(self.prob2.get_val('cell_0.Rw.R'), 0.1272691973507351, tolerance=5.0E-3)
        assert_near_equal(self.prob2.get_val('cell_0.Rwk.R'), 0.39714649767293836, tolerance=5.0E-3)
        assert_near_equal(self.prob2.get_val('cell_0_bridge.Rv.R'), 8.783819660208796e-06, tolerance=5.0E-3)
        assert_near_equal(self.prob2.get_val('cell_0.Rinter.R'), 0.00016523883294100212, tolerance=5.0E-3)
        assert_near_equal(self.prob2.get_val('cell_1.Rinter.R'), 0.00016523883294100212, tolerance=5.0E-3)
        assert_near_equal(self.prob2.get_val('cell_1.Rwk.R'), 0.39715154408249054, tolerance=5.0E-3)
        assert_near_equal(self.prob2.get_val('cell_0_bridge.Rwka.R'), 826.9902872847766, tolerance=5.0E-3)
        assert_near_equal(self.prob2.get_val('cell_0_bridge.Rwa.R'), 507.67126983060723, tolerance=5.0E-3)
        assert_near_equal(self.prob2.get_val('cell_1.Rw.R'), 0.1272691973507351, tolerance=5.0E-3)


    def _test_two_port(self):
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

        Rtota = R(Rexe) + (
                    R(Rwa) | R(Rwe) + (R(Rwka) | R(Rwke) + R(Rintere) + R(Rv) + R(Rinterc) + R(Rwkc)) + R(Rwc)) + R(
            Rexc)
        #                                                                                                   |         |
        Rtot = R(Rexe) + (
                    R(Rwa) | R(Rwe) + (R(Rwka) | R(Rwke) + R(Rintere) + R(Rv) + R(Rinterc) + R(Rwkc)) + R(Rwc)) + (
                           R(1.6 + Rexc) | (R(Rexe) + (
                               R(Rwa) | R(Rwe) + (R(Rwka) | R(Rwke) + R(Rintere) + R(Rv) + R(Rinterc) + R(Rwkc)) + R(
                           Rwc)) + R(Rexc)))
        print(Rtot.simplify())
        Rtot.draw('test.pdf')

        Rtot_2 = LSection(Rtota, Rtota)
        ans1 = Rtot_2.Y1sc
        print(ans1.simplify())
        ans2 = Rtot_2.Y2sc
        print(ans2.simplify())


if __name__ == '__main__':
    unittest.main()

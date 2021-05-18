from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.util.spec_test import assert_match_spec
from boring.src.sizing.mass.pcm_mass import pcmMass


class TestPCM(unittest.TestCase):

    def setUp(self):
        nn=1
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('comp1', pcmMass(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_mass(self):
        assert_near_equal(self.prob.get_val('mass_pcm'), 0.00862692, tolerance=1.0E-5)

    def test_partials(self):

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)

    def skip_test_io_spec(self):
        p1 = self.prob = Problem(model=Group())

        # self.prob.model.set_input_defaults('frac_absorb', 240.)
        # self.prob.model.set_input_defaults('req_flux', 50.)
        self.prob.model.set_input_defaults('frame_mass', 0.)
        self.prob.model.set_input_defaults('cell_mass', 0.)
        self.prob.model.set_input_defaults('cell_h', 0.)
        self.prob.model.set_input_defaults('ext_cooling', 0.)
        self.prob.model.set_input_defaults('dur', 0.)
        self.prob.model.set_input_defaults('t_HP', 0.)
        self.prob.model.set_input_defaults('cell_l', 0.)
        self.prob.model.set_input_defaults('ext_cool_mass', 0.)
        self.prob.model.set_input_defaults('runawayJ', 0.)
        self.prob.model.set_input_defaults('W', 0.)
        self.prob.model.set_input_defaults('LH_PCM', 0.)
        self.prob.model.set_input_defaults('rho_PCM', 0.)
        self.prob.model.set_input_defaults('cell_s_l', 0.)
        self.prob.model.set_input_defaults('q_max', 0.)
        self.prob.model.set_input_defaults('cell_s_h', 0.)
        self.prob.model.set_input_defaults('L', 0.)
        self.prob.model.set_input_defaults('frac_absorb', 0.)
        self.prob.model.set_input_defaults('v_n_c', 0.)
        self.prob.model.set_input_defaults('energy', 0.)
        self.prob.model.set_input_defaults('cell_s_w', 0.)
        self.prob.model.set_input_defaults('cell_w', 0.)
        self.prob.model.set_input_defaults('missionJ', 0.)

        p1.model.add_subsystem('size', subsys=SizingGroup(num_nodes=1), promotes_inputs=[
            'frame_mass', 'cell_mass', 'cell_h', 'ext_cooling', 'dur', 't_HP', 'cell_l',
            'ext_cool_mass', 'runawayJ', 'W', 'LH_PCM', 'rho_PCM', 'cell_s_l', 'q_max',
            'cell_s_h', 'L', 'frac_absorb', 'v_n_c', 'energy', 'cell_s_w', 'cell_w', 'missionJ',
            ('n_cpb', 'n_{cpk}'), ('n_bps', 'n_{kps}')])
        p1.setup()
        p1.model.list_inputs(prom_name=True)
        assert_match_spec(p1.model, 'Design_specs/PCM.json')


if __name__ == '__main__':
    unittest.main()

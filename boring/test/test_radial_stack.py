import unittest

import openmdao.api as om 
import numpy as np 
from boring.src.sizing.thermal_network import Radial_Stack, thermal_link
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal


class TestRadialStack(unittest.TestCase):

	def setUp(self):

		p = self.prob = om.Problem(model=om.Group())

		nn = 1

		p.model.add_subsystem('cond', Radial_Stack(num_nodes=nn, n_in=0, n_out=1, geom='flat'),
								promotes_inputs=['T_hp', 'XS:t_w', 'XS:t_wk', 'k_w', 'LW:A_inter',
												 'alpha'])#, 'epsilon', 'XS:A_wk', 'LW:L_flux', 'LW:L_adiabatic', 
												 # 'H', 'W'  ])
		p.model.add_subsystem('evap', Radial_Stack(num_nodes=nn, n_in=1, n_out=0, geom='flat'),
								promotes_inputs=['T_hp', 'XS:t_w', 'XS:t_wk', 'k_w', 'LW:A_inter', 
												 'alpha'])#, 'epsilon', 'XS:A_wk', 'LW:L_flux', 'LW:L_adiabatic', 
												 # 'H', 'W'  ])

		thermal_link(p.model, 'cond', 'evap', num_nodes=nn, geom='flat')



		p.model.connect('cond_bridge.k_wk', ['evap.k_wk', 'cond.k_wk'])

		p.setup()

		p['T_hp'] = 300
		p['XS:t_w'] = 0.0005
		p['XS:t_wk'] = 0.00069
		p['W'] = 0.02
		p['k_w'] = 11.4
		p['LW:A_inter'] = 0.0004
		p['alpha'] = 1
		p['epsilon'] = 0.46
		p['XS:A_w'] = 1E-5
		p['XS:A_wk'] = 1.38E-5
		p['LW:L_flux'] = .02
		p['LW:L_adiabatic'] = .03
		p['H'] = .02
		p['W'] = .02

		p.run_model()

	def test_resistance_outputs(self):

		assert_near_equal(self.prob.get_val('cond_bridge.axial.R_aw'), 438.5964912280702, tolerance=1E-1)
		assert_near_equal(self.prob.get_val('cond_bridge.axial.R_awk'), 562.4296959265857, tolerance=1E-1)
		assert_near_equal(self.prob.get_val('cond_bridge.axial.k_wk'), 6.442028989646447, tolerance=1E-1)

		assert_near_equal(self.prob.get_val('cond.radial.h_inter'), 1054975.2240304893, tolerance=1E-1)
		assert_near_equal(self.prob.get_val('cond.R_inter'), 0.0023697238978265794, tolerance=1E-1)
		assert_near_equal(self.prob.get_val('cond.R_w'), 0.10964912280701752, tolerance=1E-1)
		assert_near_equal(self.prob.get_val('cond.R_wk'), 0.2679540899786436, tolerance=1E-1)

		assert_near_equal(self.prob.get_val('cond_bridge.vapor.R_v'), 0.00020303930772787082, tolerance=1E-1)

if __name__ == '__main__':
    unittest.main()
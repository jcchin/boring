from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om
import numpy as np


class AxialThermalResistance(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('geom', values=['round', 'flat'], default='round')

    def setup(self):
        nn=self.options['num_nodes']

        self.add_input('k_w', np.ones(nn), units='W/(m*K)', desc='copper conductivity')
        self.add_input('k_wk', np.ones(nn), units='W/(m*K)', desc='Wick Conductivity')
        self.add_input('LW:L_eff', np.ones(nn), units='m', desc='Effective Length')
        self.add_input('XS:A_w', np.ones(nn), units='m**2', desc='Wall Area')
        self.add_input('XS:A_wk', np.ones(nn), units='m**2', desc='Wick Area')

        self.add_output('R_aw', val=np.ones(nn), units='K/W', desc='Wall Axial Resistance')
        self.add_output('R_awk', val=np.ones(nn), units='K/W', desc='Wick Axial Resistance')

    def setup_partials(self):
        nn=self.options['num_nodes']
        ar = np.arange(nn) 

        self.declare_partials('R_aw', ['LW:L_eff', 'XS:A_w', 'k_w'], rows=ar, cols=ar, val=0.)
        self.declare_partials('R_awk', ['LW:L_eff', 'XS:A_wk', 'k_wk'], rows=ar, cols=ar, val=0)

    def compute(self, inputs, outputs):

        geom = self.options['geom']

        k_w = inputs['k_w']
        k_wk = inputs['k_wk']
        L_eff = inputs['LW:L_eff']
        A_w = inputs['XS:A_w']
        A_wk = inputs['XS:A_wk']

        outputs['R_aw'] = L_eff/(A_w*k_w)
        outputs['R_awk'] = L_eff/(A_wk*k_wk)

    def compute_partials(self, inputs, J):

        geom = self.options['geom']

        k_w = inputs['k_w']
        k_wk = inputs['k_wk']
        L_eff = inputs['LW:L_eff']
        A_w = inputs['XS:A_w']
        A_wk = inputs['XS:A_wk']

        J['R_aw', 'LW:L_eff'] = 1/(A_w*k_w)
        J['R_aw', 'XS:A_w'] = -L_eff/(A_w**2*k_w)
        J['R_aw', 'k_w'] = -L_eff/(A_w*k_w**2)

        J['R_awk', 'LW:L_eff'] = 1/(A_wk*k_wk) #1/(A_wk*((1-epsilon)*k_w+epsilon*k_l))
        J['R_awk', 'XS:A_wk'] = -L_eff/(A_wk**2*k_wk) #-L_eff/(A_wk**2*((1-epsilon)*k_w+epsilon*k_l))
        #J['R_awk', 'epsilon'] = -L_eff/(A_wk*((1-epsilon)*k_w+epsilon*k_l)**2) * (-k_w + k_l)
        J['R_awk', 'k_wk'] = -L_eff/(A_w*k_wk**2)#-L_eff/(A_wk*((1-epsilon)*k_w+epsilon*k_l)**2) * (1-epsilon)
        #J['R_awk', 'k_l'] = -L_eff/(A_wk*((1-epsilon)*k_w+epsilon*k_l)**2) * epsilon


# # ------------ Derivative Checks --------------- #
if __name__ == '__main__':
    from openmdao.api import Problem

    nn = 1

    prob = Problem()
    prob.model.add_subsystem('comp1', AxialThermalResistance(num_nodes=nn), promotes_outputs=['*'], promotes_inputs=['*'])
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)

    print('R_aw = ', prob.get_val('comp1.R_aw'))
    print('R_awk = ', prob.get_val('comp1.R_awk'))
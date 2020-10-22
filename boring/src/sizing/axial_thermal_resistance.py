from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om

class AxialThermalResistance(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn=self.options['num_nodes']

        self.add_input('epsilon', 1, desc='')
        self.add_input('k_w', 1, desc='')
        self.add_input('k_l', 1, desc='')
        self.add_input('L_adiabatic', 1, units='m', desc='')
        self.add_input('A_w', 1, units='m**2', desc='')
        self.add_input('A_wk', 1, units='m**2', desc='')

        self.add_output('k_wk', 1, desc='')
        self.add_output('R_aw', 1, desc='')
        self.add_output('R_awk', 1, desc='')

        self.declare_partials('k_wk', ['epsilon', 'k_w', 'k_l'])
        self.declare_partials('R_aw', ['L_adiabatic', 'A_w', 'k_w'])
        self.declare_partials('R_awk', ['L_adiabatic', 'A_wk', 'epsilon', 'k_w', 'k_l'])

    def compute(self,inputs,outputs):

        epsilon = inputs['epsilon']
        k_w = inputs['k_w']
        k_l = inputs['k_l']
        L_adiabatic = inputs['L_adiabatic']
        A_w = inputs['A_w']
        k_w = inputs['k_w']
        A_wk = inputs['A_wk']

        outputs['k_wk']=(1-epsilon)*k_w+epsilon*k_l
        outputs['R_aw']=L_adiabatic/(A_w*k_w)
        outputs['R_awk']=L_adiabatic/(A_wk*outputs['k_wk'])

    def compute_partials(self,inputs,J):

        epsilon = inputs['epsilon']
        k_w = inputs['k_w']
        k_l = inputs['k_l']
        L_adiabatic = inputs['L_adiabatic']
        A_w = inputs['A_w']
        k_w = inputs['k_w']
        A_wk = inputs['A_wk']

        d_k_wk__d_epsilon = -k_w + k_l

        J['k_wk', 'epsilon'] = -k_w + k_l
        J['k_wk', 'k_w'] = (1 - epsilon)
        J['k_wk', 'k_l'] = epsilon

        J['R_aw', 'L_adiabatic'] = 1/(A_w*k_w) 
        J['R_aw', 'A_w'] = -L_adiabatic/(A_w**2*k_w)
        J['R_aw', 'k_w'] = -L_adiabatic/(A_w*k_w**2)

        J['R_awk', 'L_adiabatic'] = 1/(A_w*((1-epsilon)*k_w+epsilon*k_l))
        J['R_awk', 'A_wk'] = -L_adiabatic/(A_wk**2*((1-epsilon)*k_w+epsilon*k_l))
        J['R_awk', 'epsilon'] = -L_adiabatic/(A_wk*((1-epsilon)*k_w+epsilon*k_l)**2) * (-k_w + k_l)
        J['R_awk', 'k_w'] = -L_adiabatic/(A_wk*((1-epsilon)*k_w+epsilon*k_l)**2) * (1-epsilon)
        J['R_awk', 'k_l'] = -L_adiabatic/(A_wk*((1-epsilon)*k_w+epsilon*k_l)**2) * epsilon


# # ------------ Derivative Checks --------------- #
if __name__ == '__main__':
    from openmdao.api import Problem

    nn = 1

    prob = Problem()
    prob.model.add_subsystem('comp1', AxialThermalResistance(num_nodes=nn), promotes=['*'])
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)
from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om
import numpy as np


class AxialThermalResistance(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn=self.options['num_nodes']

        # Wall and Wick
        self.add_input('k_w', np.ones(nn), units='W/(m*K)', desc='wall thermal conductivity')
        self.add_input('k_wk', np.ones(nn), units='W/(m*K)', desc='wick thermal conductivity')
        self.add_input('LW:L_eff', np.ones(nn), units='m', desc='Effective Length')
        self.add_input('XS:A_w', np.ones(nn), units='m**2', desc='Wall Area')
        self.add_input('XS:A_wk', np.ones(nn), units='m**2', desc='Wick Area')
        # Vapor Channel
        self.add_input('XS:r_h', np.ones(nn), units='m', desc='hydraulic radius')
        self.add_input('R_g', 0.2 * np.ones(nn), units='J/kg/K', desc='gas constant of the vapor')
        self.add_input('mu_v', 0.03 * np.ones(nn), units='N*s/m**2', desc='vapor viscosity')
        self.add_input('T_hp', 300 * np.ones(nn), units='K', desc='Temp of heat pipe')
        self.add_input('h_fg', 100 * np.ones(nn), units='J/kg', desc='latent heat')
        self.add_input('P_v', 1000 * np.ones(nn), units='Pa', desc='pressure')
        self.add_input('rho_v', 100 * np.ones(nn), units='kg/m**3', desc='density of vapor')

        self.add_output('R_aw', val=np.ones(nn), units='K/W', desc='Wall Axial Resistance')
        self.add_output('R_awk', val=np.ones(nn), units='K/W', desc='Wick Axial Resistance')
        self.add_output('R_v', val=1.0 * np.ones(nn), units='K/W', desc='thermal resistance of vapor region')

    def setup_partials(self):
        nn=self.options['num_nodes']
        ar = np.arange(nn) 

        self.declare_partials('R_aw', ['LW:L_eff', 'XS:A_w', 'k_w'], rows=ar, cols=ar, val=0.)
        self.declare_partials('R_awk', ['LW:L_eff', 'XS:A_wk', 'k_wk'], rows=ar, cols=ar, val=0)
        self.declare_partials('R_v', ['R_g', 'mu_v', 'T_hp', 'h_fg', 'P_v', 'rho_v', 'LW:L_eff', 'XS:r_h'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        k_w = inputs['k_w']
        k_wk = inputs['k_wk']
        L_eff = inputs['LW:L_eff']
        A_w = inputs['XS:A_w']
        A_wk = inputs['XS:A_wk']

        R_g = inputs['R_g']
        mu_v = inputs['mu_v']
        T_hp = inputs['T_hp']
        h_fg = inputs['h_fg']
        P_v = inputs['P_v']
        rho_v = inputs['rho_v']
        r_h = inputs['XS:r_h']


        outputs['R_aw'] = L_eff/(A_w*k_w)
        outputs['R_awk'] = L_eff/(A_wk*k_wk)
        outputs['R_v'] = L_eff * 8 * R_g * mu_v * T_hp ** 2 / (np.pi * h_fg ** 2 * P_v * rho_v * (r_h ** 4)) 


    def compute_partials(self, inputs, J):

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

        R_g = inputs['R_g']
        mu_v = inputs['mu_v']
        T_hp = inputs['T_hp']
        h_fg = inputs['h_fg']
        P_v = inputs['P_v']
        rho_v = inputs['rho_v']
        r_h = inputs['XS:r_h']

        R_v = L_eff * 8 * R_g * mu_v * T_hp ** 2 / (np.pi * h_fg ** 2 * P_v * rho_v * (r_h ** 4))

        J['R_v', 'LW:L_eff'] = R_v / L_eff
        J['R_v', 'R_g'] = R_v / R_g
        J['R_v', 'mu_v'] = R_v / mu_v
        J['R_v', 'T_hp'] = 2 * R_v / T_hp
        J['R_v', 'h_fg'] = -2 * R_v / h_fg
        J['R_v', 'P_v'] = -R_v / P_v
        J['R_v', 'rho_v'] = -R_v / rho_v
        J['R_v', 'XS:r_h'] = -4 * R_v / r_h

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
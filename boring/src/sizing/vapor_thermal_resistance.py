from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om


class VapThermResComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn=self.options['num_nodes']
        
        self.add_input('D_v',0.00362, units='m', desc='diameter of vapor region')
        self.add_input('R_g', 1, units='J/kg/K', desc='gas constant of the vapor')
        self.add_input('mu_v', 1,units='N*s/m**2', desc='vapor viscosity')
        self.add_input('T_hp', 1, units='K', desc='Temp of heat pipe')
        self.add_input('h_fg', 1, units='J/kg', desc='latent heat')
        self.add_input('P_v', 1, units='Pa', desc='pressure')
        self.add_input('rho_v', 1, units='kg/m**3', desc='density of vapor')
        self.add_input('L_eff', 1, units='m', desc='effective length')

        self.add_output('r_h', units='m', desc='hydraulic radius')
        self.add_output('R_v', units='K/W', desc='thermal resistance of vapor region')


    def setup_partials(self):
        self.declare_partials('*', '*')#, method='fd')


    def compute(self, inputs, outputs):
        D_v = inputs['D_v']
        R_g = inputs['R_g']
        mu_v = inputs['mu_v']
        T_hp = inputs['T_hp']
        h_fg = inputs['h_fg']
        P_v = inputs['P_v']
        rho_v = inputs['rho_v']
        L_eff = inputs['L_eff']

        outputs['r_h'] = D_v/2
        outputs['R_v'] = 8*R_g*mu_v*T_hp**2/(np.pi*h_fg**2*P_v*rho_v)*(L_eff/(outputs['r_h']**4))                     

    def compute_partials(self, inputs, partials):
        D_v = inputs['D_v']
        R_g = inputs['R_g']
        mu_v = inputs['mu_v']
        T_hp = inputs['T_hp']
        h_fg = inputs['h_fg']
        P_v = inputs['P_v']
        rho_v = inputs['rho_v']
        L_eff = inputs['L_eff']
        r_h = D_v/2
        
        partials['r_h', 'D_v'] = 1/2
        
        partials['R_v', 'R_g'] = 8*mu_v*T_hp**2*L_eff/(np.pi*h_fg**2*P_v*rho_v*r_h**4)
        partials['R_v', 'mu_v'] = 8*R_g*T_hp**2*L_eff/(np.pi*h_fg**2*P_v*rho_v*r_h**4)
        partials['R_v', 'T_hp'] = 2*8*mu_v*R_g*T_hp*L_eff/(np.pi*h_fg**2*P_v*rho_v*r_h**4)
        partials['R_v', 'h_fg'] = -2*8*R_g*T_hp**2*L_eff/(np.pi*h_fg**3*P_v*rho_v*r_h**4)
        partials['R_v', 'P_v'] = -8*R_g*mu_v*T_hp**2*L_eff/(np.pi*h_fg**2*P_v**2*rho_v*r_h**4)    
        partials['R_v', 'rho_v'] = -8*R_g*mu_v*T_hp**2*L_eff/(np.pi*h_fg**2*P_v*rho_v**2*r_h**4)
        partials['R_v', 'L_eff'] = 8*R_g*mu_v*T_hp**2/(np.pi*h_fg**2*P_v*rho_v*r_h**4)
        partials['R_v', 'D_v'] = -4*8*R_g*mu_v*T_hp**2*L_eff/(np.pi*h_fg**2*P_v*rho_v*r_h**5)*1/2
        
        # Formula with no partials applied:
        # partials['R_v', 'x'] = 8*R_g*mu_v*T_hp**2*L_eff/(np.pi*h_fg**2*P_v*rho_v*r_h**4)
                
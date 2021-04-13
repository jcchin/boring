from __future__ import absolute_import
import numpy as np

import openmdao.api as om


class VaporThermalResistance(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('geom', values=['round', 'flat'], default='round')


    def setup(self):
        nn = self.options['num_nodes']
        geom = self.options['geom']

        if geom.lower() == 'round':
            self.add_input('D_v', 0.1 * np.ones(nn), units='m', desc='diameter of vapor region')

        elif geom.lower() == 'flat':
            self.add_input('H', 0.02 * np.ones(nn), units='m', desc='total thickness of heat pipe')
            self.add_input('W', 0.02 * np.ones(nn), units='m', desc='width of heat pipe into the page')
            self.add_input('t_w', 0.02 * np.ones(nn), units='m', desc='wall thickness')
            self.add_input('t_wk', 0.02 * np.ones(nn), units='m', desc='wick thickness')

        else:
            pass

        self.add_input('R_g', 0.2 * np.ones(nn), units='J/kg/K', desc='gas constant of the vapor')
        self.add_input('mu_v', 0.03 * np.ones(nn), units='N*s/m**2', desc='vapor viscosity')
        self.add_input('T_hp', 300 * np.ones(nn), units='K', desc='Temp of heat pipe')
        self.add_input('h_fg', 100 * np.ones(nn), units='J/kg', desc='latent heat')
        self.add_input('P_v', 1000 * np.ones(nn), units='Pa', desc='pressure')
        self.add_input('rho_v', 100 * np.ones(nn), units='kg/m**3', desc='density of vapor')
        self.add_input('L_flux', val=1*np.ones(nn), units='m', desc='length of cells')
        self.add_input('L_adiabatic', val=1*np.ones(nn), units='m', desc = 'length of adiabatic section')

        self.add_output('r_h', val=1.0 * np.ones(nn), units='m', desc='hydraulic radius')
        self.add_output('R_v', val=1.0 * np.ones(nn), units='K/W', desc='thermal resistance of vapor region')

    def setup_partials(self):

        nn = self.options['num_nodes']
        ar = np.arange(nn)
        geom = self.options['geom']

        if geom.lower() == 'round':
            self.declare_partials('r_h', 'D_v', rows=ar, cols=ar)
            self.declare_partials('R_v', 'D_v', rows=ar, cols=ar)

        elif geom.lower() == 'flat':
            self.declare_partials('r_h', ['H', 't_w', 't_wk', 'W'], rows=ar, cols=ar) 
            self.declare_partials('R_v', ['H', 't_w', 't_wk', 'W'], rows=ar, cols=ar)   

        else:
            pass

        self.declare_partials('R_v', ['R_g', 'mu_v', 'T_hp', 'h_fg', 'P_v', 'rho_v', 'L_adiabatic', 'L_flux'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        geom = self.options['geom']

        R_g = inputs['R_g']
        mu_v = inputs['mu_v']
        T_hp = inputs['T_hp']
        h_fg = inputs['h_fg']
        P_v = inputs['P_v']
        rho_v = inputs['rho_v']
        L_adiabatic = inputs['L_adiabatic']
        L_flux = inputs['L_flux']

        if geom.lower() == 'round':
            D_v = inputs['D_v']
            outputs['r_h'] = D_v / 2

        elif geom.lower() == 'flat':
            H = inputs['H']
            W = inputs['W']
            t_w = inputs['t_w']
            t_wk = inputs['t_wk']

            outputs['r_h'] = ((H-2*t_w-2*t_wk)*W)/(2*W+2*(H-2*t_w-2*t_wk))

        else: 
            pass

        L_eff = L_flux+L_adiabatic
        outputs['R_v'] = 8 * R_g * mu_v * T_hp ** 2 / (np.pi * h_fg ** 2 * P_v * rho_v) * (
                    L_eff / (outputs['r_h'] ** 4))

    def compute_partials(self, inputs, partials):
        geom = self.options['geom']

        R_g = inputs['R_g']
        mu_v = inputs['mu_v']
        T_hp = inputs['T_hp']
        h_fg = inputs['h_fg']
        P_v = inputs['P_v']
        rho_v = inputs['rho_v']
        L_adiabatic = inputs['L_adiabatic']
        L_flux = inputs['L_flux']
        L_eff = L_flux+L_adiabatic

        if geom.lower() == 'round':

            D_v = inputs['D_v']

            r_h = D_v / 2

            partials['r_h', 'D_v'] = 1 / 2
            partials['R_v', 'D_v'] = -4 * 8 * R_g * mu_v * T_hp ** 2 * L_eff / (
                        np.pi * h_fg ** 2 * P_v * rho_v * r_h ** 5) * 1 / 2

        elif geom.lower() == 'flat':

            H = inputs['H']
            W = inputs['W']
            t_w = inputs['t_w']
            t_wk = inputs['t_wk']

            r_h = ((H-2*t_w-2*t_wk)*W)/(2*W+2*(H-2*t_w-2*t_wk))

            dr_h_dH = partials['r_h', 'H'] = (W * (2*W+2*(H-2*t_w-2*t_wk)) - 2 * ((H-2*t_w-2*t_wk)*W)) / (2*W+2*(H-2*t_w-2*t_wk))**2
            dr_h_dW = partials['r_h', 'W'] = ((H-2*t_w-2*t_wk) * (2*W+2*(H-2*t_w-2*t_wk)) - 2 * ((H-2*t_w-2*t_wk)*W))/(2*W+2*(H-2*t_w-2*t_wk))**2
            dr_h_dt_w = partials['r_h', 't_w'] = (-2*W * (2*W+2*(H-2*t_w-2*t_wk)) + 4*((H-2*t_w-2*t_wk)*W))/(2*W+2*(H-2*t_w-2*t_wk))**2
            dr_h_dt_wk = partials['r_h', 't_wk'] = (-2*W * (2*W+2*(H-2*t_w-2*t_wk)) + 4*((H-2*t_w-2*t_wk)*W))/(2*W+2*(H-2*t_w-2*t_wk))**2

            partials['R_v', 'H'] = - 4 * 8 * R_g * mu_v * T_hp ** 2 / (np.pi * h_fg ** 2 * P_v * rho_v) * (L_eff / (r_h ** 5)) * dr_h_dH
            partials['R_v', 'W'] = - 4 * 8 * R_g * mu_v * T_hp ** 2 / (np.pi * h_fg ** 2 * P_v * rho_v) * (L_eff / (r_h ** 5)) * dr_h_dW
            partials['R_v', 't_w'] = - 4 * 8 * R_g * mu_v * T_hp ** 2 / (np.pi * h_fg ** 2 * P_v * rho_v) * (L_eff / (r_h ** 5)) * dr_h_dt_w
            partials['R_v', 't_wk'] = - 4 * 8 * R_g * mu_v * T_hp ** 2 / (np.pi * h_fg ** 2 * P_v * rho_v) * (L_eff / (r_h ** 5)) * dr_h_dt_wk

        else: 
            pass

        partials['R_v', 'R_g'] = 8 * mu_v * T_hp ** 2 * L_eff / (np.pi * h_fg ** 2 * P_v * rho_v * r_h ** 4)
        partials['R_v', 'mu_v'] = 8 * R_g * T_hp ** 2 * L_eff / (np.pi * h_fg ** 2 * P_v * rho_v * r_h ** 4)
        partials['R_v', 'T_hp'] = 2 * 8 * mu_v * R_g * T_hp * L_eff / (np.pi * h_fg ** 2 * P_v * rho_v * r_h ** 4)
        partials['R_v', 'h_fg'] = -2 * 8 * R_g * mu_v * T_hp ** 2 * L_eff / (np.pi * h_fg ** 3 * P_v * rho_v * r_h ** 4)
        partials['R_v', 'P_v'] = -8 * R_g * mu_v * T_hp ** 2 * L_eff / (np.pi * h_fg ** 2 * P_v ** 2 * rho_v * r_h ** 4)
        partials['R_v', 'rho_v'] = -8 * R_g * mu_v * T_hp ** 2 * L_eff / (
                    np.pi * h_fg ** 2 * P_v * rho_v ** 2 * r_h ** 4)
        partials['R_v', 'L_adiabatic'] = 8 * R_g * mu_v * T_hp ** 2 / (np.pi * h_fg ** 2 * P_v * rho_v * r_h ** 4)
        partials['R_v', 'L_flux'] = 8 * R_g * mu_v * T_hp ** 2 / (np.pi * h_fg ** 2 * P_v * rho_v * r_h ** 4)



# # ------------ Derivative Checks --------------- #
if __name__ == '__main__':
    from openmdao.api import Problem

    nn = 10

    prob = Problem()
    prob.model.add_subsystem('comp1', VaporThermalResistance(num_nodes=nn), promotes_outputs=['*'], promotes_inputs=['*'])
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)

    # print('k_wk = ', prob.get_val('comp1.k_wk'))
    # print('R_aw = ', prob.get_val('comp1.R_aw'))
    # print('R_awk = ', prob.get_val('comp1.R_awk'))

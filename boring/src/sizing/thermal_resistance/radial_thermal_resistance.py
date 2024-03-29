import numpy as np

import openmdao.api as om


class RadialThermalResistance(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('geom', values=['round', 'flat'], default='round')

    def setup(self):
        nn = self.options['num_nodes']
        geom = self.options['geom']

        self.add_input('alpha', np.ones(nn), desc='thermal accommodation coefficient typically 0.01 to 1')
        self.add_input('h_fg', np.ones(nn), units='J/kg', desc='latent heat')
        self.add_input('T_hp', np.ones(nn), units='K', desc='Temp of heat pipe')
        self.add_input('v_fg', np.ones(nn), units='m**3/kg', desc='specific volume')
        self.add_input('R_g', np.ones(nn), units='J/kg/K', desc='gas constant of the vapor')
        self.add_input('P_v', np.ones(nn), units='Pa', desc='pressure')
        self.add_input('k_w', np.ones(nn), units='W/(m*K)', desc='thermal conductivity of the wall')
        self.add_input('k_l', np.ones(nn), units='W/(m*K)', desc='liquid conductivity')
        self.add_input('LW:A_inter', np.ones(nn), units='m**2', desc='area of wick/vapor interface of the condenser/evaporator')
        self.add_input('epsilon', np.ones(nn), desc='wick porosity')

        if geom == 'round':
            self.add_input('XS:D_v', np.ones(nn), units='m', desc='diameter of vapor region')
            self.add_input('XS:D_od', np.ones(nn), units='m', desc='outer diameter')
            self.add_input('XS:r_i', np.ones(nn), units='m', desc='inner radius')
            self.add_input('LW:L_flux', np.ones(nn), units='m', desc='length of condensor/evaporator')

        elif geom == 'flat':
            self.add_input('XS:t_w', np.ones(nn), units='m', desc='wall thickness')
            self.add_input('XS:t_wk', np.ones(nn), units='m', desc='wick thickness')

        self.add_output('k_wk', np.ones(nn), units='W/(m*K)', desc='thermal condusctivity of the wick')
        self.add_output('h_inter', np.ones(nn), units='W/(m**2/K)', desc='HTC of wick/vapor interface of the condenser/evaporator')
        self.add_output('R_w', np.ones(nn), units='K/W', desc='thermal resistance')
        self.add_output('R_wk', np.ones(nn), units='K/W', desc='thermal resistance')
        self.add_output('R_inter', np.ones(nn), units='K/W', desc='thermal resistance of wick/vapor interface of the condenser/evaporator')

    def setup_partials(self):
        nn = self.options['num_nodes']
        geom = self.options['geom']

        ar = np.arange(nn)

        self.declare_partials('k_wk', ['epsilon', 'k_w', 'k_l'], rows=ar, cols=ar)
        self.declare_partials('h_inter', ['alpha', 'h_fg', 'T_hp', 'v_fg', 'R_g', 'P_v'], rows=ar, cols=ar)
        self.declare_partials('R_inter', ['alpha', 'h_fg', 'T_hp', 'v_fg', 'R_g', 'P_v', 'LW:A_inter'], rows=ar, cols=ar)

        if geom == 'round':
            self.declare_partials('R_w', ['XS:D_od', 'XS:r_i', 'k_w', 'LW:L_flux'], rows=ar, cols=ar)
            self.declare_partials('R_wk', ['XS:D_v', 'XS:r_i', 'k_w', 'k_l', 'epsilon', 'LW:L_flux'], rows=ar, cols=ar)

        elif geom == 'flat':  
            self.declare_partials('R_w', ['XS:t_w', 'k_w', 'LW:A_inter'], rows=ar, cols=ar)
            self.declare_partials('R_wk', ['XS:t_wk', 'k_w', 'k_l', 'epsilon', 'LW:A_inter'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        geom = self.options['geom']

        alpha = inputs['alpha'] # Look into this, need better way to determine this rather than referencing papers.
        h_fg = inputs['h_fg']
        T_hp = inputs['T_hp']
        v_fg = inputs['v_fg']
        R_g = inputs['R_g']
        P_v = inputs['P_v']
        epsilon = inputs['epsilon']
        k_l = inputs['k_l']
        k_w = inputs['k_w']
        A_inter = inputs['LW:A_inter']

        k_wk = outputs['k_wk'] = (1-epsilon)*k_w+epsilon*k_l
          
        h_inter = outputs['h_inter'] = 2 * alpha / (2 - alpha) * (h_fg ** 2 / (T_hp * v_fg)) * np.sqrt(
            1 / (2 * np.pi * R_g * T_hp)) * (1 - P_v * v_fg / (2 * h_fg))
        outputs['R_inter'] = 1 / (h_inter * A_inter)

        if geom == 'round':
            D_od = inputs['XS:D_od']
            r_i = inputs['XS:r_i']
            L_flux = inputs['LW:L_flux']
            D_v = inputs['XS:D_v']
            outputs['R_w'] = np.log((D_od / 2) / (r_i)) / (2 * np.pi * k_w * L_flux)
            outputs['R_wk'] = np.log((r_i) / (D_v / 2)) / (2 * np.pi * k_wk * L_flux)

        elif geom == 'flat':
            t_w = inputs['XS:t_w']
            t_wk = inputs['XS:t_wk']
            outputs['R_w'] = inputs['XS:t_w']/(k_w*A_inter)
            outputs['R_wk'] = inputs['XS:t_wk']/(k_wk*A_inter)

    def compute_partials(self, inputs, J):
        geom = self.options['geom']

        alpha = inputs['alpha']
        h_fg = inputs['h_fg']
        T_hp = inputs['T_hp']
        v_fg = inputs['v_fg']
        R_g = inputs['R_g']
        P_v = inputs['P_v']
        k_w = inputs['k_w']
        k_l = inputs['k_l']
        epsilon = inputs['epsilon']
        A_inter = inputs['LW:A_inter']

        k_wk = (1-epsilon)*k_w+epsilon*k_l

        h_inter = 2 * alpha / (2 - alpha) * (h_fg ** 2 / (T_hp * v_fg)) * np.sqrt(1 / (2 * np.pi * R_g * T_hp)) * (
            1 - P_v * v_fg / (2 * h_fg))

        dh_dalpha = J['h_inter', 'alpha'] = 4 / (2 - alpha) ** 2 * (h_fg ** 2 / (T_hp * v_fg)) * np.sqrt(
            1 / (2 * np.pi * R_g * T_hp)) * (1 - P_v * v_fg / (2 * h_fg))
        dh_dhfg = J['h_inter', 'h_fg'] = 4 * alpha * h_fg / ((2 - alpha) * T_hp * v_fg) * np.sqrt(
            1 / (2 * np.pi * R_g * T_hp)) * (1 - P_v * v_fg / (2 * h_fg)) + (2 * alpha / (2 - alpha) * (
                    h_fg ** 2 / (T_hp * v_fg)) * np.sqrt(1 / (2 * np.pi * R_g * T_hp)) * (P_v * v_fg / (2 * h_fg ** 2)))
        dh_dThp = J['h_inter', 'T_hp'] = (-2 * alpha * h_fg ** 2) / ((2 - alpha) * T_hp ** 2 * v_fg) * np.sqrt(
            1 / (2 * np.pi * R_g * T_hp)) * (1 - P_v * v_fg / (2 * h_fg)) + (2 * alpha * h_fg ** 2 / (
                    (2 - alpha) * T_hp * v_fg) * 0.5 * (1 / (2 * np.pi * R_g * T_hp)) ** -0.5 * (-1 / (
                    2 * np.pi * R_g * T_hp ** 2)) * (1 - P_v * v_fg / (2 * h_fg)))
        dh_dvfg = J['h_inter', 'v_fg'] = (-2 * alpha * h_fg ** 2) / ((2 - alpha) * T_hp * v_fg ** 2) * np.sqrt(
            1 / (2 * np.pi * R_g * T_hp)) * (1 - P_v * v_fg / (2 * h_fg)) + ((2 * alpha * h_fg ** 2 / (
                    (2 - alpha) * T_hp * v_fg)) * np.sqrt(1 / (2 * np.pi * R_g * T_hp)) * (-P_v / (2 * h_fg)))
        dh_dRg = J['h_inter', 'R_g'] = 2 * alpha / (2 - alpha) * (h_fg ** 2 / (T_hp * v_fg)) * 0.5 * (
                    1 / (2 * np.pi * R_g * T_hp)) ** (-0.5) * (-1 / (2 * np.pi * R_g ** 2 * T_hp)) * (
                                                          1 - P_v * v_fg / (2 * h_fg))
        dh_dPv = J['h_inter', 'P_v'] = 2 * alpha / (2 - alpha) * (h_fg ** 2 / (T_hp * v_fg)) * np.sqrt(
            1 / (2 * np.pi * R_g * T_hp)) * (-v_fg / (2 * h_fg))

        J['k_wk', 'epsilon'] = -k_w + k_l
        J['k_wk', 'k_w'] = (1 - epsilon)
        J['k_wk', 'k_l'] = epsilon
        J['R_inter', 'alpha'] = -dh_dalpha / (h_inter ** 2 * A_inter)
        J['R_inter', 'h_fg'] = -dh_dhfg / (h_inter ** 2 * A_inter)
        J['R_inter', 'T_hp'] = -dh_dThp / (h_inter ** 2 * A_inter)
        J['R_inter', 'v_fg'] = -dh_dvfg / (h_inter ** 2 * A_inter)
        J['R_inter', 'R_g'] = -dh_dRg / (h_inter ** 2 * A_inter)
        J['R_inter', 'P_v'] = -dh_dPv / (h_inter ** 2 * A_inter)
        J['R_inter', 'LW:A_inter'] = -1 / (h_inter * A_inter ** 2)

        if geom == 'round':
            D_od = inputs['XS:D_od']
            r_i = inputs['XS:r_i']
            L_flux = inputs['LW:L_flux']
            D_v = inputs['XS:D_v']

            J['R_w', 'XS:D_od'] = 1 / (D_od * (2 * np.pi * k_w * L_flux))
            J['R_w', 'XS:r_i'] = -1 / (2 * np.pi * k_w * L_flux)
            J['R_w', 'k_w'] = -1 * np.log((D_od / 2) / (r_i)) / (2 * np.pi * k_w ** 2 * L_flux)
            J['R_w', 'LW:L_flux'] = -1 * np.log((D_od / 2) / (r_i)) / (2 * np.pi * k_w * L_flux ** 2)

            J['R_wk', 'XS:r_i'] = 1 / (r_i * (2 * np.pi * k_wk * L_flux))
            J['R_wk', 'XS:D_v'] = -1 / (D_v * (2 * np.pi * k_wk * L_flux))
            dR_wk_dk_wk = -1 * np.log((r_i) / (D_v / 2)) / (2 * np.pi * k_wk ** 2 * L_flux)
            J['R_wk', 'k_w'] = dR_wk_dk_wk*J['k_wk', 'k_w']
            J['R_wk', 'k_l'] = dR_wk_dk_wk*J['k_wk', 'k_l']
            J['R_wk', 'epsilon'] = dR_wk_dk_wk*J['k_wk', 'epsilon']
            J['R_wk', 'LW:L_flux'] = -1 * np.log((r_i) / (D_v / 2)) / (2 * np.pi * k_wk * L_flux ** 2)

        elif geom == 'flat':
            t_w = inputs['XS:t_w']
            t_wk = inputs['XS:t_wk']

            J['R_w', 'XS:t_w'] = 1/(k_w*A_inter)
            J['R_w', 'LW:A_inter'] = - t_w / (k_w * A_inter**2)
            J['R_wk', 'XS:t_wk'] = 1/(k_wk*A_inter)
            #J['R_wk', 'k_wk'] = - t_wk / (k_wk**2 * A_inter)
            J['R_w', 'k_w'] = - t_w / (k_w**2 * A_inter)
            J['R_wk', 'k_l'] = -t_wk/(A_inter*((1-epsilon)*k_w+epsilon*k_l)**2) * epsilon
            J['R_wk', 'epsilon'] = -t_wk/(A_inter*((1-epsilon)*k_w+epsilon*k_l)**2) * (-k_w + k_l)
            J['R_wk', 'LW:A_inter'] = - t_wk / (k_wk * A_inter**2)


if __name__ == '__main__':
    from openmdao.api import Problem

    nn = 10

    prob = Problem()
    prob.model.add_subsystem('comp1', RadialThermalResistance(num_nodes=nn), promotes_outputs=['*'], promotes_inputs=['*'])
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    om.view_connections(prob)

        



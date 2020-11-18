import numpy as np

import openmdao.api as om

class RadialThermalResistance(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn=self.options['num_nodes']

        self.add_input('alpha', val=1.0*np.ones(nn), desc='thermal accommodation coefficient typically 0.01 to 1')
        self.add_input('h_fg', val=1.0*np.ones(nn), units='J/kg', desc='latent heat')
        self.add_input('T_hp', val=1.0*np.ones(nn), units='K', desc='Temp of heat pipe')
        self.add_input('v_fg', val=1.0*np.ones(nn), units='m**3/kg', desc='specific volume')
        self.add_input('R_g', val=1.0*np.ones(nn), units='J/kg/K', desc='gas constant of the vapor')
        self.add_input('P_v', val=1.0*np.ones(nn), units='Pa', desc='pressure')
        self.add_input('D_od', val=1.0*np.ones(nn), units='m', desc='outer diameter')
        self.add_input('r_i', val=1.0*np.ones(nn), units='m', desc='inner radius' )
        self.add_input('k_w', val=1.0*np.ones(nn), units='W/(m*K)', desc='thermal conductivity of the wall')
        self.add_input('L_flux', val=1.0*np.ones(nn), units='m', desc='length of condensor/evaporator')
        self.add_input('D_v', val=1.0*np.ones(nn), units='m', desc='diameter of vapor region')
        self.add_input('k_wk', val=1.0*np.ones(nn), units='W/(m*K)', desc='thermal condusctivity of the wick')
        self.add_input('A_inter', val=1.0*np.ones(nn), units='m**2', desc='area of wick/vapor interface of the condenser/evaporator')

        self.add_output('h_inter', val=1.0*np.ones(nn), units='W/(m**2/K)', desc='HTC of wick/vapor interface of the condenser/evaporator')
        self.add_output('R_w', val=1.0*np.ones(nn), units='K/W', desc='thermal resistance')
        self.add_output('R_wk', val=1.0*np.ones(nn), units='K/W', desc='thermal resistance')
        self.add_output('R_inter', val=1.0*np.ones(nn), units='K/W', desc='thermal resistance of wick/vapor interface of the condenser/evaporator')

    def setup_partials(self):
        nn=self.options['num_nodes']
        ar = np.arange(nn) 

        self.declare_partials('h_inter', ['alpha', 'h_fg', 'T_hp', 'v_fg', 'R_g', 'P_v'], rows=ar, cols=ar)
        self.declare_partials('R_w', ['D_od', 'r_i', 'k_w', 'L_flux'], rows=ar, cols=ar)
        self.declare_partials('R_wk', ['D_v', 'r_i', 'k_wk', 'L_flux'], rows=ar, cols=ar)
        self.declare_partials('R_inter', ['alpha', 'h_fg', 'T_hp', 'v_fg', 'R_g', 'P_v', 'A_inter'], rows=ar, cols=ar)


    def compute(self, inputs, outputs):

        alpha = inputs['alpha']
        h_fg= inputs['h_fg']
        T_hp = inputs['T_hp']
        v_fg = inputs['v_fg']
        R_g = inputs['R_g']
        P_v = inputs['P_v']
        D_od = inputs['D_od']
        r_i = inputs['r_i']
        k_w = inputs['k_w']
        L_flux = inputs['L_flux']
        D_v = inputs['D_v']
        k_wk = inputs['k_wk']
        A_inter = inputs['A_inter']

        #alpha=1           # Look into this, need better way to determine this rather than referencing papers.
        h_inter=outputs['h_inter'] = 2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg))
        print("hinter,",h_inter)
        outputs['R_w'] = np.log((D_od/2)/(r_i))/(2*np.pi*k_w*L_flux) 
        outputs['R_wk'] = np.log((r_i)/(D_v/2))/(2*np.pi*k_wk*L_flux) 
        outputs['R_inter'] = 1/(h_inter*A_inter) 

    def compute_partials(self, inputs, partials):
        alpha = inputs['alpha']
        h_fg= inputs['h_fg']
        T_hp = inputs['T_hp']
        v_fg = inputs['v_fg']
        R_g = inputs['R_g']
        P_v = inputs['P_v']
        D_od = inputs['D_od']
        r_i = inputs['r_i']
        k_w = inputs['k_w']
        L_flux = inputs['L_flux']
        D_v = inputs['D_v']
        k_wk = inputs['k_wk']
        A_inter = inputs['A_inter']

        h_inter= 2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg))

        dh_dalpha = partials['h_inter', 'alpha'] = 4/(2-alpha)**2 * (h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg)) 
        dh_dhfg = partials['h_inter', 'h_fg'] = 4*alpha*h_fg/((2-alpha)*T_hp*v_fg) * np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg)) + (2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(P_v*v_fg/(2*h_fg**2)))
        dh_dThp = partials['h_inter', 'T_hp'] = (-2*alpha*h_fg**2)/((2-alpha)*T_hp**2*v_fg)*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg)) + (2*alpha*h_fg**2/((2-alpha)*T_hp*v_fg) * 0.5*(1/(2*np.pi*R_g*T_hp))**-0.5 * (-1/(2*np.pi*R_g*T_hp**2))*(1-P_v*v_fg/(2*h_fg)))
        dh_dvfg = partials['h_inter', 'v_fg'] = (-2*alpha*h_fg**2)/((2-alpha)*T_hp*v_fg**2)*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg)) + ((2*alpha*h_fg**2/((2-alpha)*T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(-P_v/(2*h_fg)))
        dh_dRg = partials['h_inter', 'R_g'] = 2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*0.5*(1/(2*np.pi*R_g*T_hp))**(-0.5)*(-1/(2*np.pi*R_g**2*T_hp))*(1-P_v*v_fg/(2*h_fg)) 
        dh_dPv = partials['h_inter', 'P_v'] = 2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(-v_fg/(2*h_fg))

        partials['R_w', 'D_od'] = 1/(D_od*(2*np.pi*k_w*L_flux)) 
        partials['R_w', 'r_i'] = -1/(2*np.pi*k_w*L_flux)
        partials['R_w', 'k_w'] = -1 * np.log((D_od/2)/(r_i))/(2*np.pi*k_w**2*L_flux) 
        partials['R_w', 'L_flux'] = -1 * np.log((D_od/2)/(r_i))/(2*np.pi*k_w*L_flux**2) 

        partials['R_wk', 'r_i'] = 1/(r_i*(2*np.pi*k_wk*L_flux)) 
        partials['R_wk', 'D_v'] = -1/(D_v*(2*np.pi*k_wk*L_flux)) 
        partials['R_wk', 'k_wk'] = -1 * np.log((r_i)/(D_v/2))/(2*np.pi*k_wk**2*L_flux)
        partials['R_wk', 'L_flux'] = -1 * np.log((r_i)/(D_v/2))/(2*np.pi*k_wk*L_flux**2)

        partials['R_inter', 'alpha'] = -dh_dalpha / (h_inter**2 * A_inter)
        partials['R_inter', 'h_fg'] = -dh_dhfg / (h_inter**2 * A_inter)
        partials['R_inter', 'T_hp'] = -dh_dThp / (h_inter**2 * A_inter)
        partials['R_inter', 'v_fg'] = -dh_dvfg / (h_inter**2 * A_inter)
        partials['R_inter', 'R_g'] = -dh_dRg / (h_inter**2 * A_inter)
        partials['R_inter', 'P_v'] = -dh_dPv / (h_inter**2 * A_inter)
        partials['R_inter', 'A_inter'] = -1 / (h_inter * A_inter**2)


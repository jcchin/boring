import numpy as np
# from math import pi

import openmdao.api as om

class CondThermResComp(om.ExplicitComponent):

    def setup(self):

        self.add_input('alpha', val=14.0)
        self.add_input('h_fg', val=1.0)
        self.add_input('T_hp', val=1.0)
        self.add_input('v_fg', val=1.0)
        self.add_input('R_g', val=1.0)
        self.add_input('P_v', val=1.0)
        self.add_input('D_od', val=1.0)
        self.add_input('r_i', val=1.0)
        self.add_input('k_w', val=1.0)
        self.add_input('L_cond', val=1.0)
        self.add_input('D_v', val=1.0)
        self.add_input('k_wk', val=1.0)
        self.add_input('A_interc', val=1.0)

        self.add_output('h_interc', val=1.0)
        self.add_output('R_wc', val=1.0)
        self.add_output('R_wkc', val=1.0)
        self.add_output('R_interc', val=1.0)

    def setup_partials(self):

        self.declare_partials('h_interc', ['alpha', 'h_fg', 'T_hp', 'v_fg', 'R_g', 'P_v'])
        self.declare_partials('R_wc', ['D_od', 'r_i', 'k_w', 'L_cond'])
        self.declare_partials('R_wkc', ['D_v', 'r_i', 'k_wk', 'L_cond'])
        self.declare_partials('R_interc', ['alpha', 'h_fg', 'T_hp', 'v_fg', 'R_g', 'P_v', 'A_interc'])


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
        L_cond = inputs['L_cond']
        D_v = inputs['D_v']
        k_wk = inputs['k_wk']
        A_interc = inputs['A_interc']

        #alpha=1           # Look into this, need better way to determine this rather than referencing papers.
        h_interc=outputs['h_interc'] = 2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg)) # Sydney
        outputs['R_wc'] = np.log((D_od/2)/(r_i))/(2*np.pi*k_w*L_cond) # Sydney
        outputs['R_wkc'] = np.log((r_i)/(D_v/2))/(2*np.pi*k_wk*L_cond) # Sydney
        outputs['R_interc'] = 1/(h_interc*A_interc) # Sydney

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
        L_cond = inputs['L_cond']
        D_v = inputs['D_v']
        k_wk = inputs['k_wk']
        A_interc = inputs['A_interc']

        h_interc= 2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg))

        dh_dalpha = partials['h_interc', 'alpha'] = 4/(2-alpha)**2 * (h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg)) 
        dh_dhfg = partials['h_interc', 'h_fg'] = 4*alpha*h_fg/((2-alpha)*T_hp*v_fg) * np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg)) + (2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(P_v*v_fg/(2*h_fg**2)))
        dh_dThp = partials['h_interc', 'T_hp'] = (-2*alpha*h_fg**2)/((2-alpha)*T_hp**2*v_fg)*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg)) + (2*alpha*h_fg**2/((2-alpha)*T_hp*v_fg) * 0.5*(1/(2*np.pi*R_g*T_hp))**-0.5 * (-1/(2*np.pi*R_g*T_hp**2))*(1-P_v*v_fg/(2*h_fg)))
        dh_dvfg = partials['h_interc', 'v_fg'] = (-2*alpha*h_fg**2)/((2-alpha)*T_hp*v_fg**2)*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg)) + ((2*alpha*h_fg**2/((2-alpha)*T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(-P_v/(2*h_fg)))
        dh_dRg = partials['h_interc', 'R_g'] = 2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*0.5*(1/(2*np.pi*R_g*T_hp))**(-0.5)*(-1/(2*np.pi*R_g**2*T_hp))*(1-P_v*v_fg/(2*h_fg)) 
        dh_dPv = partials['h_interc', 'P_v'] = 2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(-v_fg/(2*h_fg))

        partials['R_wc', 'D_od'] = 1/(D_od*(2*np.pi*k_w*L_cond)) 
        partials['R_wc', 'r_i'] = -1/(2*np.pi*k_w*L_cond)
        partials['R_wc', 'k_w'] = -1 * np.log((D_od/2)/(r_i))/(2*np.pi*k_w**2*L_cond) 
        partials['R_wc', 'L_cond'] = -1 * np.log((D_od/2)/(r_i))/(2*np.pi*k_w*L_cond**2) 

        partials['R_wkc', 'r_i'] = 1/(r_i*(2*np.pi*k_wk*L_cond)) 
        partials['R_wkc', 'D_v'] = -1/(D_v*(2*np.pi*k_wk*L_cond)) 
        partials['R_wkc', 'k_wk'] = -1 * np.log((r_i)/(D_v/2))/(2*np.pi*k_wk**2*L_cond)
        partials['R_wkc', 'L_cond'] = -1 * np.log((r_i)/(D_v/2))/(2*np.pi*k_wk*L_cond**2)

        partials['R_interc', 'alpha'] = -dh_dalpha / (h_interc**2 * A_interc)
        partials['R_interc', 'h_fg'] = -dh_dhfg / (h_interc**2 * A_interc)
        partials['R_interc', 'T_hp'] = -dh_dThp / (h_interc**2 * A_interc)
        partials['R_interc', 'v_fg'] = -dh_dvfg / (h_interc**2 * A_interc)
        partials['R_interc', 'R_g'] = -dh_dRg / (h_interc**2 * A_interc)
        partials['R_interc', 'P_v'] = -dh_dPv / (h_interc**2 * A_interc)
        partials['R_interc', 'A_interc'] = -1 / (h_interc * A_interc**2)

if __name__ =='__main__':

    prob = om.Problem()

    prob.model.add_subsystem('cond_R_t_comp', CondThermResComp())

    prob.setup(force_alloc_complex = True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True, show_only_incorrect=True)
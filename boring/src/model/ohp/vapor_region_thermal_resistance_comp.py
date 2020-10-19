from openmdao.api import ExplicitComponent, Problem
import numpy as np

class VaporRegionThermalResistanceComp(ExplicitComponent):

    def setup(self):
        self.add_input('D_v', val=0.00362) # add units
        self.add_input('R_g', val=1.0)
        self.add_input('mu_v', val=1.0)
        self.add_input('T_hp', val=1.0)
        self.add_input('h_fg', val=1.0)
        self.add_input('P_v', val=1.0)
        self.add_input('rho_v', val=1.0)
        self.add_input('L_eff', val=1.0)

        self.add_output('r_h', val=1.0)
        self.add_output('R_v', val=1.0)

    def setup_partials(self):
        self.declare_partials('r_h', 'D_v')

        self.declare_partials('R_v', 'R_g')
        self.declare_partials('R_v', 'mu_v')
        self.declare_partials('R_v', 'T_hp')
        self.declare_partials('R_v', 'h_fg')
        self.declare_partials('R_v', 'P_v')
        self.declare_partials('R_v', 'rho_v')
        self.declare_partials('R_v', 'L_eff')
        self.declare_partials('R_v', 'D_v')

    def compute(self, inputs, outputs):
        D_v = inputs['D_v']
        R_g = inputs['R_g']
        mu_v = inputs['mu_v']
        T_hp = inputs['T_hp']
        h_fg = inputs['h_fg']
        P_v = inputs['P_v']
        rho_v = inputs['rho_v']
        L_eff = inputs['L_eff']

        r_h = outputs['r_h'] = D_v/2
        outputs['R_v'] = 8*R_g*mu_v*T_hp**2/(np.pi*h_fg**2*P_v*rho_v)*(L_eff/(r_h**4))

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

        partials['R_v', 'R_g'] = 8*mu_v*T_hp**2/(np.pi*h_fg**2*P_v*rho_v)*(L_eff/(r_h**4))
        partials['R_v', 'mu_v'] = 8*R_g*T_hp**2/(np.pi*h_fg**2*P_v*rho_v)*(L_eff/(r_h**4))
        partials['R_v', 'T_hp'] = 2*8*R_g*mu_v*T_hp**2/(np.pi*h_fg**2*P_v*rho_v)*(L_eff/(r_h**4))
        partials['R_v', 'L_eff'] = 8*R_g*mu_v*T_hp**2/(np.pi*h_fg**2*P_v*rho_v)*(1/(r_h**4))
        partials['R_v', 'h_fg'] = -2 * 8*R_g*mu_v*T_hp**2/(np.pi*h_fg**3*P_v*rho_v)*(L_eff/(r_h**4))
        partials['R_v', 'P_v'] = -1 * 8*R_g*mu_v*T_hp**2/(np.pi*h_fg**2*P_v**2*rho_v)*(L_eff/(r_h**4))
        partials['R_v', 'rho_v'] = -1 * 8*R_g*mu_v*T_hp**2/(np.pi*h_fg**2*P_v*rho_v**2)*(L_eff/(r_h**4))
        partials['R_v', 'D_v'] = -4 * 8*R_g*mu_v*T_hp**2/(np.pi*h_fg**2*P_v*rho_v)*(L_eff/(r_h**5)) * 0.5

if __name__ =='__main__':

    prob = Problem()

    prob.model.add_subsystem('vapor_R_t_comp', VaporRegionThermalResistanceComp())

    prob.setup()
    prob.run_model()
    prob.check_partials(method='cs')



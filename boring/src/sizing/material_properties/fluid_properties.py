from __future__ import absolute_import
import numpy as np
from math import pi
import openmdao.api as om
class FluidPropertiesComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('T_hp', val=1.0 * np.ones(nn), units='degC', desc='vapor temperature')
        self.add_output('P_v', val=1.0 * np.ones(nn), units='Pa', desc='pressure')
        self.add_output('h_fg', val=1.0 * np.ones(nn), units='J/kg', desc='latent heat')
        self.add_output('rho_l', val=1.0 * np.ones(nn), units='kg/m**3', desc='density of liquid')
        self.add_output('rho_v', val=1.0 * np.ones(nn), units='kg/m**3', desc='density of vapor')
        self.add_output('mu_l', val=1.0 * np.ones(nn), units='N*s/m**2', desc='liquid viscosity')
        self.add_output('mu_v', val=1.0 * np.ones(nn), units='N*s/m**2', desc='vapor viscosity')
        self.add_output('k_l', val=1.0 * np.ones(nn), units='W/(m*K)', desc='liquid conductivity')
        self.add_output('k_v', val=1.0 * np.ones(nn), units='W/(m*K)', desc='vapor conductivity')
        self.add_output('sigma_l', val=1.0 * np.ones(nn), units='N/m**3', desc='surface tension')
        self.add_output('cp_l', val=1.0 * np.ones(nn), desc='liquid specific heat')
        self.add_output('cp_v', val=1.0 * np.ones(nn), desc='vapor specific heat')
        self.add_output('v_fg', val=1.0 * np.ones(nn), units='m**3/kg', desc='specific volume')
        self.add_output('R_g', val=1.0 * np.ones(nn), units='J/kg/K', desc='gas constant of the vapor')
    # Add outputs for all properties
    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        # self.declare_partials('*', '*', method='cs')
        # self.declare_partials(of='Tdot', wrt='T', rows=arange, cols=arange)
        self.declare_partials('P_v', ['T_hp'], rows=ar, cols=ar, method='cs')
        self.declare_partials('h_fg', ['T_hp'], rows=ar, cols=ar, method='cs')
        self.declare_partials('rho_l', ['T_hp'], rows=ar, cols=ar, method='cs')
        self.declare_partials('rho_v', ['T_hp'], rows=ar, cols=ar, method='cs')
        self.declare_partials('mu_l', ['T_hp'], rows=ar, cols=ar, method='cs')
        self.declare_partials('mu_v', ['T_hp'], rows=ar, cols=ar, method='cs')
        self.declare_partials('k_l', ['T_hp'], rows=ar, cols=ar, method='cs')
        self.declare_partials('k_v', ['T_hp'], rows=ar, cols=ar, method='cs')
        self.declare_partials('sigma_l', ['T_hp'], rows=ar, cols=ar, method='cs')
        self.declare_partials('cp_l', ['T_hp'], rows=ar, cols=ar, method='cs')
        self.declare_partials('cp_v', ['T_hp'], rows=ar, cols=ar, method='cs')
    def compute(self, inputs, outputs):
        def f(T, a_0, a_1, a_2, a_3, a_4, a_5):
            poly = np.exp(a_0 + a_1 * T + a_2 * T ** 2 + a_3 * T ** 3 + a_4 * T ** 4 + a_5 * T ** 5)
            return poly
        # temperature polynomials use Celsius, everything else uses Kelvin
        outputs['P_v'] = f(inputs['T_hp'], -5.0945, 7.2280e-2, -2.8625e-4, 9.2341e-7, -2.0295e-9, 2.1645e-12) * 1e5
        outputs['h_fg'] = f(inputs['T_hp'], 7.8201, -5.8906e-4, -9.1355e-6, 8.4738e-8, -3.9635e-10, 5.9150e-13) * 1e3
        outputs['rho_l'] = f(inputs['T_hp'], 6.9094, -2.0146e-5, -5.9868e-6, 2.5921e-8, -9.3244e-11, 1.2103e-13)
        outputs['rho_v'] = f(inputs['T_hp'], -5.3225, 6.8366e-2, -2.7243e-4, 8.4522e-7, -1.6558e-9, 1.5514e-12)
        outputs['mu_l'] = f(inputs['T_hp'], -6.3530, -3.1540e-2, 2.1670e-4, -1.1559e-6, 3.7470e-9, -5.2189e-12)
        outputs['mu_v'] = f(inputs['T_hp'], -11.596, 2.6382e-3, 6.9205e-6, -6.1035e-8, 1.6844e-10, -1.5910e-13)
        outputs['k_l'] = f(inputs['T_hp'], -5.8220e-1, 4.1177e-3, -2.7932e-5, 6.5617e-8, 4.1100e-11, -3.8220e-13)
        outputs['k_v'] = f(inputs['T_hp'], -4.0722, 3.2364e-3, 6.3860e-6, 8.5114e-9, -1.0464e-10, 1.6481e-13)
        outputs['sigma_l'] = f(inputs['T_hp'], 4.3438, -3.0664e-3, 2.0743e-5, -2.5499e-7, 1.0377e-9, -1.7156e-12) / 1e3
        outputs['cp_l'] = f(inputs['T_hp'], 1.4350, -3.2231e-4, 6.1633e-6, -4.4099e-8, 2.0968e-10, -3.040e-13) * 1e3
        outputs['cp_v'] = f(inputs['T_hp'], 6.3198e-1, 6.7903e-4, -2.5923e-6, 4.4936e-8, 2.2606e-10, -9.0694e-13) * 1e3
        outputs['v_fg'] = 1 / outputs['rho_v'] - 1 / outputs['rho_l']
        outputs['R_g'] = outputs['P_v'] / ((inputs['T_hp'] + 273.15) * outputs['rho_v'])

if __name__ == '__main__':
    from openmdao.api import Problem
    nn = 1
    geom='flat'
    prob = Problem()
    prob.model.add_subsystem(name = 'fluids',
                            subsys = FluidPropertiesComp(num_nodes=nn),
                            promotes_inputs=['T_hp'],
                            promotes_outputs=['R_g', 'P_v', 'rho_v', 'mu_v', 'h_fg','v_fg','k_l'])
    prob.setup(force_alloc_complex=True)
    prob.run_model()
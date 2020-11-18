from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om

class FluidPropertiesComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)


    def setup(self):
        nn=self.options['num_nodes']

        self.add_input('Q_hp', 50, desc='heat flux')
        self.add_input('A_cond', 0.0003769911184307752, units='m**2', desc='Conductor Area')
        self.add_input('h_c', 1200, desc='external convection at the condensor')
        self.add_input('T_coolant', 285, units='K', desc='coolant temperature')

        self.add_output('T_hp', units='degC', desc='vapor temperature')
        self.add_output('P_v', units='Pa', desc='pressure')
        self.add_output('h_fg', units='J/kg', desc='latent heat')
        self.add_output('rho_l',  units='kg/m**3', desc='density of liquid')
        self.add_output('rho_v', units='kg/m**3', desc='density of vapor')
        self.add_output('mu_l',  units='N*s/m**2', desc='liquid viscosity')
        self.add_output('mu_v', units='N*s/m**2', desc='vapor viscosity')
        self.add_output('k_l', units='W/(m*K)', desc='liquid conductivity')
        self.add_output('k_v', units='W/(m*K)', desc='vapor conductivity')
        self.add_output('sigma_l', units='N/m**3', desc='surface tension')
        self.add_output('cp_l',  desc='liquid specific heat')
        self.add_output('cp_v',  desc='vapor specific heat')
        self.add_output('v_fg', units='m**3/kg', desc='specific volume')
        self.add_output('R_g', units='J/kg/K', desc='gas constant of the vapor')
        
# Add outputs for all properties

    def setup_partials(self):
        #self.declare_partials('*', '*', method='cs')
        #self.declare_partials(of='Tdot', wrt='T', rows=arange, cols=arange)
        self.declare_partials('T_hp','Q_hp')
        self.declare_partials('T_hp','A_cond')
        self.declare_partials('T_hp','h_c')
        self.declare_partials('T_hp','T_coolant')
        self.declare_partials('P_v','Q_hp')
        self.declare_partials('P_v','A_cond')
        self.declare_partials('P_v','h_c')
        self.declare_partials('P_v','T_coolant')
        self.declare_partials('h_fg','Q_hp')
        self.declare_partials('h_fg','A_cond')
        self.declare_partials('h_fg','h_c')
        self.declare_partials('h_fg','T_coolant')
        self.declare_partials('rho_l','Q_hp')
        self.declare_partials('rho_l','A_cond')
        self.declare_partials('rho_l','h_c')
        self.declare_partials('rho_l','T_coolant')
        self.declare_partials('rho_v','Q_hp')
        self.declare_partials('rho_v','A_cond')
        self.declare_partials('rho_v','h_c')
        self.declare_partials('rho_v','T_coolant')
        self.declare_partials('mu_l','Q_hp')
        self.declare_partials('mu_l','A_cond')
        self.declare_partials('mu_l','h_c')
        self.declare_partials('mu_l','T_coolant')
        self.declare_partials('mu_v','Q_hp')
        self.declare_partials('mu_v','A_cond')
        self.declare_partials('mu_v','h_c')
        self.declare_partials('mu_v','T_coolant')     
        self.declare_partials('k_l','Q_hp')
        self.declare_partials('k_l','A_cond')
        self.declare_partials('k_l','h_c')
        self.declare_partials('k_l','T_coolant')
        self.declare_partials('k_v','Q_hp')
        self.declare_partials('k_v','A_cond')
        self.declare_partials('k_v','h_c')
        self.declare_partials('k_v','T_coolant')
        self.declare_partials('sigma_l','Q_hp')
        self.declare_partials('sigma_l','A_cond')
        self.declare_partials('sigma_l','h_c')
        self.declare_partials('sigma_l','T_coolant')
        self.declare_partials('cp_l','Q_hp')
        self.declare_partials('cp_l','A_cond')
        self.declare_partials('cp_l','h_c')
        self.declare_partials('cp_l','T_coolant')
        self.declare_partials('cp_v','Q_hp')
        self.declare_partials('cp_v','A_cond')
        self.declare_partials('cp_v','h_c')
        self.declare_partials('cp_v','T_coolant')     
        
        
    def compute(self, inputs, outputs):
        Q_hp = inputs['Q_hp']
        A_cond = inputs['A_cond']
        h_c = inputs['h_c']
        T_coolant = inputs['T_coolant']

        def f(T,a_0,a_1,a_2,a_3,a_4,a_5):
                poly=np.exp(a_0+a_1*T+a_2*T**2+a_3*T**3+a_4*T**4+a_5*T**5)
                return poly
            
        outputs['T_hp'] = Q_hp/(A_cond*h_c)+T_coolant-273.15

        # temperature polynomials use Celsius, everything else uses Kelvin
        outputs['P_v'] = f(outputs['T_hp'],-5.0945,7.2280e-2,-2.8625e-4,9.2341e-7,-2.0295e-9,2.1645e-12)*1e5
        outputs['h_fg'] = f(outputs['T_hp'],7.8201,-5.8906e-4,-9.1355e-6,8.4738e-8,-3.9635e-10,5.9150e-13)*1e3
        outputs['rho_l'] = f(outputs['T_hp'],6.9094,-2.0146e-5,-5.9868e-6,2.5921e-8,-9.3244e-11,1.2103e-13)     
        outputs['rho_v'] = f(outputs['T_hp'],-5.3225,6.8366e-2,-2.7243e-4,8.4522e-7,-1.6558e-9,1.5514e-12)
        outputs['mu_l'] = f(outputs['T_hp'],-6.3530,-3.1540e-2,2.1670e-4,-1.1559e-6,3.7470e-9,-5.2189e-12)
        outputs['mu_v'] = f(outputs['T_hp'],-11.596,2.6382e-3,6.9205e-6,-6.1035e-8,1.6844e-10,-1.5910e-13)
        outputs['k_l'] = f(outputs['T_hp'],-5.8220e-1,4.1177e-3,-2.7932e-5,6.5617e-8,4.1100e-11,-3.8220e-13)
        outputs['k_v'] = f(outputs['T_hp'],-4.0722,3.2364e-3,6.3860e-6,8.5114e-9,-1.0464e-10,1.6481e-13)
        outputs['sigma_l'] = f(outputs['T_hp'],4.3438,-3.0664e-3,2.0743e-5,-2.5499e-7,1.0377e-9,-1.7156e-12)/1e3
        outputs['cp_l'] = f(outputs['T_hp'],1.4350,-3.2231e-4,6.1633e-6,-4.4099e-8,2.0968e-10,-3.040e-13)*1e3
        outputs['cp_v'] = f(outputs['T_hp'],6.3198e-1,6.7903e-4,-2.5923e-6,4.4936e-8,2.2606e-10,-9.0694e-13)*1e3
        outputs['v_fg'] = 1/outputs['rho_v']-1/outputs['rho_l']
        outputs['R_g'] = outputs['P_v']/((outputs['T_hp']+273.15)*outputs['rho_v'])
    
    
    def compute_partials(self, inputs, partials):
        Q_hp = inputs['Q_hp']
        A_cond = inputs['A_cond']
        h_c = inputs['h_c']
        T_coolant = inputs['T_coolant'] 
        #T_hp = (Q_hp/(A_cond*h_c)+T_coolant-273.15)
        
        def pd(T,a_0,a_1,a_2,a_3,a_4,a_5):
            poly=(a_1+2*a_2*T+3*a_3*T**2+4*a_4*T**3+5*a_5*T**4)*np.exp(a_0+a_1*T+a_2*T**2+a_3*T**3+a_4*T**4+a_5*T**5)
            return poly  
        
        def pdT(Q_hp,A_cond,h_c,T_coolant,a_0,a_1,a_2,a_3,a_4,a_5):
            poly=(a_1+2*a_2*(Q_hp/(A_cond*h_c)+T_coolant-273.15)+3*a_3*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**2+4*a_4*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**3+5*a_5*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**4)*np.exp(a_0+a_1*(Q_hp/(A_cond*h_c)+T_coolant-273.15)+a_2*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**2+a_3*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**3+a_4*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**4+a_5*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**5)
            return poly
        
        def pdQ(Q_hp,A_cond,h_c,T_coolant,a_0,a_1,a_2,a_3,a_4,a_5):
            poly=(a_1/(h_c*A_cond)+2*a_2/(h_c*A_cond)*(Q_hp/(A_cond*h_c)+T_coolant-273.15)+3*a_3/(h_c*A_cond)*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**2+4*a_4/(h_c*A_cond)*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**3+5*a_5/(h_c*A_cond)*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**4)*np.exp(a_0+a_1*(Q_hp/(A_cond*h_c)+T_coolant-273.15)+a_2*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**2+a_3*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**3+a_4*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**4+a_5*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**5)
            return poly
        
        def pdh(Q_hp,A_cond,h_c,T_coolant,a_0,a_1,a_2,a_3,a_4,a_5):
            poly=(a_1*-Q_hp/(h_c**2*A_cond)+2*a_2*-Q_hp/(h_c**2*A_cond)*(Q_hp/(A_cond*h_c)+T_coolant-273.15)+3*a_3*-Q_hp/(h_c**2*A_cond)*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**2+4*a_4*-Q_hp/(h_c**2*A_cond)*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**3+5*a_5*-Q_hp/(h_c**2*A_cond)*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**4)*np.exp(a_0+a_1*(Q_hp/(A_cond*h_c)+T_coolant-273.15)+a_2*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**2+a_3*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**3+a_4*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**4+a_5*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**5)
            return poly
        
        def pdA(Q_hp,A_cond,h_c,T_coolant,a_0,a_1,a_2,a_3,a_4,a_5):
            poly=(a_1*-Q_hp/(h_c*A_cond**2)+2*a_2*-Q_hp/(h_c*A_cond**2)*(Q_hp/(A_cond*h_c)+T_coolant-273.15)+3*a_3*-Q_hp/(h_c*A_cond**2)*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**2+4*a_4*-Q_hp/(h_c*A_cond**2)*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**3+5*a_5*-Q_hp/(h_c*A_cond**2)*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**4)*np.exp(a_0+a_1*(Q_hp/(A_cond*h_c)+T_coolant-273.15)+a_2*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**2+a_3*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**3+a_4*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**4+a_5*(Q_hp/(A_cond*h_c)+T_coolant-273.15)**5)
            return poly       
        
        
        partials['T_hp','Q_hp']=1/(A_cond*h_c)
        partials['T_hp','A_cond']=-Q_hp/(A_cond**2*h_c)
        partials['T_hp','h_c']=-Q_hp/(A_cond*h_c**2)
        partials['T_hp','T_coolant']=1
        
        partials['P_v','Q_hp']=pdQ(Q_hp,A_cond,h_c,T_coolant,-5.0945,7.2280e-2,-2.8625e-4,9.2341e-7,-2.0295e-9,2.1645e-12)*1e5
        partials['P_v','A_cond']=pdA(Q_hp,A_cond,h_c,T_coolant,-5.0945,7.2280e-2,-2.8625e-4,9.2341e-7,-2.0295e-9,2.1645e-12)*1e5
        partials['P_v','h_c']=pdh(Q_hp,A_cond,h_c,T_coolant,-5.0945,7.2280e-2,-2.8625e-4,9.2341e-7,-2.0295e-9,2.1645e-12)*1e5
        partials['P_v','T_coolant']=pdT(Q_hp,A_cond,h_c,T_coolant,-5.0945,7.2280e-2,-2.8625e-4,9.2341e-7,-2.0295e-9,2.1645e-12)*1e5
        
        partials['h_fg','Q_hp'] = pdQ(Q_hp,A_cond,h_c,T_coolant,7.8201,-5.8906e-4,-9.1355e-6,8.4738e-8,-3.9635e-10,5.9150e-13)*1e3
        partials['h_fg','A_cond'] = pdA(Q_hp,A_cond,h_c,T_coolant,7.8201,-5.8906e-4,-9.1355e-6,8.4738e-8,-3.9635e-10,5.9150e-13)*1e3
        partials['h_fg','h_c'] = pdh(Q_hp,A_cond,h_c,T_coolant,7.8201,-5.8906e-4,-9.1355e-6,8.4738e-8,-3.9635e-10,5.9150e-13)*1e3
        partials['h_fg','T_coolant'] = pdT(Q_hp,A_cond,h_c,T_coolant,7.8201,-5.8906e-4,-9.1355e-6,8.4738e-8,-3.9635e-10,5.9150e-13)*1e3
        
        partials['rho_l','Q_hp'] = pdQ(Q_hp,A_cond,h_c,T_coolant,6.9094,-2.0146e-5,-5.9868e-6,2.5921e-8,-9.3244e-11,1.2103e-13)     
        partials['rho_l','A_cond'] = pdA(Q_hp,A_cond,h_c,T_coolant,6.9094,-2.0146e-5,-5.9868e-6,2.5921e-8,-9.3244e-11,1.2103e-13)
        partials['rho_l','h_c'] = pdh(Q_hp,A_cond,h_c,T_coolant,6.9094,-2.0146e-5,-5.9868e-6,2.5921e-8,-9.3244e-11,1.2103e-13)
        partials['rho_l','T_coolant'] = pdT(Q_hp,A_cond,h_c,T_coolant,6.9094,-2.0146e-5,-5.9868e-6,2.5921e-8,-9.3244e-11,1.2103e-13)
        
        partials['rho_v','Q_hp'] = pdQ(Q_hp,A_cond,h_c,T_coolant,-5.3225,6.8366e-2,-2.7243e-4,8.4522e-7,-1.6558e-9,1.5514e-12)
        partials['rho_v','A_cond'] = pdA(Q_hp,A_cond,h_c,T_coolant,-5.3225,6.8366e-2,-2.7243e-4,8.4522e-7,-1.6558e-9,1.5514e-12)
        partials['rho_v','h_c'] = pdh(Q_hp,A_cond,h_c,T_coolant,-5.3225,6.8366e-2,-2.7243e-4,8.4522e-7,-1.6558e-9,1.5514e-12)
        partials['rho_v','T_coolant'] = pdT(Q_hp,A_cond,h_c,T_coolant,-5.3225,6.8366e-2,-2.7243e-4,8.4522e-7,-1.6558e-9,1.5514e-12)
        
        partials['mu_l','Q_hp'] = pdQ(Q_hp,A_cond,h_c,T_coolant,-6.3530,-3.1540e-2,2.1670e-4,-1.1559e-6,3.7470e-9,-5.2189e-12)
        partials['mu_l','A_cond'] = pdA(Q_hp,A_cond,h_c,T_coolant,-6.3530,-3.1540e-2,2.1670e-4,-1.1559e-6,3.7470e-9,-5.2189e-12)
        partials['mu_l','h_c'] = pdh(Q_hp,A_cond,h_c,T_coolant,-6.3530,-3.1540e-2,2.1670e-4,-1.1559e-6,3.7470e-9,-5.2189e-12)
        partials['mu_l','T_coolant'] = pdT(Q_hp,A_cond,h_c,T_coolant,-6.3530,-3.1540e-2,2.1670e-4,-1.1559e-6,3.7470e-9,-5.2189e-12)
        
        partials['mu_v','Q_hp'] = pdQ(Q_hp,A_cond,h_c,T_coolant,-11.596,2.6382e-3,6.9205e-6,-6.1035e-8,1.6844e-10,-1.5910e-13)
        partials['mu_v','A_cond'] = pdA(Q_hp,A_cond,h_c,T_coolant,-11.596,2.6382e-3,6.9205e-6,-6.1035e-8,1.6844e-10,-1.5910e-13)
        partials['mu_v','h_c'] = pdh(Q_hp,A_cond,h_c,T_coolant,-11.596,2.6382e-3,6.9205e-6,-6.1035e-8,1.6844e-10,-1.5910e-13)
        partials['mu_v','T_coolant'] = pdT(Q_hp,A_cond,h_c,T_coolant,-11.596,2.6382e-3,6.9205e-6,-6.1035e-8,1.6844e-10,-1.5910e-13)
        
        partials['k_l','Q_hp'] = pdQ(Q_hp,A_cond,h_c,T_coolant,-5.8220e-1,4.1177e-3,-2.7932e-5,6.5617e-8,4.1100e-11,-3.8220e-13)
        partials['k_l','A_cond'] = pdA(Q_hp,A_cond,h_c,T_coolant,-5.8220e-1,4.1177e-3,-2.7932e-5,6.5617e-8,4.1100e-11,-3.8220e-13)
        partials['k_l','h_c'] = pdh(Q_hp,A_cond,h_c,T_coolant,-5.8220e-1,4.1177e-3,-2.7932e-5,6.5617e-8,4.1100e-11,-3.8220e-13)
        partials['k_l','T_coolant'] = pdT(Q_hp,A_cond,h_c,T_coolant,-5.8220e-1,4.1177e-3,-2.7932e-5,6.5617e-8,4.1100e-11,-3.8220e-13)
        
        partials['k_v','Q_hp'] = pdQ(Q_hp,A_cond,h_c,T_coolant,-4.0722,3.2364e-3,6.3860e-6,8.5114e-9,-1.0464e-10,1.6481e-13)
        partials['k_v','A_cond'] = pdA(Q_hp,A_cond,h_c,T_coolant,-4.0722,3.2364e-3,6.3860e-6,8.5114e-9,-1.0464e-10,1.6481e-13)
        partials['k_v','h_c'] = pdh(Q_hp,A_cond,h_c,T_coolant,-4.0722,3.2364e-3,6.3860e-6,8.5114e-9,-1.0464e-10,1.6481e-13)
        partials['k_v','T_coolant'] = pdT(Q_hp,A_cond,h_c,T_coolant,-4.0722,3.2364e-3,6.3860e-6,8.5114e-9,-1.0464e-10,1.6481e-13)
        
        partials['sigma_l','Q_hp'] = pdQ(Q_hp,A_cond,h_c,T_coolant,4.3438,-3.0664e-3,2.0743e-5,-2.5499e-7,1.0377e-9,-1.7156e-12)/1e3
        partials['sigma_l','A_cond'] = pdA(Q_hp,A_cond,h_c,T_coolant,4.3438,-3.0664e-3,2.0743e-5,-2.5499e-7,1.0377e-9,-1.7156e-12)/1e3
        partials['sigma_l','h_c'] = pdh(Q_hp,A_cond,h_c,T_coolant,4.3438,-3.0664e-3,2.0743e-5,-2.5499e-7,1.0377e-9,-1.7156e-12)/1e3
        partials['sigma_l','T_coolant'] = pdT(Q_hp,A_cond,h_c,T_coolant,4.3438,-3.0664e-3,2.0743e-5,-2.5499e-7,1.0377e-9,-1.7156e-12)/1e3
        
        partials['cp_l','Q_hp'] = pdQ(Q_hp,A_cond,h_c,T_coolant,1.4350,-3.2231e-4,6.1633e-6,-4.4099e-8,2.0968e-10,-3.040e-13)*1e3
        partials['cp_l','A_cond'] = pdA(Q_hp,A_cond,h_c,T_coolant,1.4350,-3.2231e-4,6.1633e-6,-4.4099e-8,2.0968e-10,-3.040e-13)*1e3
        partials['cp_l','h_c'] = pdh(Q_hp,A_cond,h_c,T_coolant,1.4350,-3.2231e-4,6.1633e-6,-4.4099e-8,2.0968e-10,-3.040e-13)*1e3
        partials['cp_l','T_coolant'] = pdT(Q_hp,A_cond,h_c,T_coolant,1.4350,-3.2231e-4,6.1633e-6,-4.4099e-8,2.0968e-10,-3.040e-13)*1e3
        
        partials['cp_v','Q_hp'] = pdQ(Q_hp,A_cond,h_c,T_coolant,6.3198e-1,6.7903e-4,-2.5923e-6,4.4936e-8,2.2606e-10,-9.0694e-13)*1e3
        partials['cp_v','A_cond'] = pdA(Q_hp,A_cond,h_c,T_coolant,6.3198e-1,6.7903e-4,-2.5923e-6,4.4936e-8,2.2606e-10,-9.0694e-13)*1e3
        partials['cp_v','h_c'] = pdh(Q_hp,A_cond,h_c,T_coolant,6.3198e-1,6.7903e-4,-2.5923e-6,4.4936e-8,2.2606e-10,-9.0694e-13)*1e3
        partials['cp_v','T_coolant'] = pdT(Q_hp,A_cond,h_c,T_coolant,6.3198e-1,6.7903e-4,-2.5923e-6,4.4936e-8,2.2606e-10,-9.0694e-13)*1e3
      
        

     
        
        
        
        
        
        
        
        
        
        
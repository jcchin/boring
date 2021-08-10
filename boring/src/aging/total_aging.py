'''

Calculate capacity fade, and internal resistance growth


References:

Lithium–Ion Battery Modeling for Aerospace Applications
https://arc.aiaa.org/doi/full/10.2514/1.C036209
Section "C": Aging Model

A holistic aging model for Li(NiMnCo)O2 based 18650 lithium-ion batteries
https://www.sciencedirect.com/science/article/pii/S0378775314001876?via%3Dihub
'''


import numpy as np

from openmdao.api import ExplicitComponent, Group, IndepVarComp


class BatteryTotalAging(ExplicitComponent):
    """
    Calculate capacity fade, and internal resistance growth
    """

    def setup(self):

        # Inputs
        self.add_input('t', val=100., units='days', desc='aging time')
        self.add_input('Q', val=0.85, units='A*hr', desc='charge throughput')
        self.add_input('alpha_cap', val=1020.,  desc='calendar aging coefficient on capacity')
        self.add_input('beta_cap', val =921., desc='cycle aging coefficient on capacity')
        self.add_input('alpha_res', val=1020.,  desc='calendar aging coefficient on resistance')
        self.add_input('beta_res', val =921., desc='cycle aging coefficient on resistance')

        #unconnected output used for test checking
        self.add_output('E_fade', val=0.95, desc='energy capacity fade')
        self.add_output('R_growth', val=0.01, units='ohms', desc='internal resistance growth')

        # #Finite difference all partials.
        self.declare_partials('*', '*', method='cs')

        # self.declare_partials('E_fade',['t','Q','alpha_cap','beta_cap'])


    def compute(self, inputs, outputs):

        t = inputs['t']
        Q = inputs['Q']
        
        outputs['E_fade'] = 1-inputs['alpha_cap']*t**0.75 - inputs['beta_cap']*Q**0.5
        outputs['R_growth'] = 1+inputs['alpha_res']*t**0.75 - inputs['beta_res']*Q


    # def compute_partials(self, inputs, partials):

    #     t = inputs['t']
    #     Q = inputs['Q']

        #partials['R_growth','Q'] = -inputs['beta_res']


class BatteryParams(ExplicitComponent):
    """
    Calculate alpha and beta parameters
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('V', val=100*np.ones(nn), units='V', desc='Battery Voltage')
        self.add_input('T', val=85*np.ones(nn), units='degC', desc='Battery Temperature')
        self.add_input('V_rms', val=80*np.ones(nn), units='V', desc='Quadratic Average (RMS) Voltage')
        self.add_input('DOD', val=80., desc='Maximum Cycle Depth of Discharge allowable (as a percentage)')
        self.add_input('a_cap_c1', val=7.543)
        self.add_input('a_cap_c2', val=23.75)
        self.add_input('a_cap_c3', val=-6976.)
        self.add_input('a_res_c1', val=5.270)
        self.add_input('a_res_c2', val=16.32)
        self.add_input('a_res_c3', val=-5986)
        self.add_input('b_cap_c1', val=7.34810**-3)
        self.add_input('b_cap_c2', val=3.667)
        self.add_input('b_cap_c3', val=7.60010**-4)
        self.add_input('b_cap_c4', val=4.08110**-3)
        self.add_input('b_res_c1', val=2.15310**-4)
        self.add_input('b_res_c2', val=3.725)
        self.add_input('b_res_c3', val=1.52110**-5)
        self.add_input('b_res_c4', val=2.79810**-4)

        # Outputs
        self.add_output('alpha_cap', val=1020.,  desc='calendar aging coefficient on capacity')
        self.add_output('beta_cap', val =921., desc='cycle aging coefficient on capacity')
        self.add_output('alpha_res', val=1020.,  desc='calendar aging coefficient on resistance')
        self.add_output('beta_res', val =921., desc='cycle aging coefficient on resistance')


        # Finite difference all partials.
        self.declare_partials('*', '*', method='cs')

        # self.declare_partials('alpha_cap',['V','T'])


    def compute(self, inputs, outputs):

        V = inputs['V']
        T = inputs['T']
        DOD = inputs['DOD']
        V_rms = inputs['V_rms']
        a_c_1 = inputs['a_cap_c1']
        a_c_2 = inputs['a_cap_c2']
        a_c_3 = inputs['a_cap_c3']
        a_r_1 = inputs['a_res_c1']
        a_r_2 = inputs['a_res_c2']
        a_r_3 = inputs['a_res_c3']
        b_c_1 = inputs['b_cap_c1']
        b_c_2 = inputs['b_cap_c2']
        b_c_3 = inputs['b_cap_c3']
        b_c_4 = inputs['b_cap_c4']
        b_r_1 = inputs['b_res_c1']
        b_r_2 = inputs['b_res_c2']
        b_r_3 = inputs['b_res_c3']
        b_r_4 = inputs['b_res_c4']
        
        #------------------------------------------------------------------------------
        # Equations 14a-d from "Lithium–Ion Battery Modeling for Aerospace Applications"
        #------------------------------------------------------------------------------
        outputs['alpha_cap'] = (a_c_1*V - a_c_2)*10**6*np.exp(a_c_3/T)        
        outputs['alpha_res'] = (a_r_1*V - a_r_2)*10**5*np.exp(a_r_3/T)        
        outputs['beta_cap'] = b_c_1*(V_rms - b_c_2)**2 + b_c_3 + b_c_4*DOD    
        outputs['beta_res'] = b_r_1*(V_rms-b_r_2)**2 - b_r_3 + b_r_4*DOD      


    # def compute_partials(self, inputs, partials):

    #     V = inputs['V']
    #     T = inputs['T']

        #partials['alpha_cap','V'] = a_c_1*np.exp(a_c_3/T)

"""
These helper functions are a placeholder until a derivative friendly method replaces it (akima, convolution)


Apparent Heat Capacity Method
# https://link.springer.com/article/10.1007/s10973-019-08541-w

Author: Jeff Chin
"""

import openmdao.api as om
import numpy as np
#from boring.src.sizing.material_properties.

class PCM_Cp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int) # parallel execution

    def setup(self):
        nn=self.options['num_nodes']

        #pad geometry
        self.add_input('T', 280*np.ones(nn), units='K', desc='PCM temp')
        self.add_input('T_lo', 333*np.ones(nn), units='K', desc='PCM lower temp transition point')
        self.add_input('T_hi', 338*np.ones(nn), units='K', desc='PCM upper temp transition point')
        
        # outputs
        self.add_output('cp_pcm', 1.54*np.ones(nn), units='kJ/(kg*K)', desc='specific heat of the pcm')


    def setup_partials(self):
        #self.declare_partials('*', '*', method='cs')
        nn=self.options['num_nodes']
        ar = np.arange(nn) 

        self.declare_partials('cp_pcm', ['T','T_lo','T_hi'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        outputs['cp_pcm'] = Cp_func(inputs['T'],inputs['T_lo'],inputs['T_hi'])

    def compute_partials(self,inputs,partials):
        T = inputs['T']

        partials['cp_pcm','T'] = cp_dT_deriv_func(T)
        partials['cp_pcm','T_lo'] = cp_dT_deriv_func(T)
        partials['cp_pcm','T_hi'] = cp_dT_deriv_func(T)


def Cp_func(T, T1 = 60+273, T2 = 65+273, Cp_low = 1.5, Cp_high=50):  #kJ/kgK
    if T > T1 and T < T2:
        Cp = Cp_high
    else:
        Cp = Cp_low
    return Cp


def cp_dT_deriv_func(T):
    return 0
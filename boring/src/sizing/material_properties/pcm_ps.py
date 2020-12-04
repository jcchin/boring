"""
Calculate the Percent Solid (PS) of the PCM



Author: Jeff Chin
"""

from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om

from cp_func import Cp_func


class PCM_PS(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int) # parallel execution

    def setup(self):
        nn=self.options['num_nodes']

        #pad geometry
        self.add_input('T', 280*np.ones(nn), units='K', desc='PCM temp')
        self.add_input('T_lo', 333*np.ones(nn), units='K', desc='PCM lower temp transition point')
        self.add_input('T_hi', 338*np.ones(nn), units='K', desc='PCM upper temp transition point')

        # outputs
        self.add_output('PS',val=1.0*np.ones(nn), desc='PCM percent solid (1 = fully solid, 0 = fully liquid')



    def setup_partials(self):
        #self.declare_partials('*', '*', method='cs')
        nn=self.options['num_nodes']
        ar = np.arange(nn) 
        self.declare_partials('PS', ['T','T_lo','T_hi'], rows=ar, cols=ar)


    def compute(self, inputs, outputs):
        T = inputs['T']
        T_lo = inputs['T_lo']
        T_hi = inputs['T_hi']
    
        # if T < T_lo: PS = >1
        # if T > T_hi: PS = <0

        outputs['PS'] = (1.-(T-T_lo))/(T_hi-T_lo)


    def compute_partials(self, inputs, J):

        # add partial derivatives here
        T = inputs['T']
        T_lo = inputs['T_lo']
        T_hi = inputs['T_hi']


        J['PS','T'] = 1./(T_lo-T_hi)
        J['PS','T_lo'] = (T_hi-T+1.)/(T_hi-T_lo)**2.
        J['PS','T_hi'] = (-T_lo+T-1.)/(T_hi-T_lo)**2.
        

if __name__ == '__main__':
    from openmdao.api import Problem

    nn = 1

    prob = Problem()
    prob.model.add_subsystem('comp1', PCM_PS(num_nodes=nn), promotes_outputs=['*'], promotes_inputs=['*'])
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)





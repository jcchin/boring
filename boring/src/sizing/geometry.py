from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om


class SizeComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn=self.options['num_nodes']

        self.add_input('L_evap',  0.01, units='m',   desc='')
        self.add_input('L_cond', 0.02, units='m',  desc='')
        self.add_input('L_adiabatic', 0.03, units='m',  desc='')
        self.add_input('t_w', 0.0005,  desc='')
        self.add_input('t_wk', 0.00069,  desc='')
        self.add_input('D_od', 0.006, units='m', desc='')
        self.add_input('D_v', 0.00362, units='m', desc='')
        
        self.add_output('r_i', 1, units='m', desc='')
        self.add_output('A_cond', 1, units='m**2', desc='')
        self.add_output('A_evap', 1,  units='m**2', desc='')
        self.add_output('L_eff', 1, units='m', desc='')


    def setup_partials(self):
        # self.declare_partials('*', '*', method='fd')
        self.declare_partials('r_i', ['D_od', 't_w'])
        self.declare_partials('A_cond', ['D_od', 'L_cond'])
        self.declare_partials('A_evap', ['D_od', 'L_evap'])
        self.declare_partials('L_eff', ['L_evap', 'L_cond', 'L_adiabatic'])


    def compute(self,inputs, outputs):
        L_evap = inputs['L_evap']
        L_cond = inputs['L_cond']
        L_adiabatic = inputs['L_adiabatic']
        t_w = inputs['t_w']
        t_wk = inputs['t_wk']
        D_od = inputs['D_od']
        D_v = inputs['D_v']
        

        outputs['r_i'] =  (D_od/2-t_w)
        outputs['A_cond'] =  np.pi*D_od*L_cond
        outputs['A_evap'] =  np.pi*D_od*L_evap
        outputs['L_eff'] =  (L_evap+L_cond)/2+L_adiabatic

    # def compute_partials(self, inputs, J):
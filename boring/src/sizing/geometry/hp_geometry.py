"""
Author: Dustin Hall
"""

from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om

class HeatPipeSizeGroup(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='size',
                           subsys=SizeComp(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])


        self.add_subsystem(name='core',
                           subsys=CoreGeometries(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.set_input_defaults('L_evap', 6, units='m')
        self.set_input_defaults('L_cond', 5, units='m')
        self.set_input_defaults('D_v', 0.5, units='m')
        self.set_input_defaults('D_od', 2, units='m')
        self.set_input_defaults('t_w', 0.01, units='m')
        self.set_input_defaults('L_adiabatic', 0.01, units='m')


class SizeComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn=self.options['num_nodes']

        self.add_input('L_evap', 0.01, units='m',  desc='')
        self.add_input('L_cond', 0.02, units='m',  desc='')
        self.add_input('L_adiabatic', 0.03, units='m',  desc='')
        self.add_input('t_w', 0.0005, units='m',  desc='')
        self.add_input('t_wk', 0.00069, units='m',  desc='')
        self.add_input('D_od', 0.006, units='m',  desc='')
        self.add_input('D_v', 0.00362, units='m',  desc='')
        
        self.add_output('r_i',  desc='')
        self.add_output('A_cond', 1, units='m**2', desc='')
        self.add_output('A_evap', 1,  desc='')
        self.add_output('L_eff', 1, units='m', desc='')


    def setup_partials(self):
        self.declare_partials('r_i', 'D_od')
        self.declare_partials('r_i', 't_w')
        
        self.declare_partials('A_cond', 'D_od')
        self.declare_partials('A_cond', 'L_cond')
        
        self.declare_partials('A_evap', 'D_od')      
        self.declare_partials('A_evap', 'L_evap')
        
        self.declare_partials('L_eff', 'L_evap')
        self.declare_partials('L_eff', 'L_cond')
        self.declare_partials('L_eff', 'L_adiabatic')
    
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

    def compute_partials(self, inputs, partials):
        partials['r_i','D_od'] = 1/2
        partials['r_i','t_w'] = -1
        
        partials['A_cond','D_od'] = np.pi*inputs['L_cond']
        partials['A_cond','L_cond'] = np.pi*inputs['D_od']
        
        partials['A_evap','D_od'] = np.pi*inputs['L_evap']
        partials['A_evap','L_evap'] = np.pi*inputs['D_od']
        
        partials['L_eff','L_evap'] = 1/2
        partials['L_eff','L_cond'] = 1/2
        partials['L_eff','L_adiabatic'] = 1
        


class CoreGeometries(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn=self.options['num_nodes']

        self.add_input('D_od', 2, units='m', desc='')
        self.add_input('t_w', 0.01, units='m', desc='')
        self.add_input('D_v', 0.5, units='m', desc='')
        self.add_input('L_cond', 5, units='m', desc='')
        self.add_input('L_evap', 0.01, units='m', desc='')

        self.add_output('A_w', 1, units='m**2', desc='')
        self.add_output('A_wk', 1, units='m**2', desc='')
        self.add_output('A_interc', 1, units='m**2', desc='')
        self.add_output('A_intere', 1, units='m**2', desc='')

        self.declare_partials('A_w', ['D_od', 't_w'])
        self.declare_partials('A_wk', ['D_od', 't_w', 'D_v'])
        self.declare_partials('A_interc', ['D_v', 'L_cond'])
        self.declare_partials('A_intere', ['D_v', 'L_evap'])

    def compute(self, inputs, outputs):

        D_od = inputs['D_od']
        t_w = inputs['t_w']
        D_v = inputs['D_v']
        L_cond = inputs['L_cond']
        L_evap = inputs['L_evap']


        outputs['A_w'] =  np.pi*((D_od/2)**2-(D_od/2-t_w)**2)
        outputs['A_wk'] = np.pi*((D_od/2-t_w)**2-(D_v/2)**2)
        outputs['A_interc'] = np.pi*D_v*L_cond
        outputs['A_intere'] = np.pi*D_v*L_evap

    def compute_partials(self, inputs, J):
        D_od = inputs['D_od']
        t_w = inputs['t_w']
        D_v = inputs['D_v']
        L_cond = inputs['L_cond']
        L_evap = inputs['L_evap']

        J['A_w', 'D_od'] = np.pi*( (0.5*D_od) - (0.5*D_od - t_w) )
        J['A_w', 't_w'] = np.pi*2*(D_od/2 - t_w)
        
        J['A_wk', 'D_od'] = np.pi*(D_od/2 - t_w)
        J['A_wk', 't_w'] = -np.pi*2*(D_od/2 - t_w)
        J['A_wk', 'D_v'] = -np.pi*D_v/2

        J['A_interc', 'D_v'] = np.pi*L_cond
        J['A_interc', 'L_cond'] = np.pi*D_v

        J['A_intere', 'D_v'] = np.pi*L_evap
        J['A_intere', 'L_evap'] = np.pi*D_v


# # # ------------ Derivative Checks --------------- #
# if __name__ == "__main__":
#     from openmdao.api import Problem

#     nn = 1
#     prob = Problem()

#     prob.model.add_subsystem('comp1', SizeGroup(num_nodes=nn), promotes=['*'])

#     prob.setup(force_alloc_complex=True)
#     prob.run_model()
#     prob.check_partials(method='cs', compact_print=True)


#     print('A_w = ', prob.get_val('comp1.A_w'))
#     print('A_wk = ', prob.get_val('comp1.A_wk'))
#     print('A_interc = ', prob.get_val('comp1.A_interc'))
#     print('A_intere = ', prob.get_val('comp1.A_intere'))

#     print('r_i', prob.get_val('comp1.r_i'))
#     print('A_cond', prob.get_val('comp1.A_cond'))
#     print('A_evap', prob.get_val('comp1.A_evap'))
#     print('L_eff', prob.get_val('comp1.L_eff'))
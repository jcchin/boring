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

        self.set_input_defaults('L_flux', 6 * np.ones(nn), units='m')
        self.set_input_defaults('D_v', 0.5 * np.ones(nn), units='m')
        self.set_input_defaults('D_od', 2 * np.ones(nn), units='m')
        self.set_input_defaults('t_w', 0.01 * np.ones(nn), units='m')
        self.set_input_defaults('L_adiabatic', 0.01 * np.ones(nn), units='m')


class SizeComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('L_flux', 0.02 * np.ones(nn), units='m', desc='length of the battery')
        self.add_input('L_adiabatic', 0.03 * np.ones(nn), units='m', desc='adiabatic length')
        self.add_input('t_w', 0.0005 * np.ones(nn), units='m', desc='wall thickness')
        self.add_input('t_wk', 0.00069 * np.ones(nn), units='m', desc='wick thickness')
        self.add_input('D_od', 0.006 * np.ones(nn), units='m', desc='Vapor Outer Diameter')
        self.add_input('D_v', 0.00362 * np.ones(nn), units='m', desc='Vapor Diameter')

        self.add_output('r_i', val=1.0 * np.ones(nn), units='m', desc='inner radius')  # Radial
        self.add_output('A_flux', val=1.0 * np.ones(nn), units='m**2', desc='Area of battery in contact with HP')
        # self.add_output('L_eff', 1, units='m', desc='Effective Length')                     # Bridge

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.declare_partials('r_i', 'D_od', rows=ar, cols=ar)
        self.declare_partials('r_i', 't_w', rows=ar, cols=ar)

        self.declare_partials('A_flux', 'D_od', rows=ar, cols=ar)
        self.declare_partials('A_flux', 'L_flux', rows=ar, cols=ar)

        # self.declare_partials('L_eff', 'L_flux')
        # self.declare_partials('L_eff', 'L_adiabatic')

    def compute(self, inputs, outputs):
        L_flux = inputs['L_flux']
        L_adiabatic = inputs['L_adiabatic']
        t_w = inputs['t_w']
        t_wk = inputs['t_wk']
        D_od = inputs['D_od']
        D_v = inputs['D_v']

        outputs['r_i'] = (D_od / 2 - t_w)
        outputs['A_flux'] = np.pi * D_od * L_flux  # wrong formula for area
        # outputs['L_eff'] =  (L_flux+L_flux)/2+L_adiabatic # How to handle this for >2 battery cases?

    def compute_partials(self, inputs, partials):
        partials['r_i', 'D_od'] = 1 / 2
        partials['r_i', 't_w'] = -1

        partials['A_flux', 'D_od'] = np.pi * inputs['L_flux']
        partials['A_flux', 'L_flux'] = np.pi * inputs['D_od']

        # partials['L_eff','L_flux'] = 1
        # partials['L_eff','L_adiabatic'] = 1


class CoreGeometries(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('D_od', 2 * np.ones(nn), units='m', desc='')
        self.add_input('t_w', 0.01 * np.ones(nn), units='m', desc='')
        self.add_input('D_v', 0.5 * np.ones(nn), units='m', desc='')
        self.add_input('L_flux', 5 * np.ones(nn), units='m', desc='')

        self.add_output('A_w', 1 * np.ones(nn), units='m**2', desc='')  # Bridge
        self.add_output('A_wk', 1 * np.ones(nn), units='m**2', desc='')  # Bridge
        self.add_output('A_inter', 1 * np.ones(nn), units='m**2', desc='')  # Radial

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.declare_partials('A_w', ['D_od', 't_w'], rows=ar, cols=ar)
        self.declare_partials('A_wk', ['D_od', 't_w', 'D_v'], rows=ar, cols=ar)
        self.declare_partials('A_inter', ['D_v', 'L_flux'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        D_od = inputs['D_od']
        t_w = inputs['t_w']
        D_v = inputs['D_v']
        L_flux = inputs['L_flux']

        outputs['A_w'] = np.pi * ((D_od / 2) ** 2 - (D_od / 2 - t_w) ** 2)
        outputs['A_wk'] = np.pi * ((D_od / 2 - t_w) ** 2 - (D_v / 2) ** 2)
        outputs['A_inter'] = np.pi * D_v * L_flux

    def compute_partials(self, inputs, J):
        D_od = inputs['D_od']
        t_w = inputs['t_w']
        D_v = inputs['D_v']
        L_flux = inputs['L_flux']

        J['A_w', 'D_od'] = np.pi * ((0.5 * D_od) - (0.5 * D_od - t_w))
        J['A_w', 't_w'] = np.pi * 2 * (D_od / 2 - t_w)

        J['A_wk', 'D_od'] = np.pi * (D_od / 2 - t_w)
        J['A_wk', 't_w'] = -np.pi * 2 * (D_od / 2 - t_w)
        J['A_wk', 'D_v'] = -np.pi * D_v / 2

        J['A_inter', 'D_v'] = np.pi * L_flux
        J['A_inter', 'L_flux'] = np.pi * D_v

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
#     print('A_inter = ', prob.get_val('comp1.A_inter'))
#     print('A_intere = ', prob.get_val('comp1.A_intere'))

#     print('r_i', prob.get_val('comp1.r_i'))
#     print('A_flux', prob.get_val('comp1.A_flux'))
#     print('A_evap', prob.get_val('comp1.A_evap'))
#     print('L_eff', prob.get_val('comp1.L_eff'))

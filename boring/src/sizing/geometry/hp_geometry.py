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
        self.options.declare('geom', values=['ROUND', 'round', 'FLAT', 'flat'], default='ROUND')

    def setup(self):
        nn = self.options['num_nodes']
        geom = self.options['geom']

        self.add_subsystem(name='size',
                           subsys=SizeComp(num_nodes=nn, geom=geom),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name='core',
                           subsys=CoreGeometries(num_nodes=nn, geom=geom),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.set_input_defaults('L_flux', 6 * np.ones(nn), units='m')

        if geom == 'ROUND' or geom == 'round':
            self.set_input_defaults('D_v', 0.5 * np.ones(nn), units='m')
            self.set_input_defaults('D_od', 2 * np.ones(nn), units='m')

        if geom == 'FLAT' or geom == 'flat':
            self.set_input_defaults('W', .02 * np.ones(nn), units='m')

        self.set_input_defaults('t_w', 0.0005 * np.ones(nn), units='m')
        self.set_input_defaults('t_wk', 0.00069 * np.ones(nn), units='m')
        self.set_input_defaults('L_adiabatic', 0.01 * np.ones(nn), units='m')


class SizeComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('geom', values=['ROUND', 'round', 'FLAT', 'flat'], default='ROUND')

    def setup(self):
        nn = self.options['num_nodes']
        geom = self.options['geom']

        self.add_input('L_flux', 0.02 * np.ones(nn), units='m', desc='length of the battery')
        self.add_input('L_adiabatic', 0.03 * np.ones(nn), units='m', desc='adiabatic length')
        self.add_input('t_w', 0.0005 * np.ones(nn), units='m', desc='wall thickness')
        self.add_input('t_wk', 0.00069 * np.ones(nn), units='m', desc='wick thickness')
        self.add_input('num_cells', 1, desc='number of cells')

        if geom == 'ROUND' or geom == 'round':
            self.add_input('D_od', 0.006 * np.ones(nn), units='m', desc='Vapor Outer Diameter')
            self.add_input('D_v', 0.00362 * np.ones(nn), units='m', desc='Vapor Diameter')

            self.add_output('r_i', val=1.0 * np.ones(nn), units='m', desc='inner radius')  # Radial

        if geom == 'FLAT' or geom == 'flat':
            self.add_input('W', 0.02 * np.ones(nn), units='m', desc='Width of heat pipe into the page') 
             

        self.add_output('L_eff', val=0.5, units='m', desc='Effective Length')                     # Bridge
        self.add_output('A_flux', val=.005 * np.ones(nn), units='m**2', desc='Area of battery in contact with HP')
        
        

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        geom = self.options['geom']

        if geom == 'ROUND' or geom == 'round':
            # self.declare_partials('r_i', 'D_od', rows=ar, cols=ar)
            # self.declare_partials('r_i', 't_w', rows=ar, cols=ar)
            # self.declare_partials('A_flux', 'D_od', rows=ar, cols=ar)
            self.declare_partials('*', '*', method='cs')

        if geom == 'FLAT' or geom == 'flat':
            # self.declare_partials('A_flux', 'W')
            self.declare_partials('*', '*', method='cs')

        # self.declare_partials('A_flux', 'L_flux', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        geom = self.options['geom']
        L_flux = inputs['L_flux']
        L_adiabatic = inputs['L_adiabatic']
        t_w = inputs['t_w']
        t_wk = inputs['t_wk']
        num_cells = inputs['num_cells']

        if geom == 'ROUND' or geom == 'round':
            D_od = inputs['D_od']
            D_v = inputs['D_v']
            
            outputs['L_eff'] = 0.5
            outputs['r_i'] = (D_od / 2 - t_w)
            outputs['A_flux'] = np.pi * D_od * L_flux  # wrong formula for area !!!

        if geom == 'FLAT' or geom == 'flat':
            W = inputs['W']

            outputs['L_eff'] =  (L_flux*num_cells) + (L_adiabatic*(num_cells+1)) # How to handle this for >2 battery cases?
            outputs['A_flux'] = W * L_flux

    # def compute_partials(self, inputs, partials):

        # geom = self.options['geom']

        # if geom == 'ROUND' or geom == 'round':
        #     partials['r_i', 'D_od'] = 1 / 2
        #     partials['r_i', 't_w'] = -1

        #     partials['A_flux', 'D_od'] = np.pi * inputs['L_flux']
        #     partials['A_flux', 'L_flux'] = np.pi * inputs['D_od']

        # if geom == 'FLAT' or geom =='flat':

        #     partials['A_flux', 'W'] = inputs['L_flux']
        #     partials['A_flux', 'L_flux'] = inputs['W']


        # partials['L_eff','L_flux'] = 1
        # partials['L_eff','L_adiabatic'] = 1


class CoreGeometries(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('geom', values=['ROUND', 'round', 'FLAT', 'flat'], default='ROUND')

    def setup(self):
        nn = self.options['num_nodes']
        geom = self.options['geom']

        if geom == 'ROUND' or geom == 'round':
            self.add_input('D_od', 2 * np.ones(nn), units='m', desc='Outer diameter of heatpipe')
            self.add_input('D_v', 0.5 * np.ones(nn), units='m', desc='Diameter of vapor channel')

        elif geom == 'FLAT' or geom == 'flat':
            self.add_input('t_wk', 0.056*np.ones(nn), units='m', desc='wick thickness')
            self.add_input('W', 0.056*np.ones(nn), units='m', desc='width of heat pipe into the page')
            self.add_input('H', 0.056*np.ones(nn), units='m', desc='Height of heat pipe into the page')

        self.add_input('t_w', 0.056 * np.ones(nn), units='m', desc='wall thickness')
        self.add_input('L_flux', .056 * np.ones(nn), units='m', desc='length of the battery')

        self.add_output('A_w', 1 * np.ones(nn), units='m**2', desc='')  # Bridge
        self.add_output('A_wk', 1 * np.ones(nn), units='m**2', desc='')  # Bridge
        self.add_output('A_inter', 1 * np.ones(nn), units='m**2', desc='')  # Radial

    def setup_partials(self):
        nn = self.options['num_nodes']
        geom = self.options['geom']
        ar = np.arange(nn)

        if geom == 'ROUND' or geom == 'round':
            self.declare_partials('A_w', ['D_od', 't_w'], rows=ar, cols=ar)
            self.declare_partials('A_wk', ['D_od', 't_w', 'D_v'], rows=ar, cols=ar)
            self.declare_partials('A_inter', ['D_v', 'L_flux'], rows=ar, cols=ar)

        elif geom == 'FLAT' or geom == 'flat':
            self.declare_partials('A_w', ['W', 't_w'], rows=ar, cols=ar)
            self.declare_partials('A_wk', ['W', 't_wk'], rows=ar, cols=ar)
            self.declare_partials('A_inter', ['W', 'L_flux'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        geom = self.options['geom']

        L_flux = inputs['L_flux']
        t_w = inputs['t_w']

        if geom == 'ROUND' or geom == 'round':
            D_od = inputs['D_od']
            D_v = inputs['D_v']

            outputs['A_w'] = np.pi * ((D_od / 2) ** 2 - (D_od / 2 - t_w) ** 2)
            outputs['A_wk'] = np.pi * ((D_od / 2 - t_w) ** 2 - (D_v / 2) ** 2)
            outputs['A_inter'] = np.pi * D_v * L_flux

        elif geom == 'FLAT' or geom == 'flat':
            W = inputs['W']
            t_wk = inputs['t_wk']

            outputs['A_w'] = t_w*W
            outputs['A_wk'] = t_wk*W
            outputs['A_inter'] = W*L_flux
 
    def compute_partials(self, inputs, J):

        geom = self.options['geom']

        t_w = inputs['t_w']
        L_flux = inputs['L_flux']

        if geom == 'ROUND' or geom == 'round':
            D_od = inputs['D_od']
            D_v = inputs['D_v']

            J['A_w', 'D_od'] = np.pi * ((0.5 * D_od) - (0.5 * D_od - t_w))
            J['A_w', 't_w'] = np.pi * 2 * (D_od / 2 - t_w)

            J['A_wk', 'D_od'] = np.pi * (D_od / 2 - t_w)
            J['A_wk', 't_w'] = -np.pi * 2 * (D_od / 2 - t_w)
            J['A_wk', 'D_v'] = -np.pi * D_v / 2

            J['A_inter', 'D_v'] = np.pi * L_flux
            J['A_inter', 'L_flux'] = np.pi * D_v

        elif geom == 'FLAT' or geom == 'flat':
            W = inputs['W']
            t_wk = inputs['t_wk']

            J['A_w', 't_w'] = W
            J['A_w', 'W'] = t_w

            J['A_wk', 't_wk'] = W
            J['A_wk', 'W'] = t_wk

            J['A_inter', 'W'] = L_flux
            J['A_inter', 'L_flux'] = W 


# # ------------ Derivative Checks --------------- #
if __name__ == "__main__":
    from openmdao.api import Problem

    nn = 1
    geom='FLAT'
    prob = Problem()

    prob.model.add_subsystem('comp1', SizeComp(num_nodes=nn, geom=geom), promotes=['*'])
    # prob.model.add_subsystem('comp2', CoreGeometries(num_nodes=nn, geom=geom), promotes=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    # prob.check_partials(method='cs', compact_print=True)


    print('A_flux = ', prob.get_val('comp1.A_flux'))
    print('L_eff = ', prob.get_val('comp1.L_eff'))

    # print('A_w = ', prob.get_val('comp2.A_w'))
    # print('A_wk = ', prob.get_val('comp2.A_wk'))
    # print('A_inter = ', prob.get_val('comp2.A_inter'))


    # print('r_i', prob.get_val('comp1.r_i'))
    # print('A_flux', prob.get_val('comp1.A_flux'))
    # print('A_evap', prob.get_val('comp1.A_evap'))
    # print('L_eff', prob.get_val('comp1.L_eff'))

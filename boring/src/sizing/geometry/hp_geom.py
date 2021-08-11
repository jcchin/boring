"""
Authors: Jeff Chin (v2), Dustin Hall (original)
Combined all geometry calculations in a single component to avoid repeated code
"""

from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om


class HPgeom(om.ExplicitComponent):
    """ 
        This component calculates various heat pipe dimensions necessary for computing hp thermal resistance and mass, based on design inputs.
        The inner core dimensions and thickness variables for each layer are input, outer dimensions and areas are computed.
        This geom component passes data to a separate component for computing mass. The values computed here do not vary during transients.
        Two geometry options are available, round or flat heat pipes.
        To avoid confusion, area calculations are split based on the direction of the 2D plane.
        XS: Cross Section geometry orthogonal to the longest dimension of the heatpipe (concentric circle/rectangle)
        LW: Lengthwise geometry and contact area calculations along the longest dimension of the heatpipe
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('geom', values=['round', 'flat'], default='round')

    def setup(self):
        nn = self.options['num_nodes']
        geom = self.options['geom']

        # Geometry specific I/O
        if geom == 'round':
            self.add_input('XS:D_v', 0.56 * np.ones(nn), units='mm', desc='Diameter of the inner HP vapor channel')

            self.add_output('XS:D_od', np.ones(nn), units='mm', desc='Outer diameter of heatpipe')
            self.add_output('XS:r_i', np.ones(nn), units='mm', desc='HP wall inner radius, needed for radial resistance calculations')


        elif geom == 'flat':
            self.add_input('XS:W_v', 0.56*np.ones(nn), units='mm', desc='width of the inner heat pipe vapor core')
            self.add_input('XS:H_v', 0.56*np.ones(nn), units='mm', desc='height of the inner heat pipe vapor core')


            self.add_output('XS:W_hp', np.ones(nn), units='mm', desc='outer width of the heat pipe')
            self.add_output('XS:H_hp', np.ones(nn), units='mm', desc='outer height of the heat pipe')

        # Common Inputs
        self.add_input('LW:L_flux', 50.8 * np.ones(nn), units='mm', desc='length of thermal contact (battery width)')
        self.add_input('LW:L_adiabatic', 3 * np.ones(nn), units='mm', desc='adiabatic length (spacing between cells)')
        self.add_input('XS:t_wk', 0.69*np.ones(nn), units='mm', desc='wick thickness')
        self.add_input('XS:t_w', 0.5*np.ones(nn), units='mm', desc='wall thickness')
        # Common Outputs
        self.add_output('XS:r_h', np.ones(nn), units='m', desc='hydraulic radius (half hydraulic diameter), for resistance associated with vapor pressure drop')
        self.add_output('XS:A_w', np.ones(nn), units='mm**2', desc='cross sectional area of the heat pipe wall, needed for axial bridge and mass calcs')
        self.add_output('XS:A_wk', np.ones(nn), units='mm**2', desc='cross sectional area of the heat pipe wick, needed for axial bridge and mass calcs')
        self.add_output('LW:A_flux', np.ones(nn), units='mm**2', desc='Area of battery/pcm pad in contact with HP, needed for radial resistance calcs')
        self.add_output('LW:A_inter', np.ones(nn), units='mm**2', desc='radially projected (interfacial) area of the battery/pcm pad of the inner wick, needed for radial resitance calcs')
        self.add_output('LW:L_eff', np.ones(nn), units='mm', desc='Effective Length, from battery center to center, necessary for axial bridge calculations')

        self.declare_partials('*', '*', method='cs')

    # def setup_partials(self):
    #     nn = self.options['num_nodes']
    #     geom = self.options['geom']
    #     ar = np.arange(nn)

    #     if geom == 'round':
    #         self.declare_partials('XS:D_od', ['XS:D_v', 'XS:t_w', 'XS:t_wk'], rows=ar, cols=ar)
    #         self.declare_partials('XS:r_i', ['XS:D_v','XS:t_wk'], rows=ar, cols=ar)
    #         self.declare_partials('XS:A_wk', ['XS:D_v', 'XS:t_wk'], rows=ar, cols=ar)
    #         self.declare_partials('XS:A_w', ['XS:D_v', 'XS:t_w', 'XS:t_wk'], rows=ar, cols=ar)
    #         self.declare_partials('LW:A_flux', ['XS:D_v','XS:t_w', 'XS:t_wk','LW:L_flux'], rows=ar, cols=ar)
    #         self.declare_partials('LW:A_inter', ['XS:D_v', 'XS:t_wk','LW:L_flux'], rows=ar, cols=ar)

    #     elif geom == 'flat':
    #         self.declare_partials('XS:H_hp', ['XS:H_v','XS:t_w','XS:t_wk'], rows=ar, cols=ar)
    #         self.declare_partials('XS:W_hp', ['XS:W_v','XS:t_w','XS:t_wk'], rows=ar, cols=ar)
    #         # self.declare_partials('H_wk', ['XS:H_v', 'XS:t_w', 'XS:t_wk'], rows=ar, cols=ar)  # instead of r_i?
    #         # self.declare_partials('W_wk', ['XS:H_v', 'XS:t_w', 'XS:t_wk'], rows=ar, cols=ar)  # instead of r_i?
    #         self.declare_partials('XS:W_hp', ['XS:W_v','XS:t_w','XS:t_wk'], rows=ar, cols=ar)
    #         self.declare_partials('XS:A_wk', ['XS:H_v','XS:W_v','XS:t_wk'], rows=ar, cols=ar)
    #         self.declare_partials('XS:A_w', ['XS:H_v','XS:W_v','XS:t_w', 'XS:t_wk'], rows=ar, cols=ar)
    #         self.declare_partials('LW:A_flux', ['XS:H_v','XS:W_v','XS:t_w','XS:t_wk','LW:L_flux'], rows=ar, cols=ar)
    #         self.declare_partials('LW:A_inter', ['XS:H_v','XS:W_v','XS:t_wk','LW:L_flux'], rows=ar, cols=ar)

    #     self.declare_partials('LW:L_eff', ['LW:L_flux','LW:L_adiabatic'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        geom = self.options['geom']
        L_flux = inputs['LW:L_flux']
        L_adiabatic = inputs['LW:L_adiabatic']
        # num_cells = inputs['num_cells']
        t_wk = inputs['XS:t_wk']
        t_w = inputs['XS:t_w']

        if geom == 'round':
            D_v = inputs['XS:D_v']
            
            outputs['LW:L_eff'] = L_flux + L_adiabatic  # (L_flux*num_cells) + (L_adiabatic*(num_cells+1))
            outputs['XS:r_i'] = (D_v / 2 + t_wk)
            outputs['XS:D_od'] = D_v + t_wk*2 + t_w*2
            outputs['LW:A_flux'] = np.pi * L_flux * outputs['XS:D_od']  # circumference * length
            outputs['LW:A_inter'] = np.pi * L_flux * D_v  # circumference * length
            outputs['XS:A_wk'] = np.pi*t_wk**2 + np.pi*t_wk*D_v  # simplified from pi*((Dv/2+t_wk))^2 - pi*((Dv/2))^2
            outputs['XS:A_w'] = np.pi*t_w**2 + np.pi*t_w*D_v + 2*np.pi*t_wk*t_w # simplified from pi*((Dv/2+t_wk+t_w))^2 - pi*((Dv/2+t_k))^2
            outputs['XS:r_h'] = D_v / 2

        elif geom == 'flat':
            W_v = inputs['XS:W_v']
            H_v = inputs['XS:H_v']

            outputs['LW:L_eff'] = L_flux + L_adiabatic # (L_flux*num_cells) + (L_adiabatic*(num_cells-1)) # How to handle this for >2 battery cases?
            outputs['XS:W_hp'] = W_v + 2*t_wk + 2*t_w
            outputs['XS:H_hp'] = H_v + 2*t_wk + 2*t_w
            outputs['LW:A_flux'] = outputs['XS:W_hp'] * L_flux
            outputs['LW:A_inter'] = W_v * L_flux
            outputs['XS:A_wk'] = 4*t_wk**2 + 2*H_v*t_wk + 2*W_v*t_wk  # simplified from ((H_v+2*t_wk)*(W_v+2*t_wk)) - (H_v*W_v)
            outputs['XS:A_w'] = 4*t_w**2 + 2*H_v*t_w+ 2*W_v*t_w + 8*t_wk*t_w  # simplified from ((H_v+2*t_wk+2*t_w)(W_v+2*t_wk+2*t_w))-((H_v+2*t_wk)*(W_v+2*t_wk))
            outputs['XS:r_h'] = (H_v*W_v)/(2*H_v+2*W_v)

    # def compute_partials(self, inputs, J):

    #     self.declare_partials('*', '*', method='cs')
        # geom = self.options['geom']

        # t_w = inputs['XS:t_w']
        # t_wk = inputs['XS:t_wk']
        # L_flux = inputs['LW:L_flux']

        # if geom == 'round':
        #     D_od = inputs['XS:D_od']
        #     D_v = inputs['XS:D_v']

        #     J['XS:r_i', 'XS:D_od'] = 1 / 2
        #     J['XS:r_i', 'XS:t_w'] = -1
        #     J['A_flux', 'XS:D_od'] = np.pi * inputs['LW:L_flux']
        #     J['A_flux', 'LW:L_flux'] = np.pi * inputs['XS:D_od']
        #     J['A_inter', 'XS:D_v'] = np.pi * L_flux
        #     # J['A_inter', 'LW:L_flux'] = np.pi * D_v
        #     J['A_w', 'XS:D_od'] = np.pi * ((0.5 * D_od) - (0.5 * D_od - t_w))
        #     J['A_w', 'XS:t_w'] = np.pi * 2 * (D_od / 2 - t_w)

        #     J['A_wk', 'XS:D_od'] = np.pi * (D_od / 2 - t_w)
        #     J['A_wk', 'XS:t_w'] = -np.pi * 2 * (D_od / 2 - t_w)
        #     J['A_wk', 'XS:D_v'] = -np.pi * D_v / 2


        # elif geom == 'flat':
        #     W_v = inputs['XS:W_v']
        #     t_wk = inputs['XS:t_wk']

        #     J['A_flux', 'W'] = inputs['LW:L_flux']
        #     J['A_flux', 'LW:L_flux'] = inputs['W']

        #     J['A_inter', 'W'] = L_flux
        #     J['A_inter', 'LW:L_flux'] = W_v


        #     J['LW:L_eff','LW:L_flux'] = 1
        #     J['LW:L_eff','LW:L_adiabatic'] = 1

        #     J['A_w', 'XS:t_w'] = W_v
        #     J['A_w', 'W'] = t_w

        #     J['A_wk', 'XS:t_wk'] = W_v
        #     J['A_wk', 'W'] = t_wk


# # ------------ Derivative Checks --------------- #
if __name__ == "__main__":
    from openmdao.api import Problem

    nn = 1
    geom='flat'
    prob = Problem()

    prob.model.add_subsystem('comp1', HPgeom(num_nodes=nn, geom=geom), promotes=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    # prob.check_partials(method='cs', compact_print=True)

    prob.model.list_inputs(values=True, prom_name=True)
    prob.model.list_outputs(values=True, prom_name=True)

    print('A_flux = ', prob.get_val('LW:A_flux'))
    print('L_eff = ', prob.get_val('LW:L_eff'))

    # print('A_w = ', prob.get_val('comp2.A_w'))
    # print('A_wk = ', prob.get_val('comp2.A_wk'))
    # print('A_inter = ', prob.get_val('comp2.A_inter'))


    # print('XS:r_i', prob.get_val('comp1.r_i'))
    # print('A_flux', prob.get_val('comp1.A_flux'))
    # print('A_evap', prob.get_val('comp1.A_evap'))
    # print('LW:L_eff', prob.get_val('comp1.L_eff'))
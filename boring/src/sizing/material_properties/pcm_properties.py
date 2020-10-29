"""
Calculates bulk properties following the method outlined in 
"Experimental investigation on copper foam/hydrated salt composite phase change material for thermal energy storage" T.X. Li, D.L. Wu, F. He, R.Z. Wang

Author: Jeff Chin
"""

from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om


class PCM_props(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int) # parallel execution

    def setup(self):
        nn=self.options['num_nodes']

        #pad geometry
        self.add_input('t_pad', 0.001, units='m', desc='PCM pad thickness')
        self.add_input('porosity', 0.9,  desc='percentage porosity, 1 = completely void, 0 = solid')
        self.add_input('pad_area', 0.0571*.102, units='m', desc='cell frontal area')
        # conductive foam properties
        self.add_input('k_foam', 1., desc='thermal conductivity of the foam')
        self.add_input('rho_foam', 1., desc='intrinsic density of the foam material (unrelated to porosity)')
        self.add_input('lh_foam', 0., desc='latent heat of the foam skeleton')
        self.add_input('cp_foam', 1., desc='specific heat of the foam')
        # phase change material properties
        self.add_input('k_pcm', 1., desc='thermal conductivity of the pcm')
        self.add_input('rho_pcm', 1., desc='intrinsic density of the pcm (unrelated to porosity)')
        self.add_input('lh_pcm', 1., desc='latent heat of the pcm')
        self.add_input('cp_pcm', 1., desc='specific heat of the pcm')
        # outputs
        self.add_output('k_bulk', units='W/m*K', desc='PCM pad thermal conductivity')
        self.add_output('R_PCM', units='K/W', desc='PCM pad thermal resistance')
        self.add_input('lh_PCM', desc='latent heat of the PCM pad')



    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')


    def compute(self, inputs, outputs):
        porosity = inputs['porosity']
        k_foam = inputs['k_foam']
        k_pcm = inputs['k_pcm']
        
        outputs['k_bulk'] = 1./(porosity/k_pcm + (1-porosity)/k_foam)

        outputs['R_PCM'] = inputs['t_pad']/(inputs['pad_area']*outputs['k_bulk'])


    # def compute_partials(self, inputs, J):

        # add partial derivatives here
"""
Calculates bulk properties following the method outlined in 
"Experimental investigation on copper foam/hydrated salt composite phase change material for thermal energy storage" T.X. Li, D.L. Wu, F. He, R.Z. Wang
"Modeling and Analysis of Phase Change Materials for Efficient Thermal Management" Fulya Kaplan et al

Apparent Heat Capacity Method  Cp = f(T)  (Need upper and lower Cp, and 2 bounding temperatures)
Latent Heat Energy Model       T = f(U)  (Need melting temp, and latent heat)

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
        self.add_input('k_foam', 401., units='W/m*K', desc='thermal conductivity of the foam')
        self.add_input('rho_foam', 8960., units='kg/m**3', desc='intrinsic density of the foam material (unrelated to porosity)')
        self.add_input('lh_foam', 0., desc='latent heat of the foam skeleton')
        self.add_input('cp_foam', 0.39, units='kJ/kg*K', desc='specific heat of the foam')
        # phase change material properties
        self.add_input('k_pcm', 2., units='W/m*K', desc='thermal conductivity of the pcm')
        self.add_input('rho_pcm', 1450., units='kg/m**3', desc='intrinsic density of the pcm (unrelated to porosity)')
        self.add_input('lh_pcm', 271., units='kJ/kg', desc='latent heat of the pcm')
        self.add_input('cp_pcm', 1.54, units='kJ/kg*K', desc='specific heat of the pcm')
        # outputs
        self.add_output('k_bulk', units='W/m*K', desc='PCM pad thermal conductivity')
        self.add_output('R_PCM', units='K/W', desc='PCM pad thermal resistance')
        self.add_input('lh_PCM', desc='latent heat of the PCM pad')



    def setup_partials(self):
        self.declare_partials('*', '*')#, method='cs')


    def compute(self, inputs, outputs):
        porosity = inputs['porosity']
        k_foam = inputs['k_foam']
        k_pcm = inputs['k_pcm']
        
        # two materials arranged in series (conservative)
        outputs['k_bulk'] = 1./(porosity/k_pcm + (1-porosity)/k_foam)

        # two materials arranged in parallel (optimistic)
        # outputs['k_bulk'] = porosity*k_pcm + (1-porosity)*k_foam

        # Fancier Combined Option
        # Thermophysical properties of high porosity metal foams, A = 0.35 (empirically derived for both Aluminum and RVC foam with water/air)
        # outputs['k_bulk'] = A*(porosity*k_pcm + (1-porosity)*k_foam) + (1.-A)/(porosity/k_pcm + (1-porosity)/k_foam)

        outputs['R_PCM'] = inputs['t_pad']/(inputs['pad_area']*outputs['k_bulk'])


    def compute_partials(self, inputs, J):

        # add partial derivatives here
        porosity = inputs['porosity']
        k_foam = inputs['k_foam']
        k_pcm = inputs['k_pcm']
        A = inputs['pad_area']
        t_pad = inputs['t_pad']
        k_bulk = 1./(porosity/k_pcm + (1-porosity)/k_foam)

        J['k_bulk','porosity'] = -k_foam*k_pcm*(k_foam-k_pcm)/(k_pcm*(porosity-1.)-k_foam*porosity)**2
        J['k_bulk','k_pcm'] = k_foam**2*porosity/(k_pcm*(porosity-1.)-k_foam*porosity)**2
        J['k_bulk','k_foam'] = -k_pcm**2*(porosity-1.)/(k_pcm*(porosity-1.)-k_foam*porosity)**2


        J['R_PCM','porosity'] = t_pad*(k_foam-k_pcm)/(A*k_foam*k_pcm)
        J['R_PCM','k_pcm'] = -porosity*t_pad/(A*k_pcm**2)
        J['R_PCM','k_foam'] = (porosity-1)*t_pad/(A*k_foam**2)
        J['R_PCM','t_pad'] = 1./(inputs['pad_area']*k_bulk)
        J['R_PCM','pad_area'] = -t_pad/(k_bulk*inputs['pad_area']**2)

if __name__ == '__main__':
    from openmdao.api import Problem

    nn = 1

    prob = Problem()
    prob.model.add_subsystem('comp1', PCM_props(num_nodes=nn), promotes_outputs=['*'], promotes_inputs=['*'])
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)





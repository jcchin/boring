""""
An example battery pack used for reference, based on COMSOL data. The purpose
of this model is to size a battery pack, and check power input and heat outputs
against the COMSOL model.

Author: Jeff Chin
"""
import openmdao.api as om
import numpy as np

from math import pi

class MetaPackSizeComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)


    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('cell_rad', 9, units='mm', desc='radius of an 18650 cell')
        self.add_input('extra',  1, units='mm', desc='extra spacing along the diagonal')
        self.add_input('length', 65, units='mm', desc='length of the case')
        self.add_input('al_density', 2.7e-6, units='kg/mm**3', desc='density of aluminum')
        self.add_input('n', 4,  desc='cell array deminsion')

        # self.add_output('side', 100, units='mm', desc='side length of the case')
        self.add_output('solid_area', 1000, units='mm**2', desc='solid surface area assuming no cutouts')
        self.add_output('cell_cutout_area', 800, units='mm**2', desc='area removed for cells')
        self.add_output('area', 1000, units='mm**2', desc='final area of the surface')
        self.add_output('volume', 85000, units='mm**3', desc='volume of the case with voids removed')
        self.add_output('mass', 5, units='kg', desc='mass of the case')

        self.declare_partials('*', '*', method='cs')


    def compute(self, inputs, outputs):
        cell_rad = inputs['cell_rad']
        length = inputs['length']
        n = inputs['n']
        extra = inputs['extra']
        ratio = inputs['ratio']
        al_density = inputs['al_density']
        
        width = cell_rad*2*extra
        height = width*(n-1)
        square = width*height*n
        arc = 0.25*pi*width**2  # two half arcs (top and bottom) = full circle
        solid_area = square*(3**0.5)/2 + arc*n
        cell_cutout_area = (pi*cell_rad**2)*n**2
        outputs['area'] = solid_area - cell_cutout_area
        outputs['volume'] = outputs['area'] * length
        outputs['mass'] = outputs['area'] * length * al_density


    def _compute_partials(self, inputs, J):
        cell_rad = inputs['cell_rad']
        length = inputs['length']
        n = inputs['n']
        extra = inputs['extra']
        ratio = inputs['ratio']
        al_density = inputs['al_density']


if __name__ == '__main__':
    from openmdao.api import Problem

    nn = 1
    p=Problem()
    p.model.add_subsystem('sys1', MetaPackSizeComp(num_nodes=nn), promotes=['*'])

    


    p.setup(force_alloc_complex=True)
    p.run_model()
    p.check_partials(method='cs', compact_print=True)

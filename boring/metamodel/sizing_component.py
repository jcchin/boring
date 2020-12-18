""""
An example battery pack used for reference, based on COMSOL data. The purpose
of this model is to size a battery pack, and check power input and heat outputs
against the COMSOL model.

Author: Dustin Hall
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
        self.add_input('ratio', 1.1, desc='cell radius to cutout radius')
        self.add_input('length', 65, units='mm', desc='length of the case')
        self.add_input('al_density', 2700, units='kg/m**3', desc='density of aluminum')
        self.add_input('n', 4 ,  desc='cell array deminsion')

        self.add_output('area', 1000, units='mm**2', desc='final area of the surface')
        self.add_output('solid_area', 1000, units='mm**2', desc='solid surface area assuming no cutouts')
        self.add_output('cell_cutout_area', 800, units='mm**2', desc='area removed for cells')
        self.add_output('air_cutout_area', 800, units='mm**2', desc='area removed for air')
        self.add_output('volume', 85000, units='mm**3', desc='volume of the case with voids removed')
        self.add_output('mass', 5, units='kg', desc='mass of the case')

        r = c = np.arange(nn) 
        self.declare_partials('area', ['cell_rad', 'extra', 'n', 'ratio'])
        self.declare_partials('solid_area', ['cell_rad', 'extra', 'n'])
        self.declare_partials('cell_cutout_area', ['cell_rad', 'n'])
        self.declare_partials('air_cutout_area', ['cell_rad', 'ratio', 'n'])
        self.declare_partials('volume', ['cell_rad', 'extra', 'n', 'ratio', 'length'])
        self.declare_partials('mass', ['cell_rad', 'extra', 'n', 'ratio', 'length', 'al_density'])
      

    def compute(self, inputs, outputs):
        cell_rad = inputs['cell_rad']
        length = inputs['length']
        n = inputs['n']
        extra = inputs['extra']
        ratio = inputs['ratio']
        al_density = inputs['al_density']
        
        '''
        area = (square with sides equal to the diameter + the extra) - (cell cutouts) - (air cutouts)
        '''
        diagonal = (n*cell_d)+((n)*(cell_d/ratio))*extra    ## From Jeff
        side = diagonal/(2^0.5)                             ## From Jeff
        solid_area = side^2                                 ## From Jeff

        outputs['solid_area'] = ( (2*cell_rad + 2*extra)**2) **2 
        outputs['cell_cutout_area'] = (pi*cell_rad**2)*n**2
        outputs['air_cutout_area'] = (pi*(cell_rad*ratio)**2)*n**2
        outputs['area'] = outputs['solid_area'] - outputs['cell_cutout_area'] - outputs['air_cutout_area']

        outputs['volume'] = outputs['area'] * length

        outputs['mass'] = outputs['area'] * length * al_density


    def compute_partials(self, inputs, J):
        cell_rad = inputs['cell_rad']
        length = inputs['length']
        n = inputs['n']
        extra = inputs['extra']
        ratio = inputs['ratio']
        al_density = inputs['al_density']

        J['area', 'cell_rad'] = 4*n**2*(2*cell_rad + 2*extra) - 2*pi*cell_rad*n**2 - 2*pi*cell_rad*ratio**2 * n**2
        J['area', 'extra']    = 4*n**2*(2*cell_rad + 2*extra) 
        J['area', 'n']        = ( (2*cell_rad + 2*extra)**2) * n*2 - (pi*cell_rad**2)*n*2 - (pi*(cell_rad*ratio)**2)*n*2
        J['area', 'ratio']    = -2*pi*n**2*cell_rad**2 * ratio

        J['solid_area', 'cell_rad'] = 4*n**2*(2*cell_rad + 2*extra)
        J['solid_area', 'extra'] = 4*n**2*(2*cell_rad + 2*extra) 
        J['solid_area', 'n'] = ( (2*cell_rad + 2*extra)**2) * n*2

        J['cell_cutout_area', 'cell_rad'] = 2*pi*cell_rad*n**2
        J['cell_cutout_area', 'n'] = (pi*cell_rad**2)*n*2

        J['air_cutout_area', 'cell_rad'] = 2*pi*cell_rad*ratio**2 * n**2
        J['air_cutout_area', 'ratio'] = 2*pi*n**2*cell_rad**2 * ratio
        J['air_cutout_area', 'n'] = (pi*(cell_rad*ratio)**2)*n*2

        J['volume', 'cell_rad'] = (4*n**2*(2*cell_rad + 2*extra) - 2*pi*cell_rad*n**2 - 2*pi*cell_rad*ratio**2 * n**2) * length
        J['volume', 'extra']    = 4*n**2*(2*cell_rad + 2*extra)  * length
        J['volume', 'n']        = (( (2*cell_rad + 2*extra)**2) * n*2 - (pi*cell_rad**2)*n*2 - (pi*(cell_rad*ratio)**2)*n*2 )* length
        J['volume', 'ratio']    = -2*pi*n**2*cell_rad**2 * ratio * length
        J['volume', 'length']   = (( (2*cell_rad + 2*extra)**2) * n**2 - (pi*cell_rad**2)*n**2 - (pi*(cell_rad*ratio)**2)*n**2 )

        J['mass', 'cell_rad'] = ( 4*n**2*(2*cell_rad + 2*extra) - 2*pi*cell_rad*n**2 - 2*pi*cell_rad*ratio**2 * n**2 ) * length * al_density
        J['mass', 'extra']    = (4*n**2*(2*cell_rad + 2*extra)  ) * length * al_density
        J['mass', 'n']        = ( ( (2*cell_rad + 2*extra)**2) * n*2 - (pi*cell_rad**2)*n*2 - (pi*(cell_rad*ratio)**2)*n*2 ) * length * al_density
        J['mass', 'ratio']    = -2*pi*n**2*cell_rad**2 * ratio * length * al_density
        J['mass', 'length']   = (( (2*cell_rad + 2*extra)**2) * n**2 - (pi*cell_rad**2)*n**2 - (pi*(cell_rad*ratio)**2)*n**2 )* al_density
        J['mass', 'al_density']  = (( (2*cell_rad + 2*extra)**2) * n**2 - (pi*cell_rad**2)*n**2 - (pi*(cell_rad*ratio)**2)*n**2) * length



if __name__ == '__main__':
    from openmdao.api import Problem

    nn = 1
    prob=Problem()
    prob.model.add_subsystem('sys1', MetaPackSizeComp(num_nodes=nn), promotes=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)
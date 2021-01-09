""""
An example battery pack used for reference, based on COMSOL data. The purpose
of this model is to size a battery pack, and check power input and heat outputs
against the COMSOL model.

Author: Dustin Hall, Jeff Chin
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
        self.add_input('ratio', 1., desc='cell radius to cutout radius')
        self.add_input('length', 65, units='mm', desc='length of the case')
        self.add_input('al_density', 2.7e-6, units='kg/mm**3', desc='density of aluminum')
        self.add_input('n', 4,  desc='cell array deminsion')

        self.add_output('diagonal', 100, units='mm', desc='diagonal length of the square case')
        self.add_output('side', 100, units='mm', desc='side length of the case')
        self.add_output('solid_area', 1000, units='mm**2', desc='solid surface area assuming no cutouts')
        self.add_output('cell_cutout_area', 800, units='mm**2', desc='area removed for cells')
        self.add_output('air_cutout_area', 800, units='mm**2', desc='area removed for air')
        self.add_output('area', 1000, units='mm**2', desc='final area of the surface')
        self.add_output('volume', 85000, units='mm**3', desc='volume of the case with voids removed')
        self.add_output('mass', 5, units='kg', desc='mass of the case')

        r = c = np.arange(nn) 
        self.declare_partials('diagonal', ['n', 'cell_rad', 'ratio', 'extra'])
        self.declare_partials('side', ['n', 'cell_rad', 'ratio', 'extra'])
        self.declare_partials('solid_area', ['n', 'cell_rad', 'ratio', 'extra'])
        self.declare_partials('cell_cutout_area', ['cell_rad', 'n'])
        self.declare_partials('air_cutout_area', ['cell_rad', 'ratio', 'n'])
        self.declare_partials('area', ['cell_rad', 'extra', 'n', 'ratio'])
        self.declare_partials('volume', ['cell_rad', 'extra', 'n', 'ratio', 'length'])
        self.declare_partials('mass', ['cell_rad', 'extra', 'n', 'ratio', 'length', 'al_density'])
      

    def compute(self, inputs, outputs):
        cell_rad = inputs['cell_rad']
        length = inputs['length']
        n = inputs['n']
        extra = inputs['extra']
        ratio = inputs['ratio']
        al_density = inputs['al_density']
        
        outputs['diagonal'] = (n*cell_rad*2)+((n)*(cell_rad*2/ratio))*extra
        outputs['side'] = outputs['diagonal']/(2**0.5)
        outputs['solid_area'] = outputs['side']**2

        outputs['cell_cutout_area'] = (pi*cell_rad**2)*n**2
        outputs['air_cutout_area'] = (pi*(cell_rad/ratio)**2)*n**2

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

        diagonal = (n*cell_rad*2)+((n)*(cell_rad*2/ratio))*extra
        side = diagonal/(2**0.5)
        solid_area = side**2
        cell_cutout_area = (pi*cell_rad**2)*n**2
        air_cutout_area = (pi*(cell_rad/ratio)**2)*n**2

        area = solid_area - cell_cutout_area - air_cutout_area

        d_diagonal__d_n         = 2*cell_rad*(1+extra/ratio)
        d_diagonal__d_cell_rad  = 2*n*(1+extra/ratio)
        d_diagonal__d_ratio     = -2*cell_rad*n*(extra/ratio**2)
        d_diagonal__d_extra     = 2*cell_rad*n*(1/ratio)

        d_side__d_n       =  d_diagonal__d_n       /2**0.5
        d_side__d_cell_rad=  d_diagonal__d_cell_rad/2**0.5
        d_side__d_ratio   =  d_diagonal__d_ratio   /2**0.5
        d_side__d_extra   =  d_diagonal__d_extra   /2**0.5

        d_solid_area__d_n        = 2*(side)*(d_side__d_n)
        d_solid_area__d_cell_rad = 2*(side)*(d_side__d_cell_rad)
        d_solid_area__d_ratio    = 2*(side)*(d_side__d_ratio)
        d_solid_area__d_extra    = 2*(side)*(d_side__d_extra)
        

        J['diagonal', 'n']          =  d_diagonal__d_n       
        J['diagonal', 'cell_rad']   =  d_diagonal__d_cell_rad
        J['diagonal', 'ratio']      =  d_diagonal__d_ratio   
        J['diagonal', 'extra']      =  d_diagonal__d_extra   

        J['side', 'n']          = d_side__d_n       
        J['side', 'cell_rad']   = d_side__d_cell_rad
        J['side', 'ratio']      = d_side__d_ratio   
        J['side', 'extra']      = d_side__d_extra   


        J['solid_area', 'n']        = d_solid_area__d_n       
        J['solid_area', 'cell_rad'] = d_solid_area__d_cell_rad
        J['solid_area', 'ratio']    = d_solid_area__d_ratio   
        J['solid_area', 'extra']    = d_solid_area__d_extra   

        J['cell_cutout_area', 'cell_rad'] = 2*pi*cell_rad*n**2
        J['cell_cutout_area', 'n'] = (pi*cell_rad**2)*n*2

        J['air_cutout_area', 'cell_rad'] = 2*pi*cell_rad/ratio**2 * n**2
        J['air_cutout_area', 'ratio'] = -2*pi*n**2*cell_rad**2 / ratio**3
        J['air_cutout_area', 'n'] = (pi*(cell_rad/ratio)**2)*n*2

        J['area', 'cell_rad'] = d_solid_area__d_cell_rad - 2*pi*cell_rad*n**2 - 2*pi*cell_rad/ratio**2 * n**2
        J['area', 'extra']    = d_solid_area__d_extra
        J['area', 'n']        = d_solid_area__d_n - (pi*cell_rad**2)*n*2 - (pi*(cell_rad/ratio)**2)*n*2
        J['area', 'ratio']    = d_solid_area__d_ratio + 2*pi*n**2*cell_rad**2 / ratio**3

        J['volume', 'cell_rad'] = (d_solid_area__d_cell_rad - 2*pi*cell_rad*n**2 - 2*pi*cell_rad/ratio**2 * n**2) * length
        J['volume', 'extra']    = (d_solid_area__d_extra) * length 
        J['volume', 'n']        = (d_solid_area__d_n - (pi*cell_rad**2)*n*2 - (pi*(cell_rad/ratio)**2)*n*2) * length 
        J['volume', 'ratio']    = (d_solid_area__d_ratio + 2*pi*n**2*cell_rad**2 / ratio**3) * length 
        J['volume', 'length']   = area 

        J['mass', 'cell_rad']   = (d_solid_area__d_cell_rad - 2*pi*cell_rad*n**2 - 2*pi*cell_rad/ratio**2 * n**2) * length*al_density
        J['mass', 'extra']      = (d_solid_area__d_extra) * length * al_density
        J['mass', 'n']          = (d_solid_area__d_n - (pi*cell_rad**2)*n*2 - (pi*(cell_rad/ratio)**2)*n*2) * length * al_density
        J['mass', 'ratio']      = (d_solid_area__d_ratio - -2*pi*n**2*cell_rad**2 / ratio**3) * length * al_density
        J['mass', 'length']     = area * al_density
        J['mass', 'al_density'] = area * length



if __name__ == '__main__':
    from openmdao.api import Problem

    nn = 1
    prob=Problem()
    prob.model.add_subsystem('sys1', MetaPackSizeComp(num_nodes=nn), promotes=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)
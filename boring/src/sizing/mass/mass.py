"""
Author: Jeff Chin
"""

import numpy as np
import openmdao.api as om


class MassGroup(om.Group): 
    """sum all individual masses to estimate total mass and mass fractions"""
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='bus',
                           subsys=busMass(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])


        self.add_subsystem(name='mass',
                           subsys=packMass(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])


        self.set_input_defaults('frame_mass', 0.01, units='kg')




class packMass(om.ExplicitComponent):
    """sum all individual masses to estimate total mass and mass fractions"""

    def initialize(self):
        self.options.declare('num_nodes', types=int) #argument for eventual dymos transient model

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('PCM_tot_mass', units='kg', desc='total pack PCM mass') 
        self.add_input('mass_OHP', units='kg', desc='total pack OHP mass')
        self.add_input('frame_mass', units='kg', desc='frame mass per cell')
        self.add_input('n_cells', desc='number of cells')
        self.add_input('cell_mass', 0.0316*2, units='kg', desc='individual cell mass')
        self.add_input('ext_cool_mass', units='kg', desc='mass from external cooling')

        self.add_output('p_mass', desc='inactive pack mass')
        self.add_output('tot_mass', desc='total pack mass')
        self.add_output('mass_frac', desc='fraction of mass not fromt the battery cells')

    def compute(self,i,o):

        o['p_mass'] = i['PCM_tot_mass'] + i['mass_OHP'] + i['frame_mass']*i['n_cells'] + i['ext_cool_mass']
        o['tot_mass'] = o['p_mass'] + i['cell_mass']*i['n_cells']
        o['mass_frac'] = o['p_mass']/o['tot_mass']

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

class frameMass(om.ExplicitComponent):
    """Calculate the mass of the frame per cell"""
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('bar_mass', desc='bus bar mass', units='kg')
        self.add_input('d', desc='insulation thickness', units='mm')

        # Outputs
        self.add_output('frame_mass', desc='inactive structural mass per cell', units='kg')
    
    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')


class busMass(om.ExplicitComponent):
    """Calculate the mass of the bus bar per cell"""
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('t_bar', val=0.03*np.ones(nn), desc='bus bar thickness', units='mm')
        self.add_input('rho_bar', val=2.7*np.ones(nn), desc='bus bar density', units='g/cm**3')
        self.add_input('cell_w', val=0.0571, units='m' , desc='cell length (2.0" Amprius)')
        self.add_input('cell_h', val=0.00635, units='m' , desc='cell thickness (0.25" Amprius)')

        # Outputs
        self.add_output('lead_area', desc='area above the leads', units='cm')
        self.add_output('bar_mass', desc='bus bar mass', units='kg')


    def compute(self,i,o):

        o['lead_area'] = i['cell_h']*i['cell_w']
        o['bar_mass'] = i['t_bar']*o['lead_area']*i['rho_bar']


    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')


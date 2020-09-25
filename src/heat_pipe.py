
#https://www.digikey.com/product-detail/en/wakefield-vette/126114/345-126114-ND/12150524
# 3mm x 17.14mm x 600mm
# (flattened from a 12mm diam pipe)

#http://www.wakefield-vette.com/Portals/0/heat%20pipe%20bending%20tool/Wakefield%20Vette%20Heat%20Pipe%20Selection%20and%20Design%20Guide%20R3%202020.pdf

# Small-Sized Pulsating Heat Pipes/Oscillating Heat
# Pipes with Low Thermal Resistance and High Heat
# Transport Capability


# https://www.electronics-cooling.com/2016/08/design-considerations-when-using-heat-pipes/

import numpy as np
import openmdao.api as om
from math import pi


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



class FHP(om.ExplicitComponent):
    """flat heat pipe calculation"""

    def initialize(self):
        self.options.declare('num_nodes', types=int) 

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('d_init', val=12, units='mm', desc='initial diameter of the heat pipe before flattening') 
        self.add_input('ref_len', val=240, units='mm', desc='reference length for the regression')
        self.add_input('tot_len', val=600, units='mm', desc='total heat pipe length')
        self.add_input('flux', val=50, units='W', desc='required heat flux')
        self.add_input('rho_FHP', val=200, units='kg/m**3', desc='bulk density of the flat heat pipe')

        self.add_input('p_flux', units='W', desc='single FHP flux capability')
        self.add_output('p_mass', units='kg', desc='mass of a single pipe')
        self.add_output('n_pipes')
        self.add_output('fhp_mass')

    def compute(self,i,o):

        D = i['d_init']
        L_scale = (i['tot_len']+50) / i['ref_len']
        
        o['p_flux'] = (0.7335*D**2 - 2.3294*D + 8.7876)/L_scale
        o['p_mass'] = i['tot_len']*pi/4.*i['d_init']**2
        o['n_pipes'] = i['req_flux']/p_flux
        o['fhp_mass'] = o['n_pipes']*o['p_mass'] 


class OHP(om.ExplicitComponent):
    """ Sizing the Heat Pipe """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # inputs
        self.add_input('frame_mass', 0.01, units='kg', desc='frame mass per cell')
        self.add_input('cell_area', 0.102*0.0571, units='m**2', desc='cell area')
        self.add_input('runawayJ', 48., units='kJ', desc='Runaway heat of a 18650 battery')
        self.add_input('dur', 45.0, units='s', desc='runaway event duration')
        self.add_input('n_cells', 320, desc='number of cells')

        # outputs
        self.add_output('flux', units='W', desc='Heat Load through Heat Pipe')
        self.add_output('mass_OHP', units='kg', desc='Total Pack OHP mass')
        self.add_output('Areal_weight', units='kg/m**2', desc='Oscillating Heat Pipe Areal Weight')

    def compute(self, i, o):

        o['flux'] = i['runawayJ']*1000/i['dur'] #heat flux during runaway
        o['Areal_weight'] = 0.3866*o['flux']+1.7442 # NH3   kg/m^2
        o['mass_OHP'] = o['Areal_weight']*i['cell_area']*i['n_cells']/2

    def compute_partials(self, inputs, J):
        pass #ToDo once calculations are complete

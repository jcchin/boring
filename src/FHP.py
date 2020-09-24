
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



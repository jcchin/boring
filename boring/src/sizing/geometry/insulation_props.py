"""
Author: Jeff Chin

http://www.aerogeltechnologies.com/airloy/airloy-product-selection-guide/
T = High Thermal Limit
H = Hydrophobic
 
110 = Polyimide (300C)
120 = Polyamide (250C)
 
Density classes
L = 0.1 g/cc (light, but easy to break)
M = 0.2 g/cc (difficult to break) <-- this seems like the sweet spot
H = 0.4 g/cc (heavy but not brittle)
 
Airloy T116-L looks good except it's rigid/breakable
22 mW/m-K
0.1 g/cc
stability up to 400C (withstand flames 3000+F)
easy to machine

$90 per 2.5″ x 3″ x 0.4″ tile, or $490 per 12x12x0.4" panel (for reference our cells are 2.25" x 2" x 0.25")
http://www.buyaerogel.com/product/airloy-x116-large-panels/
(it can also be purchased as a flexible thin wrap)
"""

from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import dymos as dm
from dymos.examples.plotting import plot_results


class tempODE(om.ExplicitComponent):
    """Calculate the temperature rise between cells with heat flux across the insulation thickness
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('K', val=0.03*np.ones(nn), desc='insulation conductivity', units='W/m*K')
        self.add_input('A', val=.102*.0003*np.ones(nn), desc='battery side area', units='m**2')
        self.add_input('d', val=0.03*np.ones(nn), desc='insulation thickness', units='m')
        self.add_input('m', val=0.06*np.ones(nn), desc='cell mass', units='kg')
        self.add_input('Cp', val=3.56*np.ones(nn), desc='cell specific heat capacity', units='kJ/kg*K')
        self.add_input('Th', val=773.*np.ones(nn), desc='hot battery side temp', units='K')
        self.add_input('T', val=373.*np.ones(nn), desc='cold battery side temp', units='K')

        # Outputs
        self.add_output('Tdot', val=np.zeros(nn), desc='temp rate of change', units='K/s')

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])
        c = np.zeros(self.options['num_nodes'])
        self.declare_partials('Tdot', ['K','A','d','m','Cp','Th','T'], rows=arange, cols=arange)
        # self.declare_partials(of='*', wrt='*', method='cs') # use this if you don't provide derivatives

    def compute(self, i, o):

        dT_num = i['K']*i['A']*(i['Th']-i['T'])/i['d']
        dT_denom = i['m']*i['Cp']
        o['Tdot'] = dT_num/dT_denom

    def compute_partials(self, i, partials):
    
        partials['Tdot','T'] = -i['K']*i['A']/(i['d']*i['m']*i['Cp'])
        partials['Tdot','K']  = i['A']*(i['Th']-i['T'])/(i['d']*i['m']*i['Cp'])
        partials['Tdot','A']  = i['K']*(i['Th']-i['T'])/(i['d']*i['m']*i['Cp'])
        partials['Tdot','Th'] = i['K']*i['A']/(i['d']*i['m']*i['Cp'])
        partials['Tdot','d']  = -i['K']*i['A']*(i['Th']-i['T'])/(i['m']*i['Cp']*i['d']**2)
        partials['Tdot','m']  = -i['K']*i['A']*(i['Th']-i['T'])/(i['d']*i['Cp']*i['m']**2)
        partials['Tdot','Cp'] = -i['K']*i['A']*(i['Th']-i['T'])/(i['d']*i['m']*i['Cp']**2)

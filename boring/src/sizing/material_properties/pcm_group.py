import numpy as np
from math import pi

import openmdao.api as om

from boring.src.sizing.material_properties.pcm_ps import PCM_PS
from boring.src.sizing.material_properties.cp_func import PCM_Cp
from boring.src.sizing.material_properties.pcm_properties import PCM_props
from boring.src.sizing.thermal_network import TempRateComp


class PCM_Group(om.Group):
    """ Computes PCM pad bulk properties, percent solid, and state (temp) rates"""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)  

    def setup(self):
        nn = self.options['num_nodes']


        self.add_subsystem(name = 'cp',
                          subsys = PCM_Cp(num_nodes=nn),
                          promotes_inputs=['T','T_lo','T_hi'],
                          promotes_outputs=['cp_pcm'])

        self.add_subsystem(name = 'bulk',
                          subsys = PCM_props(num_nodes=nn),
                          promotes_inputs=['cp_pcm'],
                          promotes_outputs=['cp_bulk'])

        self.add_subsystem(name = 'ps',
                            subsys = PCM_PS(num_nodes=nn),
                            promotes_inputs=['T','T_lo','T_hi'],
                            promotes_outputs=['PS'])

        self.add_subsystem(name = 'rate',
                           subsy = TempRateComp(num_nodes=nn),
                           promotes_inputs=['(c_p,cp_bulk)'],
                           promotes_outputs=['Tdot'])

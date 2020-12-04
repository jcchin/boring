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
                           subsys = TempRateComp(num_nodes=nn),
                           promotes_inputs=[('c_p','cp_bulk')],
                           promotes_outputs=['Tdot'])

if __name__ == "__main__":
    p = om.Problem(model=om.Group())
    nn = 1

    p.model.add_subsystem(name='pcm',
                          subsys=PCM_Group(num_nodes=nn),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
    
    p.setup(force_alloc_complex=True)
    
    p['T'] = 334
    p['T_lo'] = 333
    p['T_hi'] = 338

    p.check_partials(compact_print=True)

    p.run_model()
    om.n2(p)
    # om.view_connections(p)
    p.model.list_inputs(values=True, prom_name=True)   
    p.model.list_outputs(values=True, prom_name=True) 
    print('Finished Successfully')

    print('\n', '\n')
    print('--------------Outputs---------------')
    print('The Percent Solid is ......', p.get_val('PS'))
    print('\n', '\n')
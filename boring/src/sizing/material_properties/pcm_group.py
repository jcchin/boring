import numpy as np

import openmdao.api as om

from boring.src.sizing.material_properties.pcm_ps import PCM_PS
from boring.src.sizing.material_properties.pcm_properties import PCM_props
from boring.src.sizing.material_properties.cp_func import PCM_Cp


class TempRateComp(om.ExplicitComponent):
    """Computes temperature rise"""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('q', val=np.ones(nn), desc= 'heat flux', units='W')
        self.add_input('mass', val=np.ones(nn), desc='cell mass', units='kg')
        self.add_input('c_p', val=np.ones(nn), desc='cell specific heat', units='J/(kg*K)')

        self.add_output('Tdot', val=np.ones(nn), desc='change in temperature per second', units='K/s')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.declare_partials('Tdot', 'q', rows=ar, cols=ar)
        self.declare_partials('Tdot', 'mass', rows=ar, cols=ar)
        self.declare_partials('Tdot', 'c_p', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        q = inputs['q']
        mass = inputs['mass']
        c_p = inputs['c_p']

        outputs['Tdot'] = -q / mass / c_p

    def compute_partials(self, inputs, J):
        q = inputs['q']
        mass = inputs['mass']
        c_p = inputs['c_p']

        J['Tdot', 'q'] = -1 / mass / c_p
        J['Tdot', 'mass'] = q / mass ** 2 / c_p
        J['Tdot', 'c_p'] = q / mass / c_p ** 2


class PCM_Group(om.Group):
    """ Computes PCM pad bulk properties, percent solid, and state (temp) rates"""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='cp',
                           subsys=PCM_Cp(num_nodes=nn),
                           promotes_inputs=['T', 'T_lo', 'T_hi'],
                           promotes_outputs=['cp_pcm'])

        self.add_subsystem(name='bulk',
                           subsys=PCM_props(num_nodes=nn),
                           promotes_inputs=['cp_pcm'],
                           promotes_outputs=['cp_bulk'])

        self.add_subsystem(name='ps',
                           subsys=PCM_PS(num_nodes=nn),
                           promotes_inputs=['T', 'T_lo', 'T_hi'],
                           promotes_outputs=['PS'])

        self.add_subsystem(name='rate',
                           subsys=TempRateComp(num_nodes=nn),
                           promotes_inputs=[('c_p', 'cp_bulk'), 'q', 'mass'],
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
    # om.n2(p)
    # om.view_connections(p)
    p.model.list_inputs(values=True, prom_name=True)
    p.model.list_outputs(values=True, prom_name=True)
    print('Finished Successfully')

    print('\n', '\n')
    print('--------------Outputs---------------')
    print('The Percent Solid is ......', p.get_val('PS'))
    print('\n', '\n')

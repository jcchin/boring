""""
This group instantiates "n" radial stacks, and then links them.

Calculate steady-state heat pipe performance by converging the following subsystems

1) Calculate the fluid properties based on temeprature
2) Caclulate the resistances in each part of the heat pipe
3) Construct the equivalent thermal resistance network to determine the temperatures (40 connections per battery pair)

(repeat until convergence)

The network can include PCM pads, with an option for round or flat heatpipes,
and the network is constructed using a for loop to handle n cells in the system.

Author: Dustin Hall, Jeff Chin, Sydney Schnulo
"""
import openmdao.api as om
import numpy as np

from boring.src.sizing.thermal_network import Radial_Stack, thermal_link
from boring.src.sizing.material_properties.pcm_group import TempRateComp

from boring.src.sizing.material_properties.pcm_group import PCM_Group

# from boring.util.load_inputs import load_inputs

class HeatPipeGroup(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_cells', types=int, default=3)
        self.options.declare('pcm_bool', types=bool, default=False)
        self.options.declare('geom', values=['round', 'flat'], default='round')


    def setup(self):
        nn = self.options['num_nodes']
        n = self.options['num_cells']
        pcm_bool = self.options['pcm_bool']
        geom = self.options['geom']

        n_out = np.append(np.ones(n-1),0)
        n_in = np.append(0, np.ones(n-1))


        for i in np.arange(n):
            # insantiate the radial stacks based on geometry
            if geom == 'round':
                self.add_subsystem('cell_{}'.format(i), Radial_Stack(n_in=int(n_in[i]), n_out=int(n_out[i]), num_nodes=nn, geom=geom),
                                                        promotes_inputs=['XS:D_od', 'XS:t_wk', 'XS:t_w', 'k_w', 'XS:D_v', 'LW:L_adiabatic', 'alpha'])
            if geom == 'flat':
                self.add_subsystem('cell_{}'.format(i), Radial_Stack(n_in=int(n_in[i]), n_out=int(n_out[i]), num_nodes=nn, geom=geom),
                                                        promotes_inputs=['W', 'XS:t_wk', 'XS:t_w', 'k_w', 'LW:L_adiabatic', 'alpha'])
            # add temp rate comps
            if pcm_bool:
                self.add_subsystem(name='T_rate_pcm_{}'.format(i),
                               subsys=PCM_Group(num_nodes=nn))
            else:   
                self.add_subsystem(name='T_rate_cell_{}'.format(i),
                               subsys=TempRateComp(num_nodes=nn))

            # connect external flux
            if pcm_bool:
                self.connect('cell_{}.Rex.q'.format(i), 'T_rate_pcm_{}.q'.format(i))
            else:
                self.connect('cell_{}.Rex.q'.format(i), 'T_rate_cell_{}.q'.format(i))

        for j in range(n-1):

            thermal_link(self, 'cell_{}'.format(j), 'cell_{}'.format(j+1), num_nodes=nn, geom=geom)

            self.connect('cell_0_bridge.k_wk', 'cell_{}.k_wk'.format(j))

        self.connect('cell_0_bridge.k_wk', 'cell_{}.k_wk'.format(n-1))

        self.set_input_defaults('k_w', 11.4 * np.ones(nn), units='W/(m*K)')
        self.set_input_defaults('epsilon', 0.46 * np.ones(nn), units=None)
        self.set_input_defaults('LW:L_flux', 0.02 * np.ones(nn), units='m')
        self.set_input_defaults('LW:L_adiabatic', 0.03 * np.ones(nn), units='m')
        self.set_input_defaults('XS:t_w', 0.5 * np.ones(nn), units='mm')
        self.set_input_defaults('XS:t_wk', 0.69 * np.ones(nn), units='mm')

        if pcm_bool: # manually set mass for debugging
            self.set_input_defaults('T_rate_pcm_1.mass', 0.003*np.ones(nn), units='kg')
            self.set_input_defaults('T_rate_pcm_0.mass', 0.003*np.ones(nn), units='kg')

        if geom == 'round':
            self.set_input_defaults('XS:D_od', 6. * np.ones(nn), units='mm')
            self.set_input_defaults('XS:D_v', 3.62 * np.ones(nn), units='mm')

        elif geom == 'flat':
            self.set_input_defaults('H', 20. * np.ones(nn), units='mm')
            self.set_input_defaults('W', 20. * np.ones(nn), units='mm')

        # load_inputs('boring.input.assumptions2', self, nn)


if __name__ == "__main__":
    p = om.Problem(model=om.Group())
    nn = 1

    num_cells_tot = 2

    p.model.add_subsystem(name='hp',
                          subsys=HeatPipeGroup(num_nodes=nn, num_cells=num_cells_tot, pcm_bool=False, geom='round'),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

    p.setup(force_alloc_complex=True)

    T_in = 20 * np.ones(num_cells_tot)
    T_in[1] = 100

    for x in np.arange(num_cells_tot):
        p['cell_{}.Rex.T_in'.format(x)] = T_in[x]
        p['cell_{}.L_flux'.format(x)] = 0.02
        p['cell_{}.Rex.R'.format(x)] = [0.0001],

    p.run_model()

    # om.view_connections(p)
    # p.model.list_inputs(values=True, prom_name=True)
    # p.model.list_outputs(values=True, prom_name=True)
    print('Finished Successfully')

    print('\n', '\n')
    print('--------------Outputs---------------')
    print('The HEATPIPE mass is ......', p.get_val('mass_heatpipe'))
    print('The LIQUID mass is.........', p.get_val('mass_liquid'))
    print('The WICK mass is...........', p.get_val('mass_wick'))
    print('\n', '\n')

    show_plots = True

    if show_plots:
        import matplotlib.pyplot as plt

        cells = np.arange(num_cells_tot)
        T_cells_1 = np.ones(len(cells))
        T_cells_2 = np.ones(len(cells))
        T_cells_3 = np.ones(len(cells))
        T_cells_4 = np.ones(len(cells))

        for i in cells:

            T_cells_1[i] = p.get_val('cell_{}.n1.T'.format(i))[0]
            T_cells_2[i] = p.get_val('cell_{}.n2.T'.format(i))[0]
            T_cells_3[i] = p.get_val('cell_{}.n3.T'.format(i))[0]
            T_cells_4[i] = p.get_val('cell_{}.n4.T'.format(i))[0]

        plt.plot(cells, T_cells_1, 'o', label='node 1')
        plt.plot(cells, T_cells_2, 'o', label='node 2')
        plt.plot(cells, T_cells_3, 'o', label='node 3')
        plt.plot(cells, T_cells_4, 'o', label='node 4')
        plt.xlabel('cell')
        plt.ylabel('T, K')
        plt.legend()

        plt.show()

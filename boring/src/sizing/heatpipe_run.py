""""
Calculate steady-state heat pipe performance by converging the following subsystems

1) Calculate the fluid properties based on temeprature
2) Caclulate the resistances in each part of the heat pipe
3) Construct the equivalent thermal resistance network to determine the temperatures (40 connections per battery pair)

(repeat until convergence)


Author: Dustin Hall, Jeff Chin
"""
import openmdao.api as om
import numpy as np

from boring.src.sizing.thermal_network import Radial_Stack, thermal_link, TempRateComp
from boring.src.sizing.mass.mass import heatPipeMass

from boring.util.load_inputs import load_inputs


class HeatPipeGroup(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_cells', types=int, default=3)
        self.options.declare('pcm_bool', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        n = self.options['num_cells']
        pcm_bool = self.options['pcm_bool']

        n_out = np.append(np.ones(n-1),0)
        n_in = np.append(0, np.ones(n-1))

        print(n_out)

        for i in np.arange(n):

            self.add_subsystem('cell_{}'.format(i), Radial_Stack(n_in=int(n_in[i]), n_out=int(n_out[i]), num_nodes=nn, pcm_bool=pcm_bool),
                                                    promotes_inputs=['D_od', 't_wk', 't_w', 'k_w', 'D_v', 'L_adiabatic', 'alpha'])

            self.add_subsystem(name='T_rate_cell_{}'.format(i),
                               subsys=TempRateComp(num_nodes=nn))

            self.connect('cell_{}.Rex.q'.format(i), 'T_rate_cell_{}.q'.format(i))

        self.add_subsystem(name='hp_mass',
                           subsys=heatPipeMass(num_nodes=nn),
                           promotes_inputs=['D_od','D_v','L_heatpipe','t_w','t_wk','cu_density',('fill_wk','epsilon'),'liq_density','fill_liq'],
                           promotes_outputs=['mass_heatpipe', 'mass_wick', 'mass_liquid'])

        for j in range(n-1):

            thermal_link(self, 'cell_{}'.format(j), 'cell_{}'.format(j+1))

            self.connect('cell_0_bridge.k_wk', 'cell_{}.k_wk'.format(j))

        self.connect('cell_0_bridge.k_wk', 'cell_{}.k_wk'.format(n-1))

        load_inputs('boring.input.assumptions2', self, nn)


if __name__ == "__main__":
    p = om.Problem(model=om.Group())
    nn = 1

    num_cells_tot = 15
    cell_therm_run = 5

    p.model.add_subsystem(name='hp',
                          subsys=HeatPipeGroup(num_nodes=nn, num_cells=num_cells_tot, pcm_bool=False),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

    p.setup(force_alloc_complex=True)

    p['L_eff'] = (0.02 + 0.1) / 2. + 0.03

    T_in = 20 * np.ones(num_cells_tot)
    T_in[cell_therm_run] = 100

    for x in np.arange(num_cells_tot):
        p['cell_{}.Rex.T_in'.format(x)] = T_in[x]

    # p['cell_0.Rex.T_in'] = 100
    # p['cell_1.Rex.T_in'] = 20
    # p['cell_2.Rex.T_in'] = 20

        p['cell_{}.L_flux'.format(x)] = 0.02
        # p['cell_{}.pcm.cp_bulk'.format(x)] = 1500,  # Specific Heat
        # p['cell_{}.pcm.mass'.format(x)] = .06,
        p['cell_{}.Rex.R'.format(x)] = [0.0001],

    # p.set_val('L_adiabatic',0.03)
    # p.set_val('t_w',0.0005)
    # p.set_val('t_wk',0.00069)
    # p.set_val('D_od', 0.006)
    # p.set_val('D_v',0.00362)
    # p.set_val('Q_hp',1)
    # p.set_val('h_c',1200)
    # p.set_val('T_coolant',293)

    p.check_partials(compact_print=True)

    p.run_model()
    # om.n2(p)
    # om.view_connections(p)
    p.model.list_inputs(values=True, prom_name=True)
    p.model.list_outputs(values=True, prom_name=True)
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

            T_cells_1[i] = p.get_val('cell_{}.n1.T'.format(i))
            T_cells_2[i] = p.get_val('cell_{}.n2.T'.format(i))
            T_cells_3[i] = p.get_val('cell_{}.n3.T'.format(i))
            T_cells_4[i] = p.get_val('cell_{}.n4.T'.format(i))

        plt.plot(cells, T_cells_1, 'o', label='node 1')
        plt.plot(cells, T_cells_2, 'o', label='node 2')
        plt.plot(cells, T_cells_3, 'o', label='node 3')
        plt.plot(cells, T_cells_4, 'o', label='node 4')
        plt.xlabel('cell')
        plt.ylabel('T, K')
        plt.legend()

        plt.show()

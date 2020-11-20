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

from boring.src.sizing.thermal_network import Radial_Stack, thermal_link


class HeatPipeRun(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('evap', Radial_Stack(n_in=0, n_out=1),
                               promotes_inputs=['D_od','t_wk','t_w','k_wk','k_w','D_v','L_adiabatic','alpha']) # promote shared values (geometry, mat props)
        self.add_subsystem('cond', Radial_Stack(n_in=1, n_out=0),
                               promotes_inputs=['D_od','t_wk','t_w','k_wk','k_w','D_v','L_adiabatic','alpha'])

        thermal_link(self,'evap','cond')

        self.set_input_defaults('k_w',11.4)
        self.set_input_defaults('evap.Rex.R', 0.0000001)
        self.set_input_defaults('cond.Rex.R', 0.0000001)

        self.set_input_defaults('cond.L_flux', 0.02)
        self.set_input_defaults('evap.L_flux', 0.01)
        self.set_input_defaults('L_adiabatic', 0.03)
        self.set_input_defaults('t_wk', 0.00069)
        self.set_input_defaults('t_w', 0.0005)
        self.set_input_defaults('D_od', 0.006)
        self.set_input_defaults('k_w', 11.4)
        self.set_input_defaults('epsilon', 0.46)
        self.set_input_defaults('D_v', 0.00362)
        self.set_input_defaults('L_eff', 0.045)


if __name__ == "__main__":
    p = om.Problem(model=om.Group())
    nn = 1

    p.model.add_subsystem(name='hp',
                          subsys=HeatPipeRun(num_nodes=nn),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
    
    p.setup()
    
    p['L_eff'] = (0.02+0.1)/2.+0.03
    p['evap.Rex.T_in'] = 100
    p['cond.Rex.T_in'] = 20

    # p.set_val('L_evap',0.01)
    # p.set_val('L_cond',0.02)
    # p.set_val('L_adiabatic',0.03)
    # p.set_val('t_w',0.0005)
    # p.set_val('t_wk',0.00069)
    # p.set_val('D_od', 0.006)
    # p.set_val('D_v',0.00362)
    # p.set_val('Q_hp',1)
    # p.set_val('h_c',1200)
    # p.set_val('T_coolant',293)

    # p.check_partials(compact_print=True)

    p.run_model()
    #om.n2(p)
    om.view_connections(p)
    p.model.list_inputs(values=True, prom_name=True)   
    p.model.list_outputs(values=True, prom_name=True) 
    print('Finished Successfully')

    print('\n', '\n')
    print('--------------Outputs---------------')
    print('The r_h Value is.......... ', p.get_val('evap_bridge.vapor.r_h'))
    print('The R_v Value is.......... ', p.get_val('evap_bridge.vapor.R_v'))
    print('\n', '\n')
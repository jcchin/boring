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

from geometry.hp_geometry import HeatPipeSizeGroup
from mass.mass import MassGroup

from boring.src.sizing.thermal_network import Circuit, Radial_Stack, thermal_link


class HeatPipeRun(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name = 'sizing',
                          subsys = HeatPipeSizeGroup(num_nodes=nn),
                          promotes_inputs=['L_evap', 'L_cond', 'L_adiabatic', 't_w', 't_wk', 'D_od', 'D_v',
                                           't_w', 'D_v', 'L_cond', 'L_evap'],
                          promotes_outputs=['r_i', 'A_cond', 'A_evap', 'L_eff', 'A_w', 'A_wk', ('A_interc', 'A_inter'),'A_intere'])


        # need a new radial stack instance for every battery
        self.add_subsystem(name= 'evap', 
                           subsys = Radial_Stack(num_nodes=nn, n_in=0, n_out=1),
                           promotes_inputs=['T_hp','v_fg','D_od','R_g','P_v','k_wk','A_inter','k_w','L_cond','r_i','D_v','h_fg','alpha','A_cond'],
                           promotes_outputs=[])

        self.add_subsystem(name= 'cond', 
                           subsys= Radial_Stack(num_nodes=nn, n_in=1, n_out=0),
                           promotes_inputs=['T_hp','v_fg','D_od','R_g','P_v','k_wk','A_inter','k_w','L_cond','r_i','D_v','h_fg','alpha','A_cond'],
                           promotes_outputs=[])

        # self.add_subsystem(name = 'cond2',
        #                    subsys = Radial_Stack(n_in=1, n_out=0),
        #                    )
        
        thermal_link(self,'evap','cond') # this creates all the axial connections between radial stacks
        #thermal_link(self,'cond','cond2')



        # Connect fluid props to resistance calcs
        #self.connect('')

        # Connect resistance vals to the resistor network


        # Connect node temperatures back to fluid props

        self.set_input_defaults('k_w', val=11.4*np.ones(nn))
        self.set_input_defaults('L_evap', 6.0, units='mm')
        self.set_input_defaults('L_cond', 5, units='mm')
        self.set_input_defaults('D_v', 0.5, units='mm')
        self.set_input_defaults('D_od', 2, units='mm')
        self.set_input_defaults('t_w', 0.01, units='mm')
        self.set_input_defaults('L_adiabatic', 0.01, units='mm')




if __name__ == "__main__":
    p = om.Problem(model=om.Group())
    nn = 1

    p.model.add_subsystem(name='hp',
                          subsys=HeatPipeRun(num_nodes=nn),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
    

    p.setup()

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

    om.n2(p)

    p.run_model()
    p.model.list_inputs(values=True, prom_name=True)   
    p.model.list_outputs(values=True, prom_name=True) 
    print('Finished Successfully')

    print('\n', '\n')
    print('--------------Outputs---------------')
    print('The r_h Value is.......... ', p.get_val('evap_bridge.vapor.r_h'))
    print('The R_v Value is.......... ', p.get_val('evap_bridge.vapor.R_v'))
    print('\n', '\n')
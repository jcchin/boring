""""
Calculate steady-state heat pipe performance by converging the following subsystems

1) Calculate the fluid properties based on temeprature
2) Caclulate the resistances in each part of the heat pipe
3) Construct the equivalent thermal resistance network to determine the temperatures (40 connections per battery pair)

(repeat until convergence)


Author: Dustin Hall, Jeff Chin
"""
import openmdao.api as om

from material_properties.fluid_properties import FluidPropertiesComp
from geometry.hp_geometry import HeatPipeSizeGroup
from mass.mass import MassGroup

from boring.src.sizing.thermal_resistance.axial_thermal_resistance import AxialThermalResistance
from boring.src.sizing.thermal_resistance.vapor_thermal_resistance import VaporThermalResistance
from boring.src.sizing.thermal_resistance.radial_thermal_resistance import RadialThermalResistance


class HeatPipeRun(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name = 'sizing',
                          subsys = HeatPipeSizeGroup(num_nodes=nn),
                          promotes_inputs=['L_evap', 'L_cond', 'L_adiabatic', 't_w', 't_wk', 'D_od', 'D_v',
                                            'D_od', 't_w', 'D_v', 'L_cond', 'L_evap'],
                          promotes_outputs=['r_i', 'A_cond', 'A_evap', 'L_eff', 'A_w', 'A_wk', 'A_interc', 'A_intere'])

        self.add_subsystem(name = 'fluids',
                          subsys = FluidPropertiesComp(num_nodes=nn),
                          promotes_inputs=['Q_hp', 'A_cond', 'h_c', 'T_coolant'],
                          promotes_outputs=['R_g', 'P_v', 'T_cond', 'T_hp', 'rho_v', 'mu_v', 'h_fg'])
    
        self.add_subsystem(name='axial',
                           subsys=AxialThermalResistance(num_nodes=nn),
                           promotes_inputs=['epsilon', 'k_w', 'k_l', 'L_adiabatic', 'A_w', 'A_wk'],
                           promotes_outputs=['k_wk', 'R_aw', 'R_awk'])

        self.add_subsystem(name='vapor',
                           subsys=VaporThermalResistance(num_nodes=nn),
                           promotes_inputs=['D_v', 'R_g', 'mu_v', 'T_hp', 'h_fg', 'P_v', 'rho_v', 'L_eff'],
                           promotes_outputs=['r_h', 'R_v'])

        self.add_subsystem(name='radial',
                           subsys=RadialThermalResistance(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name = 'mass',
                          subsys=MassGroup(num_nodes=nn),
                          promotes=['*'])



        # Connect fluid props to resistance calcs
        #self.connect('')

        # Connect resistance vals to the resistor network


        # Connect node temperatures back to fluid props

        # self.set_input_defaults('L_evap', 6, units='m')
        # self.set_input_defaults('L_cond', 5, units='m')
        # self.set_input_defaults('D_v', 0.5, units='m')
        # self.set_input_defaults('D_od', 2, units='m')
        # self.set_input_defaults('t_w', 0.01, units='m')
        # self.set_input_defaults('L_adiabatic', 0.01, units='m')




if __name__ == "__main__":
    p = om.Problem()
    model = p.model
    nn = 1

    
    

    p.setup()

    p.set_val('L_evap',0.01)
    p.set_val('L_cond',0.02)
    p.set_val('L_adiabatic',0.03)
    p.set_val('t_w',0.0005)
    p.set_val('t_wk',0.00069)
    p.set_val('D_od', 0.006)
    p.set_val('D_v',0.00362)
    p.set_val('Q_hp',1)
    p.set_val('h_c',1200)
    p.set_val('T_coolant',293)

    # p.check_partials(compact_print=True)

    om.n2(p)

    p.run_model()

    print('Finished Successfully')

    print('\n', '\n')
    print('--------------Outputs---------------')
    print('The r_h Value is.......... ', p.get_val('r_h'))
    print('The R_v Value is.......... ', p.get_val('R_v'))
    print('\n', '\n')
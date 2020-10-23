import openmdao.api as om

# from geometry import SizeComp
from fluid_properties import FluidPropertiesComp
from vapor_thermal_resistance import VapThermResComp
from geometry import SizeGroup
from axial_thermal_resistance import AxialThermalResistance
from mass import MassGroup

if __name__ == "__main__":
    p = om.Problem()
    model = p.model
    nn = 1



    p.model.add_subsystem(name = 'sizing',
                          subsys = SizeGroup(num_nodes=nn),
                          promotes_inputs=['L_evap', 'L_cond', 'L_adiabatic', 't_w', 't_wk', 'D_od', 'D_v',
                                            'D_od', 't_w', 'D_v', 'L_cond', 'L_evap'],
                          promotes_outputs=['r_i', 'A_cond', 'A_evap', 'L_eff', 'A_w', 'A_wk', 'A_interc', 'A_intere'])

    p.model.add_subsystem(name = 'fluids',
                          subsys = FluidPropertiesComp(num_nodes=nn),
                          promotes_inputs=['Q_hp', 'A_cond', 'h_c', 'T_coolant'],
                          promotes_outputs=['R_g', 'P_v', 'T_cond', 'T_hp', 'rho_v', 'mu_v', 'h_fg'])
    
    p.model.add_subsystem(name = 'vapors',
                          subsys = VapThermResComp(num_nodes=nn),
                          promotes_inputs=['R_g', 'mu_v', 'T_hp', 'h_fg', 'P_v', 'rho_v', 'L_eff', 'D_v'],
                          promotes_outputs=['r_h', 'R_v'])

    p.model.add_subsystem(name = 'axialtherm',
                          subsys = AxialThermalResistance(num_nodes=nn),
                          promotes_inputs=['epsilon', 'k_w', 'k_l', 'L_adiabatic', 'A_w', 'A_wk',],
                          promotes_outputs=['k_wk', 'R_aw', 'R_awk'])

    p.model.add_subsystem(name = 'mass',
                          subsys=MassGroup(num_nodes=nn),
                          promotes=['*'])
    

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


p.run_model()

print('Finished Successfully')

print('\n', '\n')
print('--------------Outputs---------------')
print('The r_h Value is.......... ', p.get_val('r_h'))
print('The R_v Value is.......... ', p.get_val('R_v'))
print('\n', '\n')
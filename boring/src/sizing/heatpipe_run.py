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

class HeatPipeRun(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('evap', Radial_Stack(n_in=0, n_out=1, num_nodes=nn),
                               promotes_inputs=['D_od','t_wk','t_w','k_wk','k_w','D_v','L_adiabatic','alpha']) # promote shared values (geometry, mat props)
        self.add_subsystem('cond', Radial_Stack(n_in=1, n_out=0,  num_nodes=nn),
                               promotes_inputs=['D_od','t_wk','t_w','k_wk','k_w','D_v','L_adiabatic','alpha'])
        # self.add_subsystem(name='T_rate_evap',
        #                    subsys=TempRateComp(num_nodes=nn),
        #                    promotes_inputs=['q'],
        #                    promotes_outputs=['T_dot'])
        self.add_subsystem(name='T_rate_cond',
                           subsys=TempRateComp(num_nodes=nn),
                           promotes_inputs=[('q', 'q_cond')],
                           promotes_outputs=[('Tdot', 'Tdot_cond')])

        self.add_subsystem(name='hp_mass',
                           subsys=heatPipeMass(num_nodes=nn),
                           promotes_inputs=['D_od','D_v','L_heatpipe','t_w','t_wk','cu_density','fill_wk','liq_density','fill_liq'],
                           promotes_outputs=['mass_heatpipe', 'mass_wick', 'mass_liquid'])

        self.connect('cond.Rex.q', 'q_cond')

        thermal_link(self,'evap','cond', num_nodes=nn)

        # self.set_input_defaults('k_w',11.4*np.ones(nn))
        self.set_input_defaults('evap.Rex.R', 0.0001*np.ones(nn))
        self.set_input_defaults('cond.Rex.R', 0.0001*np.ones(nn))

        self.set_input_defaults('cond.L_flux', 0.02*np.ones(nn))
        self.set_input_defaults('evap.L_flux', 0.01*np.ones(nn))
        
        self.set_input_defaults('k_w', 11.4*np.ones(nn))
        self.set_input_defaults('epsilon', 0.46*np.ones(nn))        
        self.set_input_defaults('L_eff', 0.045*np.ones(nn))
        self.set_input_defaults('fill_wk', 0.1*np.ones(nn))         # Porosity of the wick (0=void, 1=solid)
        self.set_input_defaults('liq_density', 1000*np.ones(nn))    # Density of the liquid in HP
        self.set_input_defaults('fill_liq', 0.70*np.ones(nn))       # Fill perentage of liquid in HP (1=full, 0=empty)
        self.set_input_defaults('T_rate_cond.c_p', 1500*np.ones(nn))
        self.set_input_defaults('T_rate_cond.mass', .06*np.ones(nn))

        #----------------------------------------Heatpipe Sizing Vars-----------------------------------------#
        """ Currently no check to ensure feasability of dimensions """
        self.set_input_defaults('D_od', 0.006*np.ones(nn))       # Outer diameter of overall heatpipe
        self.set_input_defaults('D_v', 0.00362*np.ones(nn))      # Outer diameter of HP vapor chamber
        self.set_input_defaults('t_wk', 0.00069*np.ones(nn))     # Thickness of the wick in the interior of the HP
        self.set_input_defaults('t_w', 0.0005*np.ones(nn))       # Thickness of the HP wall
        self.set_input_defaults('L_heatpipe', 0.30*np.ones(nn))  # Overall length of the HP (including adiabatic portion)
        self.set_input_defaults('L_adiabatic', 0.03*np.ones(nn)) # Length of the adiabatic section


if __name__ == "__main__":
    p = om.Problem(model=om.Group())
    nn = 1

    p.model.add_subsystem(name='hp',
                          subsys=HeatPipeRun(num_nodes=nn),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
    
    p.setup(force_alloc_complex=True)
    
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

    p.check_partials(compact_print=True)

    p.run_model()
    #om.n2(p)
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
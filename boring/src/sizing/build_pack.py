"""
Top-level optimization

Design Variables:
> Runaway Heat Load (based on cell energy density)
> PCM thickness, PCM porosity
> HP (height, width) or (D_od), HP wall thickness, HP wick thickness, HP wick porosity

Constraints = Max Temp (neighboring cell)

Objective = minimize total mass

Authors: Jeff Chin
"""

# General Imports
import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
# Static Imports
from boring.src.sizing.static_sizing import StaticSizing
# Transient Imports
from boring.src.sizing.pcm_transient import PCM_transient


class Build_Pack(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)  # number of dymos nodes
        self.options.declare('num_cells', types=int, default=3)  # number of cells in the stack
        self.options.declare('pcm_bool', types=bool, default=False)  # boolean for including phase change material 
        self.options.declare('geom', values=['round', 'flat'], default='flat')   # heat pipe geometry


    def setup(self):
        nn = self.options['num_nodes']
        geom = self.options['geom']
        num_cells = self.options['num_cells']
        pcm_bool = self.options['pcm_bool']

        # common design variables
        design_inpts = ['pcm_thickness','pcm_porosity','XS:t_w','XS:t_wk', 'hp_porosity']
        # geometry specific heatpipe variables
        if geom == 'round':
            design_inpts += ['XS:D_v']
        if geom == 'flat':
            design_inpts += ['XS:W_v','XS:H_v']
       
        # Compute all geometry and mass calculations that don't change during a transient
        self.add_subsystem(name='static',
                           subsys= StaticSizing(),
                           promotes_inputs=design_inpts,
                           promotes_outputs=['mass_total'])


        # Run dymos transient, return neighboring temperatures
        self.add_subsystem(name='temp',
                           subsys=PCM_transient(num_nodes=nn, num_cells = num_cells, pcm_bool= True, geom=geom),
                           promotes_inputs=inpts,
                           promotes_outputs=['temp2_data'])

        # Compute temperature distribution/max
        self.add_subsystem(name='temp_distr',
                                subsys=om.ExecComp('temp_ratio = temp2_data/temp3_data'),
                                promotes_inputs=['temp2_data','temp3_data'],
                                promotes_outputs=['temp_distr'])


if __name__ == "__main__":
    p = om.Problem()
    model = p.model
    nn = 1
    geom = 'flat'


    p.model.add_subsystem(name='optimize',
                          subsys=Build_Pack(num_nodes=nn, geom=geom), #,config='honeycomb'
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
    
    p.driver = om.pyOptSparseDriver()

    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['Major optimality tolerance'] = 6e-5
    # p.driver.opt_settings['Major iterations limit'] = 10000
    # p.driver.opt_settings['Major step limit'] = 0.01
    p.set_solver_print(level=2)

    # scaling guide: (DV-ref0)/ref

    # add design variables (7 flat, 6 round)
    p.model.add_design_var('pcm_thickness', lower=1.0, upper=2.02, ref=2e-3)
    p.model.add_design_var('pcm_porosity', lower=0.1, upper=1.0, ref=1)  # porosity of the foam, 1 = completely void, 0 = solid
    if geom == 'flat':
        p.model.add_design_var('XS:H_v', lower=0.003, upper=0.009)
        p.model.add_design_var('XS:W_v', lower=0.003, upper=0.009)
    if geom == 'round':
        p.model.add_design_var('XS:D_v', lower=0.003, upper=0.009)
    p.model.add_design_var('XS:t_w', lower=0.003, upper=0.009)
    p.model.add_design_var('XS:t_wk', lower=0.003, upper=0.009)
    p.model.add_design_var('hp_porosity', lower=0.01, upper=1.0)


    p.model.add_objective('mass_total', ref=1e-2)

    p.model.add_constraint('temp2_data', upper=340, ref=1e2)
    p.model.add_constraint('temp_distr', upper=1.1)


    p.model.linear_solver = om.DirectSolver()

    p.setup(force_alloc_complex=True)


    p.set_val('energy',16., units='kJ')

    # x = 30
    # nrg_list = np.linspace(16.,32.,x)

    nrg_list = np.array([24,])
    x = len(nrg_list)


    opt_mass = np.zeros(x)
    opt_pcm_t = np.zeros(x)
    opt_pcm_porosity = np.zeros(x)
    opt_hp_t_w = np.zeros(x)
    opt_hp_t_wk = np.zeros(x)
    opt_hp_porosity = np.zeros(x)
    density = np.zeros(x)
    opt_success = np.zeros(x)
    opt_temp = np.zeros(x)


    for i, nrg in enumerate(nrg_list):
        p.set_val('energy',nrg_list[i])
        print('current energy: ',nrg_list[i], " (running between 16 - 32)" )

        p.set_val('extra', 1.3)  # 1.3
        p.set_val('ratio', 0.4)  # .75

        p.run_driver()
        p.run_model()
        # p.check_totals(method='cs')
        #opt_success[i]=p.driver.pyopt_solution.optInform['stopCriteria']
        opt_success[i]=p.driver.pyopt_solution.optInform['value']
        print(p.driver.pyopt_solution.optInform)  # print out the convergence success (SNOPT only)
        opt_mass[i]=p.get_val('mass')

    
    cell_dens = 225*((nrg_list*2/3)/12)
    pack_dens = (16*nrg_list*2/3)/(.048*16 + opt_mass)

    #p.run_driver()

    # print('\n \n')
    # print('-------------------------------------------------------')
    # print('Temperature (deg C). . . . . . . . .', p.get_val('temp2_data', units='degK'))  
    # print('Mass (kg). . . . . . . . . . . . . .', p.get_val('mass', units='kg'))  
    # print('-------------------------------------------------------')
    # print('\n \n')


    #  # save to CSV
    df=pd.DataFrame({
                    'mass':opt_mass,
                    'temp':opt_temp,
                    't_ratio':opt_t_ratio,
                    'cell_dens': cell_dens,
                    'pack_dens': pack_dens,
                    'success': opt_success
                    })
    df.to_csv('opt_out.csv',index=False)

    # opt_plots(['opt_out.csv'],x)



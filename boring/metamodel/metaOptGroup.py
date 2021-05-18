""""
Meta Model optimization group

Author: Dustin Hall, Jeff Chin
"""

import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from boring.metamodel.sizing_component import MetaPackSizeComp
from boring.metamodel.training_data import MetaTempGroup
from boring.util.opt_plots import opt_plots

from boring.fenics.fenics_baseline import FenicsBaseline

class MetaOptimize(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('config', types=str, default='')


    def setup(self):
        nn = self.options['num_nodes']
        config = self.options['config']

        if (config == 'honeycomb'):
            inpts = ['energy','extra','ratio']
            outpts = ['temp2_data','temp3_data','mass']
        else: # use manual mass calculations
            self.add_subsystem(name='size',
                               subsys = MetaPackSizeComp(num_nodes=nn),
                               promotes_inputs=['cell_rad', 'extra', 'ratio', 'length','al_density','n'],
                               promotes_outputs=['side','solid_area', 'cell_cutout_area', 'air_cutout_area', 'area', 'volume', 'mass'])
            inpts = ['energy','extra', 'ratio']
            outpts = ['temp2_data','temp3_data']


        self.add_subsystem(name='temp',
                           subsys=MetaTempGroup(num_nodes=nn,config=config),
                           promotes_inputs=inpts,
                           promotes_outputs=outpts)

        # # use FENICS
        # self.add_subsystem(name='baseline_temp',
        #                    subsys=FenicsBaseline(num_nodes=nn),
        #                    promotes_inputs=inpts,
        #                    promotes_outputs=outpts)


        self.add_subsystem(name='temp_ratio',
                           subsys=om.ExecComp('temp_ratio = temp2_data/temp3_data'),
                           promotes_inputs=['temp2_data','temp3_data'],
                           promotes_outputs=['temp_ratio'])

        if (config != 'honeycomb'):
            self.add_subsystem(name='obj',
                               subsys=om.ExecComp('obj = mass + side/80'),
                               promotes_inputs=['mass','side'],
                               promotes_outputs=['obj'])
            self.set_input_defaults('extra',1)
            self.set_input_defaults('ratio',1)
        # else:
        #     self.add_subsystem(name='obj',
        #                        subsys=om.ExecComp('obj = mass + extra'),
        #                        promotes_inputs=['mass','extra'],
        #                        promotes_outputs=['obj'])


if __name__ == "__main__":
    p = om.Problem()
    model = p.model
    nn = 1


    p.model.add_subsystem(name='meta_optimize',
                          subsys=MetaOptimize(num_nodes=nn, config='grid'), #,config='honeycomb'
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
    
    p.driver = om.pyOptSparseDriver()

    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['Major optimality tolerance'] = 6e-5
    # p.driver.opt_settings['Major iterations limit'] = 10000
    # p.driver.opt_settings['Major step limit'] = 0.01
    p.set_solver_print(level=2)

    # p.driver.options['optimizer'] = 'ALPSO'
    # p.driver.opt_settings['SwarmSize'] = 200
    # p.driver.options['print_results'] = True
    # # p.driver.pyopt_solution.optInform["value"]
    # p.driver = om.ScipyOptimizeDriver()

    # (DV-ref0)/ref


    p.model.add_design_var('extra', lower=1.0, upper=2.0, ref=1e3)
    p.model.add_design_var('ratio', lower=0.2, upper=0.8, ref=1e-3) 
    # p.model.add_design_var('resistance', lower=0.003, upper=0.009)
    # p.model.add_objective('obj', ref=1e-2)
    p.model.add_objective('mass', ref=1e-2)
    # p.model.add_objective('side', ref=1)
    p.model.add_constraint('temp2_data', upper=340, ref=1e2)
    p.model.add_constraint('temp_ratio', upper=1.1)
    # p.model.add_constraint('side', upper=120, ref=1e2) # 
    # p.model.add_constraint('solid_area', lower=6000)


    p.model.linear_solver = om.DirectSolver()


    p.setup(force_alloc_complex=True)
    # p.final_setup()
    # om.n2(p)
    #p.set_val('cell_rad', 9, units='mm')
    #p.set_val('resistance', 0.003)
    # p.set_val('extra', 1.4)
    p.set_val('ratio', 0.7)
    p.set_val('energy',16., units='kJ')
    #p.set_val('length', 65.0, units='mm')
    #p.set_val('al_density', 2.7e-6, units='kg/mm**3')
    #p.set_val('n',4)

    x = 30
    nrg_list = np.linspace(1.,48.,x)
    # ------- or ---------------------
    # nrg_list = np.array([24,])
    # x = len(nrg_list)


    opt_mass = np.zeros(x)
    opt_ratio = np.zeros(x)
    opt_spacing = np.zeros(x)
    opt_t_ratio = np.zeros(x)
    opt_res = np.zeros(x)
    density = np.zeros(x)
    opt_success = np.zeros(x)
    opt_side = np.zeros(x)
    opt_temp = np.zeros(x)



    for i, nrg in enumerate(nrg_list):
        p.set_val('energy',nrg_list[i])
        print('current energy: ',nrg_list[i], " (running between 16 - 32)" )

        p.set_val('extra', 1.3)  # 1.3
        p.set_val('ratio', 0.4)# - nrg_list[i]/100)  # .75

        p.run_driver()
        p.run_model()
        # p.check_totals(method='cs')
        #opt_success[i]=p.driver.pyopt_solution.optInform['stopCriteria']
        opt_success[i]=p.driver.pyopt_solution.optInform['value']
        print(p.driver.pyopt_solution.optInform)  # print out the convergence success (SNOPT only)
        opt_mass[i]=p.get_val('mass')
        opt_ratio[i]=p.get_val('ratio')
        opt_spacing[i]=p.get_val('extra')
        opt_t_ratio[i]=p.get_val('temp_ratio')
        # opt_res[i]=p.get_val('resistance')
        #opt_side[i]=p.get_val('side')
        opt_temp[i]=p.get_val('temp2_data')
        # density[i] = 128/(.048*16*12/kJ + mass*(12/kJ))

    
    cell_dens = 225*((nrg_list*2/3)/12)
    pack_dens = (16*nrg_list*2/3)/(.048*16 + opt_mass)

    #p.run_driver()

    #225 Wh/kg
    #150 Wh/kg


    # 128 Wh pack
    # n = 128/Wh

    # 12 Wh -> 18 kJ
    # 16 Wh -> 24 kJ

    # Wh  = kJ*2/3
    # tot_mass = .048*n + mass

    # Wh/kg_pack = 128/(.048*16*12/kJ + mass*(12/kJ))
    # Wh/kg_cell = 225*((kJ*2/3)/12)

    # print('\n \n')
    # print('-------------------------------------------------------')
    # print('Temperature (deg C). . . . . . . . .', p.get_val('temp2_data', units='degK'))  
    # print('Mass (kg). . . . . . . . . . . . . .', p.get_val('mass', units='kg'))  
    # print('Total X-Sectional Area (mm**2). . . ', p.get_val('solid_area', units='mm**2'))
    # print('Area, with holes (mm**2). . . . . . ', p.get_val('area', units='mm**2'))
    # print('Material Volume (mm**3). . . . . . .', p.get_val('volume', units='mm**3'))
    print('energy . . . . . . . . . . . . . . .', p.get_val('energy'))
    print('extra. . . . . . . . . . . . . . . .', p.get_val('extra'))
    print('ratio. . . . . . . . . . . . . . . .', p.get_val('ratio'))
    # print('sanity check. . . . . . . . . . . . ', p.get_val('energy')/p.get_val('extra'))
    # # print('resistance . . . . . . . . . . . . .', p.get_val('resistance'))
    # print('temp ratio . . . . . . . . . . . . .', p.get_val('temp_ratio'))
    # print('-------------------------------------------------------')
    # print('\n \n')


    #  # save to CSV
    df=pd.DataFrame({
                    'mass':opt_mass,
                    'temp':opt_temp,
                    'side':opt_side,
                    'energy':nrg_list,
                    'spacing':opt_spacing,
                    'ratio':opt_ratio,
                    't_ratio':opt_t_ratio,
                    'cell_dens': cell_dens,
                    'pack_dens': pack_dens,
                    'success': opt_success
                    })

    ofile = 'grid_48_opt.csv'
    df.to_csv(ofile,index=False)

    opt_plots([ofile],x)

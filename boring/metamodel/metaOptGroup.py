""""
Meta Model optimization group

Author: Dustin Hall, Jeff Chin
"""

import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from boring.metamodel.metaGroup import MetaCaseGroup


class MetaOptimize(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='meta_model_group',
                           subsys = MetaCaseGroup(num_nodes=nn),
                           promotes_inputs=[ 'cell_rad', 'energy', 'extra', 'ratio', 'resistance', 'length','al_density','n'],
                           promotes_outputs=['solid_area', 'cell_cutout_area', 'air_cutout_area', 'area', 'volume', 'mass', 'temp2_data', 'temp3_data'])

        self.add_subsystem(name='temp_ratio',
                                subsys=om.ExecComp('temp_ratio = temp2_data/temp3_data', units='degK'),
                                promotes_inputs=['temp2_data','temp3_data'],
                                promotes_outputs=['temp_ratio'])

if __name__ == "__main__":
    p = om.Problem()
    model = p.model
    nn = 1


    p.model.add_subsystem(name='meta_optimize',
                          subsys=MetaOptimize(num_nodes=nn),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
    
    p.driver = om.pyOptSparseDriver()
    # p.driver.options['optimizer'] = 'ALPSO'
    # p.driver.options['print_results'] = True
    # # p.driver.pyopt_solution.optInform["value"]
    # p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['Major optimality tolerance'] = 1e-8
    p.driver.opt_settings['Linesearch tolerance'] = 0.001
    p.set_solver_print(level=2)
    # p.driver = om.SimpleGADriver()
    p.model.add_design_var('extra', lower=1.0, upper=1.5)
    p.model.add_design_var('ratio', lower=1.0, upper=2) 
    # p.model.add_design_var('resistance', lower=0.003, upper=0.009)
    p.model.add_objective('mass', ref=1)
    p.model.add_constraint('temp2_data', upper=450)
    p.model.add_constraint('temp_ratio', upper=1.29)
    # p.model.add_constraint('solid_area', lower=6000)
    p.setup()
    p.set_val('cell_rad', 9, units='mm')
    p.set_val('resistance', 0.004)
    # p.set_val('extra', 1.5)
    # p.set_val('ratio', 1.2)
    p.set_val('energy',16., units='kJ')
    p.set_val('length', 65.0, units='mm')
    p.set_val('al_density', 2.7e-6, units='kg/mm**3')
    p.set_val('n',4)

    x = 30
    opt_mass = np.zeros(x)
    opt_ratio = np.zeros(x)
    opt_spacing = np.zeros(x)
    opt_t_ratio = np.zeros(x)
    density = np.zeros(x)
    nrg_list = np.linspace(16.,24.,x)

    for i in range(x):
        p.set_val('energy',nrg_list[i])
        p.run_driver()
        print(p.driver.pyopt_solution.optInform)  # print out the convergence success (SNOPT only)
        opt_mass[i]=p.get_val('mass')
        opt_ratio[i]=p.get_val('ratio')
        opt_spacing[i]=p.get_val('extra')
        opt_t_ratio[i]=p.get_val('temp_ratio')
        # density[i] = 128/(.048*16*12/kJ + mass*(12/kJ))

    cell_dens = 225*((nrg_list*2/3)/12)
    pack_dens = (16*nrg_list*2/3)/(.048*16+ opt_mass)
    fig, ax = plt.subplots(4,2)
    ax[0,1].plot(cell_dens,pack_dens)
    ax[0,1].plot(cell_dens,cell_dens*0.412+80)
    ax[0,1].set_ylabel('Wh/kg pack')
    ax[0,1].set_xlabel('Wh/kg cell')
    fig.delaxes(ax[1][1])
    fig.delaxes(ax[2][1])
    fig.delaxes(ax[3][1])
    ax[0,0].plot(nrg_list,opt_mass)
    ax[0,0].set_ylabel('optimal mass (kg)')
    ax[1,0].plot(nrg_list,opt_spacing)
    ax[1,0].set_ylabel('optimal spacing')
    ax[2,0].plot(nrg_list,opt_ratio)
    ax[2,0].set_ylabel('optimal hole ratio')
    ax[3,0].plot(nrg_list,opt_t_ratio)
    ax[3,0].set_ylabel('temp ratio')
    ax[3,0].set_xlabel('energy (kJ)')
    plt.show()
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

    print('\n \n')
    print('-------------------------------------------------------')
    print('Temperature (deg C). . . . . . . . .', p.get_val('temp2_data', units='degC'))  
    print('Mass (kg). . . . . . . . . . . . . .', p.get_val('mass', units='kg'))  
    print('Total X-Sectional Area (mm**2). . . ', p.get_val('solid_area', units='mm**2'))
    print('Area, with holes (mm**2). . . . . . ', p.get_val('area', units='mm**2'))
    print('Material Volume (mm**3). . . . . . .', p.get_val('volume', units='mm**3'))
    print('energy . . . . . . . . . . . . . . .', p.get_val('energy'))
    print('extra. . . . . . . . . . . . . . . .', p.get_val('extra'))
    print('ratio. . . . . . . . . . . . . . . .', p.get_val('ratio'))
    print('resistance . . . . . . . . . . . . .', p.get_val('resistance'))
    print('temp ratio . . . . . . . . . . . . .', p.get_val('temp_ratio'))
    print('-------------------------------------------------------')
    print('\n \n')


    #  # save to CSV
    df=pd.DataFrame({
                    'mass':opt_mass,
                    'energy':nrg_list,
                    'spacing':opt_spacing,
                    'ratio':opt_ratio,
                    't_ratio':opt_t_ratio,
                    'cell_dens': cell_dens,
                    'pack_dens': pack_dens
                    })
    df.to_csv('opt_out.csv',index=False)
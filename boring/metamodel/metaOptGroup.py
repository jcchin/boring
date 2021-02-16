""""
Meta Model optimization group

Author: Dustin Hall, Jeff Chin
"""

import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import more_itertools as mit

from boring.metamodel.sizing_component import MetaPackSizeComp
from boring.metamodel.training_data import MetaTempGroup

class MetaOptimize(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='size',
                           subsys = MetaPackSizeComp(num_nodes=nn),
                           promotes_inputs=['cell_rad', 'extra', 'ratio', 'length','al_density','n'],
                           promotes_outputs=['diagonal','solid_area', 'cell_cutout_area', 'air_cutout_area', 'area', 'volume', 'mass'])

        self.add_subsystem(name='temp',
                           subsys=MetaTempGroup(num_nodes=nn),
                           promotes_inputs=['energy','extra', 'ratio','resistance'],
                           promotes_outputs=['temp2_data','temp3_data'])

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
    p.driver.opt_settings['Major optimality tolerance'] = 1e-6
    # p.driver.opt_settings['Major step limit'] = .1
    # p.driver.opt_settings['Linesearch tolerance'] = 0.99
    p.set_solver_print(level=2)
    # p.driver = om.SimpleGADriver()
    p.model.add_design_var('extra', lower=1.0, upper=2.0)
    p.model.add_design_var('ratio', lower=1.0, upper=2) 
    # p.model.add_design_var('resistance', lower=0.003, upper=0.009)
    # p.model.add_objective('mass', ref=1e1)
    p.model.add_objective('diagonal', ref=1)
    p.model.add_constraint('temp2_data', upper=460)
    p.model.add_constraint('temp_ratio', upper=1.25)
    #p.model.add_constraint('diagonal', upper=145)
    # p.model.add_constraint('solid_area', lower=6000)
    p.setup()
    p.set_val('cell_rad', 9, units='mm')
    p.set_val('resistance', 0.004)
    p.set_val('extra', 1.0)
    p.set_val('ratio', 1.0)
    p.set_val('energy',16., units='kJ')
    p.set_val('length', 65.0, units='mm')
    p.set_val('al_density', 2.7e-6, units='kg/mm**3')
    p.set_val('n',4)

    x = 30
    nrg_list = np.linspace(16.,32.,x)
    # print('foobar', nrg_list[7])
    # exit()

    # nrg_list = np.array([28.444444444444443,])
    # x = len(nrg_list)

    opt_mass = np.zeros(x)
    opt_ratio = np.zeros(x)
    opt_spacing = np.zeros(x)
    opt_t_ratio = np.zeros(x)
    opt_res = np.zeros(x)
    density = np.zeros(x)
    opt_success = np.zeros(x)
    opt_diag = np.zeros(x)
    opt_temp = np.zeros(x)


    for i, nrg in enumerate(nrg_list):
        p.set_val('energy',nrg_list[i])

        p.set_val('extra', 1.0)
        p.set_val('ratio', 1.0)

        p.run_driver()
        opt_success[i]=p.driver.pyopt_solution.optInform['value']
        print(p.driver.pyopt_solution.optInform)  # print out the convergence success (SNOPT only)
        opt_mass[i]=p.get_val('mass')
        opt_ratio[i]=p.get_val('ratio')
        opt_spacing[i]=p.get_val('extra')
        opt_t_ratio[i]=p.get_val('temp_ratio')
        opt_res[i]=p.get_val('resistance')
        opt_diag[i]=p.get_val('diagonal')
        opt_temp[i]=p.get_val('temp2_data')
        # density[i] = 128/(.048*16*12/kJ + mass*(12/kJ))

    # p['energy'] = 28.444444444444443
    # p['extra'] = 1.45 # 1.11
    # p['ratio'] = 1.9 # 1.1465579

    # p.run_driver()
    # print(p.driver.pyopt_solution.optInform)  # print out the convergence success (SNOPT only)


    # print('\n \n')
    # print('-------------------------------------------------------')
    # print('Temperature (deg C). . . . . . . . .', p.get_val('temp2_data', units='degK'))  
    # print('Mass (kg). . . . . . . . . . . . . .', p.get_val('mass', units='kg'))  
    # print('Total X-Sectional Area (mm**2). . . ', p.get_val('solid_area', units='mm**2'))
    # print('Area, with holes (mm**2). . . . . . ', p.get_val('area', units='mm**2'))
    # print('Material Volume (mm**3). . . . . . .', p.get_val('volume', units='mm**3'))
    # print('energy . . . . . . . . . . . . . . .', p.get_val('energy'))
    # print('extra. . . . . . . . . . . . . . . .', p.get_val('extra'))
    # print('ratio. . . . . . . . . . . . . . . .', p.get_val('ratio'))
    # print('resistance . . . . . . . . . . . . .', p.get_val('resistance'))
    # print('temp ratio . . . . . . . . . . . . .', p.get_val('temp_ratio'))
    # print('-------------------------------------------------------')
    # print('\n \n')
    # exit()

    cell_dens = 225*((nrg_list*2/3)/12)
    pack_dens = (16*nrg_list*2/3)/(.048*16+ opt_mass)

    indices = [i for i in range(len(opt_success)) if opt_success[i] > 3]  # record indices where 32,41
    def find_ranges(iterable):  #     Yield range of consecutive numbers
        for group in mit.consecutive_groups(iterable):
            group = list(group)
            if len(group) == 1:
                yield group[0]
            else:
                yield group[0], group[-1]

    zones = list(find_ranges(indices))
    print(zones)

    fig, ax = plt.subplots(3,3)

    ax[0,2].plot(nrg_list,opt_temp)
    ax[0,2].set_ylabel('Neighbor Temp')
    ax[1,2].plot(nrg_list,opt_diag)
    ax[1,2].set_ylabel('diagonal')
    ax[2,2].plot(cell_dens,pack_dens)
    ax[2,2].plot(cell_dens,cell_dens*0.412+80)
    ax[2,2].set_ylabel('Wh/kg pack')
    ax[2,2].set_xlabel('Wh/kg cell')
    #fig.delaxes(ax[1][2])
    #fig.delaxes(ax[2][2])
    ax[0,0].plot(nrg_list,opt_mass)
    ax[0,0].set_ylabel('optimal mass (kg)')
    ax[1,0].plot(nrg_list,opt_spacing)
    ax[1,0].set_ylabel('optimal spacing')
    ax[2,0].plot(nrg_list,opt_ratio)
    ax[2,0].set_ylabel('optimal hole ratio')
    ax[2,0].set_xlabel('energy (kJ)')
    ax[0,1].plot(nrg_list,opt_res)
    ax[0,1].set_ylabel('opt resistance')
    ax[1,1].plot(nrg_list,opt_t_ratio)
    ax[1,1].set_ylabel('temp ratio')
    ax[2,1].plot(nrg_list,opt_success)
    for zone in zones:  # plot vertical red zones on all subplots (except the last plot)
        if type(zone) is tuple: #it's a range
            (minz,maxz) = zone
            [ax2.axvspan(nrg_list[minz], nrg_list[maxz], alpha=0.5, color='red') for ax2 in ax.flatten()[:-1]]
        else: # it's just one point
            [ax2.axvspan(nrg_list[zone-1], nrg_list[zone+1], alpha=0.5, color='red') for ax2 in ax.flatten()[:-1]]
    ax[2,1].set_ylabel('opt_success')
    ax[2,1].set_xlabel('energy (kJ)')
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
    print('Temperature (deg C). . . . . . . . .', p.get_val('temp2_data', units='degK'))  
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
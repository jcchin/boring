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


import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from boring.metamodel.sizing_component import MetaPackSizeComp
from boring.metamodel.training_data import MetaTempGroup
from boring.util.opt_plots import opt_plots

from boring.src.sizing.mass.pcm_mass import pcmMass
from boring.src.sizing.mass.insulation_mass import insulationMass
from boring.src.sizing.mass.flat_hp_mass import flatHPmass
from boring.src.sizing.mass.round_hp_mass import roundHPmass



class Build_Pack(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_cells', types=int, default=3)
        self.options.declare('pcm_bool', types=bool, default=False)
        self.options.declare('geom', values=['round', 'flat'], default='flat')


    def setup(self):
        nn = self.options['num_nodes']
        geom = self.options['geom']
        num_cells = self.options['num_cells']
        pcm_bool = self.options['pcm_bool']


        # Size the pack components
        self.add_subsystem(name='sizePack',
                           subsys = sizePack(num_nodes=nn),
                           promotes_inputs=['batt_l', 'batt_h', 'L_flux','num_cells'],
                           promotes_outputs=['A_pad','length_hp','A_w','A_wk'])
        self.add_subsystem(name='calcThickness',
                           subsys= calcThickness(),
                           promotes_inputs=['temp_limit'],
                           promotes_outputs=['ins_thickness'])

        # Calculate total mass
        self.add_subsystem(name='sizePCM',
                           subsys = pcmMass(num_nodes=nn),
                           promotes_inputs=['t_pad', 'A_pad', 'porosity'],
                           promotes_outputs=['mass_pcm'])
        self.add_subsystem(name='sizeInsulation',
                           subsys = insulationMass(num_nodes=nn),
                           promotes_inputs=['num_cells','batt_l','L_flux','batt_h',],
                           promotes_outputs=['ins_mass'])
        if geom == 'flat'
            self.add_subsystem(name='sizeHP',
                               subsys = flatHPMass(num_nodes=nn),
                               promotes_inputs=['length_hp', 'width_hp', 'wick_t', 'wall_t','wick_porosity'],
                               promotes_outputs=['mass_hp'])
        if geom == 'round'
            self.add_subsystem(name='sizeHP',
                               subsys = roundHPMass(num_nodes=nn),
                               promotes_inputs=['D_od_hp', 'wick_t', 'wall_t','wick_porosity'],
                               promotes_outputs=['mass_hp'])
        adder = om.AddSubtractComp()
        adder.add_equation('mass_total',
                           input_names=['mass_pcm','mass_hp','ins_mass','mass_battery'],
                           vec_size=nn, length=2, units='kg')
        self.add_subsystem(name='mass_comp',
                           subsys = adder,
                           promotes_inputs=['mass_pcm','mass_hp','ins_mass','mass_battery']
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
    if geom == 'flat'
        p.model.add_design_var('hp_height', lower=0.003, upper=0.009)
        p.model.add_design_var('hp_width', lower=0.003, upper=0.009)
    if geom == 'round'
        p.model.add_design_var('hp_D_od', lower=0.003, upper=0.009)
    p.model.add_design_var('hp_t_w', lower=0.003, upper=0.009)
    p.model.add_design_var('hp_t_wk', lower=0.003, upper=0.009)
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
        p.set_val('ratio', 0.4)  # .75

        p.run_driver()
        p.run_model()
        # p.check_totals(method='cs')
        #opt_success[i]=p.driver.pyopt_solution.optInform['stopCriteria']
        opt_success[i]=p.driver.pyopt_solution.optInform['value']
        print(p.driver.pyopt_solution.optInform)  # print out the convergence success (SNOPT only)
        opt_mass[i]=p.get_val('mass')
        #opt_ratio[i]=p.get_val('ratio')
        opt_spacing[i]=p.get_val('extra')
        opt_t_ratio[i]=p.get_val('temp_ratio')
        # opt_res[i]=p.get_val('resistance')
        #opt_side[i]=p.get_val('side')
        opt_temp[i]=p.get_val('temp2_data')
        # density[i] = 128/(.048*16*12/kJ + mass*(12/kJ))

    
    cell_dens = 225*((nrg_list*2/3)/12)
    pack_dens = (16*nrg_list*2/3)/(.048*16 + opt_mass)

    #p.run_driver()

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
    df.to_csv('opt_out.csv',index=False)

    # opt_plots(['opt_out.csv'],x)



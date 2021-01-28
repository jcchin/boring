""""
Meta Model optimization group

Author: Dustin Hall
"""

import openmdao.api as om

from boring.metamodel.metaGroup import MetaCaseGroup


class MetaOptimize(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='meta_model_group',
                           subsys = MetaCaseGroup(num_nodes=nn),
                           promotes_inputs=[ 'cell_rad', 'extra', 'ratio', 'length','al_density','n'],
                           promotes_outputs=['solid_area', 'cell_cutout_area', 'air_cutout_area', 'area', 'volume', 'mass', 'temp_data'])



if __name__ == "__main__":
    p = om.Problem()
    model = p.model
    nn = 1


    p.model.add_subsystem(name='meta_optimize',
                          subsys=MetaOptimize(num_nodes=nn),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
    
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['Major feasibility tolerance'] = 1e-6

    p.model.add_design_var('extra', lower=1.1, upper=1.5)
    p.model.add_design_var('ratio', lower=0.5, upper=2)

    p.model.add_objective('mass', ref=1)

    p.model.add_constraint('temp_data', upper=550)
    # p.model.add_constraint('solid_area', lower=6000)


    p.setup()

    p.set_val('cell_rad', 9, units='mm')
    # p.set_val('extra', 1.5)
    # p.set_val('ratio', 1.0)
    p.set_val('length', 65.0, units='mm')
    p.set_val('al_density', 2.7e-6, units='kg/mm**3')
    p.set_val('n',4)

    p.run_driver()



    print('\n \n')
    print('-------------------------------------------------------')
    print('Temperature (deg C). . . . . . . . .', p.get_val('temp_data', units='C'))  
    print('Mass (kg). . . . . . . . . . . . . .', p.get_val('mass', units='kg'))  
    print('Total X-Sectional Area (mm**2). . . ', p.get_val('solid_area', units='mm**2'))
    print('Area, with holes (mm**2). . . . . . ', p.get_val('area', units='mm**2'))
    print('Material Volume (mm**3). . . . . . .', p.get_val('volume', units='mm**3'))
    print('extra. . . . . . . . . . . . . . . .', p.get_val('extra'))
    print('ratio. . . . . . . . . . . . . . . .', p.get_val('ratio'))
    print('-------------------------------------------------------')
    print('\n \n')
""""
Top level group for the pack meta model

Author: Dustin Hall
"""

import openmdao.api as om

from boring.metamodel.sizing_component import MetaPackSizeComp
from boring.metamodel.training_data import MetaTempGroup



class MetaCaseGroup(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='size',
                           subsys = MetaPackSizeComp(num_nodes=nn),
                           promotes_inputs=['cell_rad','extra', 'ratio', 'length','al_density','n'],
                           promotes_outputs=['solid_area', 'cell_cutout_area', 'air_cutout_area', 'area', 'volume', 'mass'])

        self.add_subsystem(name='temp',
                           subsys=MetaTempGroup(num_nodes=nn),
                           promotes_inputs=['ratio','extra'],
                           promotes_outputs=['temp_data'])


if __name__ == "__main__":
    p = om.Problem()
    model = p.model
    nn = 1

    p.model.add_subsystem(name='meta_pack',
                          subsys=MetaCaseGroup(num_nodes=nn),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

    p.setup()


    p.set_val('cell_rad', 9, units='mm')
    p.set_val('extra', 1.0)
    p.set_val('ratio', 2.0)
    p.set_val('length', 65.0, units='mm')
    p.set_val('al_density', 2.7e-6, units='kg/mm**3')
    p.set_val('n',4)

    p.run_model()

    print('\n \n')
    print('-----------Area of a dollar bill is 10,338 mm**2----------------')
    print('Solid Area (mm**2). . . . . . . . . . . . . . ', p.get_val('solid_area', units='mm**2'))
    print('Cell Cut out Area (mm**2) . . . . . . . . . . ', p.get_val('cell_cutout_area', units='mm**2'))
    print('Air Void Cut out Area (mm**2) . . . . . . . . ', p.get_val('air_cutout_area', units='mm**2'))
    print('Cross sectional area (voids removed) (mm**2). ', p.get_val('area', units='mm**2'))
    print('Volume (mm**3). . . . . . . . . . . . . . . . ', p.get_val('volume', units='mm**3'))
    print('Mass (kg). .. . . . . . . . . . . . . . . . . ', p.get_val('mass', units='kg'))  
    print('Temperature (deg C). . . . . . . . . . . . . .', p.get_val('temp_data', units='C'))  
    # print('Cell Area . . . . . . . . . . . . . . ', p.get_val('cell_cutout_area'))
    # print('Air Area. . . . . . . . . . . . . . . ', p.get_val('air_cutout_area'))
    print('-------------------------------------------------------------')
    print('\n \n')
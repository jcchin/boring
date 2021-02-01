'''
Training data recieved from the COMSOL model

Author: Dustin Hall, Jeff Chin
'''

import numpy as np
import openmdao.api as om

## File containing the COMSOL temperature data from xlsx2npy.py script
t_data = np.load('../util/xlsx2np/outputs/test3.npy')


# For loop to pick off the highest temp
t_max_intermediate = []
for i in t_data:
    t_max_intermediate.append(np.max(i, axis=1, keepdims=True))

# This is passed into the MetaTempGroup class
t_max_data = np.array(t_max_intermediate)


class MetaTempGroup(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        temp_interp = om.MetaModelStructuredComp(method='scipy_cubic')

        time_data = np.linspace(0,0,1)
        extra_data = np.linspace(1.,1.5,6)
        ratio_data = np.linspace(0.5,2.0,7)
         
        temp_interp.add_input('extra', val=1, training_data=extra_data, units='mm')
        temp_interp.add_input('ratio', val=1, training_data=ratio_data)
        temp_interp.add_input('time', val=0, training_data= time_data, units='s')
        

        temp_interp.add_output('temp_data', val=300*np.ones(nn), training_data=t_max_data, units='C')



        self.add_subsystem('meta_temp_data', temp_interp,
            promotes_inputs=['extra', 'ratio'],
            promotes_outputs=['temp_data'])
'''
Training data recieved from the COMSOL model
OpenMDAO viewer for the metamodel data. Run the command:
<openmdao view_mm training_data.py -m temp.meta_temp2_data>
Then open a browser and navigate to 
<localhost:5007>
Author: Dustin Hall, Jeff Chin
'''
import numpy as np
import openmdao.api as om
import pickle
## File containing the COMSOL temperature data from xlsx2npy.py script
#t_data = np.load('../util/xlsx2np/outputs/test3.npy')
# t_data2 = np.load('cell2_16_32kj_exra.npy')
# t_data3 = np.load('cell3_16_32kj_exra.npy')
# t_data2 = np.load('cell2_2_24.npy')  # Al grid
# t_data3 = np.load('cell3_2_24.npy')  # Al grid
# t_data2 = np.load('cell2_hny.npy')  # Al honeycomb
# t_data3 = np.load('cell3_hny.npy')  # Al honeycomb
# t_data2 = np.load('cell2_pcm.npy')  # PCM grid
# t_data3 = np.load('cell3_pcm.npy')  # PCM grid
# t_data2 = np.load('cell2_hny_hole.npy')  # Al honeycomb with holes
# t_data3 = np.load('cell3_hny_hole.npy')  # Al honeycomb with holes
# t_data2 = np.load('cell2_48kj_h10.npy')  # Al grid 8-48kj
# t_data3 = np.load('cell3_48kj_h10.npy')  # Al grid 8-48kj
# t_data2 = np.load('cell2_hny_hole_h100.npy')  # Al hny with holes htc 100 w/m2k
# t_data3 = np.load('cell3_hny_hole_h100.npy')  # Al hny with holes htc 100 w/m2k
t_data2 = np.load('cell2_hny_hole_h250.npy')  # Al hny with holes htc 250 w/m2k
t_data3 = np.load('cell3_hny_hole_h250.npy')  # Al hny with holes htc 250 w/m2k
# t_data2 = np.load('cell2_hny_hole_h250m.npy')  # Al hny with holes htc 250 w/m2k middle trigger cell
# t_data3 = np.load('cell3_hny_hole_h250m.npy')  # Al hny with holes htc 250 w/m2k middle trigger cell

#m_data = np.squeeze(np.load('mass_hny_hole.npy'))
#m_data = np.squeeze(np.load('mass_hny_hole_h100.npy'))
m_data = np.squeeze(np.load('mass_hny_hole_h250.npy'))
# print(m_data.shape)
# print(t_data2)

t_data2[t_data2 == 0] = 2400.  # replace broken cases with a (doubly) high value (for ratio calc =2 for invalid cases)
t_data3[t_data3 == 0] = 1200.  # replace broken cases with a high value
# bp = pickle.load( open( "cell2.pickle", "rb" ) )
# print(bp)
# # For loop to pick off the highest temp
# t_max_intermediate = []
# for i in t_data:
#     t_max_intermediate.append(np.max(i, axis=1, keepdims=True))
# # This is passed into the MetaTempGroup class
# t_max_data = np.array(t_max_intermediate)
class MetaTempGroup(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('config', types=str, default='')
    def setup(self):
        nn = self.options['num_nodes']
        config = self.options['config']

        temp2_interp = om.MetaModelStructuredComp(method='lagrange2', extrapolate=True)
        temp3_interp = om.MetaModelStructuredComp(method='lagrange2', extrapolate=True)

        # #grid, honeycomb, pcm
        # energy_bp = np.linspace(16,32,5) 
        # extra_bp = np.linspace(1.,2.,6) 
        # ratio_bp = np.linspace(0.2,1.,5) 
        # # 48kj sweep
        # energy_bp = np.array([0.1, 8,16,24,32,40,48]) 
        # extra_bp = np.linspace(1.0,1.6,4)
        # ratio_bp = np.linspace(0.2,0.8,4) 
        # # cooling hny hole sweep 10
        # energy_bp = np.linspace(16.,32.,5)
        # extra_bp = np.linspace(1.0,1.5,6)
        # ratio_bp = np.linspace(0.1,0.9,5)
        # # cooling hny hole sweep 100
        # energy_bp = np.linspace(16.,32.,5)
        # extra_bp = np.linspace(1.1,1.5,5)
        # ratio_bp = np.linspace(0.1,0.9,5)
        # # cooling hny hole sweep 250
        energy_bp = np.linspace(16.,32.,5)
        extra_bp = np.linspace(1.01,1.41,5)
        ratio_bp = np.linspace(0.1,0.9,5)

        #res_bp = np.linspace(0.006, 0.006, 1)
        temp2_interp.add_input('energy', val=16, training_data=energy_bp, units='kJ')         
        temp2_interp.add_input('extra', val=1, training_data=extra_bp)
        temp2_interp.add_input('ratio', val=1, training_data=ratio_bp)  # <--
        #temp2_interp.add_input('resistance', val=0.006, training_data=res_bp, units='degK*mm**2/W')
        #temp2_interp.add_input('time', val=0, training_data= time_bp, units='s')
        temp3_interp.add_input('energy', val=16, training_data=energy_bp, units='kJ')
        temp3_interp.add_input('extra', val=1, training_data=extra_bp)
        temp3_interp.add_input('ratio', val=1, training_data=ratio_bp)  # <--
        #temp3_interp.add_input('resistance', val=0.006, training_data=res_bp, units='degK*mm**2/W')
        #temp3_interp.add_input('time', val=0, training_data= time_bp, units='s')

        temp2_interp.add_output('temp2_data', val=300*np.ones(nn), training_data=t_data2, units='degK')
        temp3_interp.add_output('temp3_data', val=300*np.ones(nn), training_data=t_data3, units='degK') 
        
        inpts = ['energy','extra', 'ratio']
        self.add_subsystem('meta_temp2_data', temp2_interp,
                            promotes_inputs=inpts,
                            promotes_outputs=['temp2_data'])
        self.add_subsystem('meta_temp3_data', temp3_interp,                         
                            promotes_inputs=inpts,
                            promotes_outputs=['temp3_data'])                            

        # if mass is computed by COMSOL
        mass_interp = om.MetaModelStructuredComp(method='lagrange2', extrapolate=True)
        mass_interp.add_input('energy', val=16, training_data=energy_bp, units='kJ')         
        mass_interp.add_input('extra', val=1, training_data=extra_bp)
        mass_interp.add_input('ratio', val=1, training_data=ratio_bp)  # <--
        mass_interp.add_output('mass', val=0.5*np.ones(nn), training_data=m_data, units='kg')
        self.add_subsystem('meta_mass_data', mass_interp,
                            promotes_inputs=['energy','extra','ratio'],
                            promotes_outputs=['mass'])

if __name__ == '__main__':
    prob = om.Problem()
    nn=1
    mm = prob.model.add_subsystem(name='temp',
                           subsys=MetaTempGroup(num_nodes=nn),
                           promotes_inputs=['energy','extra', 'ratio'],
                           promotes_outputs=['temp2_data','temp3_data'])     
    # mm = prob.model.add_subsystem(name='temp',
    #                    subsys=MetaTempGroup(num_nodes=nn, config='honeycomb'),
    #                    promotes_inputs=['energy','extra'],
    #                    promotes_outputs=['temp2_data','temp3_data','mass'])     
    prob.setup()
    prob.final_setup()
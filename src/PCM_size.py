"""
This file sizes the battery pack structure
"""

import openmdao.api as om


#                        Insulation 
#         --------- --------- --------- --------- 
#        | Battery | Battery | Battery | Battery |
#         --------- --------- --------- ---------
#        |   PCM   |   PCM   |   PCM   |   PCM   |
#         --------- --------- --------- ---------
#    ... <        Oscillating Heat Pipe          > ...
#        ---------------------------------------
#        |   PCM   |   PCM   |   PCM   |   PCM   |
#        --------- --------- --------- ---------
#        | Battery | Battery | Battery | Battery |
#         --------- --------- --------- ---------
#                       Insulation 

class packSize(om.ExplicitComponent):
    """ Sizing the Overall Size of the Battery Pack"""
    def initialize(self):
        self.options.declare('num_nodes', types=int) #argument for eventual dymos transient model

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('L', 2, units='m', desc='max length')
        self.add_input('W', 0.4, units='m', desc='max width')
        self.add_input('energy', 70000., units='W*h', desc='nominal total pack energy')
        self.add_input('cell_l', 0.102, units='m', desc='cell width (4" Amprius)')
        self.add_input('cell_w', 0.0571, units='m' , desc='cell length (2.0" Amprius)')
        self.add_input('cell_h', 0.00635, units='m' , desc='cell thickness (0.25" Amprius)')
        self.add_input('cell_s_w', 0.003, units='m', desc='cell spacing, width')
        self.add_input('cell_s_h', 0.001, units='m', desc='cell spacing, height')
        self.add_input('cell_s_l', 0.001, units='m', desc='cell spacing, height')
        self.add_input('v_n_c', 3.4, units='V', desc='nominal cell voltage')
        self.add_input('q_max', 7., units='A*h', desc='nominal cell amp-hr capacity')
        self.add_input('t_PCM', 0.006, units='m', desc='PCM pad thickness')
        self.add_input('t_OHP', 0.006, units='m', desc='OHP thickness')
        

        self.add_output('cell_area', units='m**2', desc='cell area')
        self.add_output('n_bps', desc='number of bars per stack')
        self.add_output('n_stacks', desc='number of stacks per battery shipset')
        self.add_output('n_cpb', desc='number of cells per bar')
        self.add_output('n_cells', desc='number of cells')
        self.add_output('H', 0.8, units='m', desc='height')



    def compute(self, i, o):

        o['n_cells'] = i['energy'] / (i['q_max'] * i['v_n_c'])
        o['cell_area'] = i['cell_w'] * i['cell_l']
        o['n_cpb'] = 2.*i['L']/(i['cell_w']+i['cell_s_w'])  # bars have cells side-by-side, across their width (like a candy bar)
        o['n_bps'] = i['W']/(i['cell_l']+i['cell_s_l']) # packs have stacks, each the width of a cell length
        o['n_stacks'] = o['n_cells']/(o['n_cpb']*o['n_bps']) # the number of stacks is driven by the max cells that can be fit into bars and stacks
        o['H'] = o['n_stacks']*(i['cell_h']*2.+i['cell_s_h']+i['t_PCM']*2.+i['t_OHP']) # stacks have vertically stacked bars, driven by cell, pad, and OHP height/thickness


    def compute_partials(self, inputs, J):
        pass #ToDo once calculations are complete


class pcmSize(om.ExplicitComponent):
    """ Sizing the Phase Change Material Pads"""
    def initialize(self):
        self.options.declare('num_nodes', types=int) #argument for eventual dymos transient model

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('frame_mass', 0.01, units='kg', desc='frame mass per cell')
        self.add_input('LH_PCM', 190., units='kJ/kg', desc='Latent Heat of the Phase change material')
        self.add_input('rho_PCM', 900., units='kg/m**3', desc='Density of the Phase change material')
        self.add_input('n_cells', 320, desc='number of cells')
        self.add_input('ext_cooling', 0, units='W', desc='external cooling')
        self.add_input('missionJ', 1.2, units='kJ', desc='Energy rejected in a normal mission per cell')
        self.add_input('frac_absorb', 1.0, desc='fraction of energy absorbed during runaway')
        self.add_input('runawayJ', 48.0, units='kJ', desc='Runaway heat of a 18650 battery')
        self.add_input('cell_area', 0.059*2*0.0571, units='m**2', desc='cell area')
        self.add_input('n_bps', 1, desc='number of bars per stack')
        self.add_input('n_cpb', 8, desc='number of cells per bar')
        self.add_input('n_stacks', 40, desc='number of stacks')

        self.add_output('t_PCM', units='mm', desc='PCM thickness')
        self.add_output('k_PCM', desc='PCM pad conductivity')
        self.add_output('PCM_bar_mass', units='kg', desc='Bar PCM mass')
        self.add_output('PCM_tot_mass', units='kg', desc='Total Pack PCM mass')


    def compute(self, inputs, outputs):

        n_bps = inputs['n_bps']
        n_cpb = inputs['n_cpb']
        n_stacks = inputs['n_stacks']
        n_cells = inputs['n_cells']
        rho_PCM = inputs['rho_PCM']
        runawayJ = inputs['runawayJ']
        missionJ = inputs['missionJ']
        ext_cooling = inputs['ext_cooling']
        LH = inputs['LH_PCM']
        frac_absorb = inputs['frac_absorb']
        cell_area = inputs['cell_area']


        outputs['PCM_bar_mass'] = (runawayJ*frac_absorb + n_cpb*missionJ - ext_cooling)/LH
        outputs['t_PCM'] = outputs['PCM_bar_mass']/(rho_PCM*n_cpb*cell_area)
        outputs['PCM_tot_mass'] = outputs['PCM_bar_mass']*n_bps*n_cpb


    def compute_partials(self, inputs, J):
        pass #ToDo once calculations are complete


class ohpSize(om.ExplicitComponent):
    """ Sizing the Heat Pipe """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # inputs
        self.add_input('frame_mass', 0.01, units='kg', desc='frame mass per cell')
        self.add_input('cell_area', 0.102*0.0571, units='m**2', desc='cell area')
        self.add_input('runawayJ', 48.0, units='kJ', desc='Runaway heat of a 18650 battery')
        self.add_input('dur', units='s', desc='runaway event duration')
        self.add_input('n_cells', 320, desc='number of cells')

        # outputs
        self.add_output('flux', units='W', desc='Heat Load through Heat Pipe')
        self.add_output('mass_OHP', units='kg', desc='Total Pack OHP mass')
        self.add_output('Areal_weight', units='kg/m**2', desc='Oscillating Heat Pipe Areal Weight')

    def compute(self, i, o):

        o['flux'] = i['runawayJ']/i['dur'] #heat flux during runaway
        o['Areal_weight'] = 0.3866*o['flux']+1.7442 # NH3   kg/m^2
        o['mass_OHP'] = o['Areal_weight']*i['cell_area']*i['n_cells']/2

    def compute_partials(self, inputs, J):
        pass #ToDo once calculations are complete


class packMass(om.ExplicitComponent):
    """sum all individual masses to estimate total mass and mass fractions"""

    def initialize(self):
        self.options.declare('num_nodes', types=int) #argument for eventual dymos transient model

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('PCM_tot_mass', units='kg', desc='total pack PCM mass') 
        self.add_input('mass_OHP', units='kg', desc='total pack OHP mass')
        self.add_input('frame_mass', units='kg', desc='frame mass per cell')
        self.add_input('n_cells', desc='number of cells')
        self.add_input('cell_mass', 0.0316*2, units='kg', desc='individual cell mass')
        self.add_input('ext_cool_mass', units='kg', desc='mass from external cooling')

        self.add_output('p_mass', desc='inactive pack mass')
        self.add_output('tot_mass', desc='total pack mass')
        self.add_output('mass_frac', desc='fraction of mass not fromt the battery cells')

    def compute(self,i,o):

        o['p_mass'] = i['PCM_tot_mass'] + i['mass_OHP'] + i['frame_mass']*i['n_cells'] + i['ext_cool_mass']
        o['tot_mass'] = o['p_mass'] + i['cell_mass']*i['n_cells']
        o['mass_frac'] = o['p_mass']/o['tot_mass']


class SizingGroup(om.Group): 
    
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='pack',
                           subsys=packSize(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name='pcm',
                           subsys=pcmSize(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name='ohp',
                           subsys=ohpSize(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name='mass',
                           subsys=packMass(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])


        self.set_input_defaults('frame_mass', 0.01, units='kg')



if __name__ == "__main__":


    p = om.Problem()
    model = p.model
    nn = 1

    record_file = 'geometry.sql'
    p.add_recorder(om.SqliteRecorder(record_file))
    p.recording_options['includes'] = ['*']
    p.recording_options['record_objectives'] = True
    p.recording_options['record_constraints'] = True
    p.recording_options['record_desvars'] = True
    p.recording_options['record_inputs'] = True


    model.add_subsystem('sizing', SizingGroup(num_nodes=nn), promotes=['*'])
    #model.add_design_var('sizing.L', lower=-1, upper=1)
    #model.add_objective('OD1.Eff')

    p.setup(force_alloc_complex=True)

    #p.set_val('DESIGN.rot_ir' , 60)

    p.run_model()
    p.record('final') #trigger problem record (or call run_driver if it's attached to the driver)

    # p.run_driver()
    # p.cleanup()

    model.list_inputs(prom_name=True)
    model.list_outputs(prom_name=True)
    # p.check_partials(compact_print=True, method='cs')

    print("num of cells: ", p['n_cells'])
    print("flux: ", p['flux'])
    print("PCM mass: ", p['PCM_tot_mass'])
    print("PCM thickness (mm): ", p['t_PCM'])
    print("OHP mass: ", p['mass_OHP'])
    print("packaging mass: ", p['p_mass'])
    print("total mass: ", p['tot_mass'])
    print("package mass fraction: ", p['mass_frac'])
    print("pack energy density: ", p['energy']/(p['tot_mass']))
    print("cell energy density: ", (p['q_max'] * p['v_n_c']) / (p['cell_mass']))
    print("pack energy (kWh): ", p['energy']/1000.)
    print("pack cost ($K): ", p['n_cells']*0.4)
    #print("overall pack dimensions: %.3f  ft x %.3f ft x %.3f ft" % (cell_l*s_h*n_cpb*3.28, stack_h*s_v*n_stacks*3.28, cell_w*s_h*2*3.28))

    #from terminal run:
    #openmdao n2 PCM_size.py



# # Starting with copper foam and Eicosane
# n_bps = 1 # arches
# n_cpb = 8 # cells per module
# n_stacks = 40 # stacks
# n_stacks_show = 3
# s_h = 1.1 # horizontal spacing
# s_v = 1.3 # vertical spacing
# s_h2 = ((s_h-1)/2 +1)

# n_cells = n_bps*n_cpb*n_stacks*4 # number of prismatic cells
# frame_mass = 10 # grams, frame mass per cell
# k_pcm = 3.06 # W/m*K, Conductivity of Eicosane with copper foam
# LH = 190 # J/g, Latent Heat of Eicosane
# rho_pcm = 900 #kg/m^3
# melt = 60 # degC, Metling Point of Eicosane
# missionJ = 1200 #J, Energy rejected in a normal mission
# runawayJ = 48000 # J, Runaway heat of a 18650 battery
# frac_absorb = 1.0 # fraction of energy absorbed during runaway
# dur = 45 # seconds, of runaway duration
# v_n_c = 3.4 #  nominal voltage
# q_max = 3.5*2. # A-h cells
# cell_mass = 31.6*2. #g, cell mass Dimensions: 57mm x 50mm x 6.35mm
# cell_w = 0.059*2. #m , (2.25")
# cell_l = 0.0571 #m , (2.0")
# cell_h = 0.00635 #m , (0.25")
# cell_area = cell_w*cell_l # Dimensions: 2.25" x 2" x 0.25"
# ext_cooling = 0 # W, external cooling
# ext_cool_mass = 0 # g, mass of external cooling
# #mass_OHP = 26 # g,  #2.5mm x 50mm x 250mm (< 270W) 0.2C/W
# #https://amecthermasol.co.uk/datasheets/MHP-2550A-250A.pdf


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

        self.add_input('L', 1, units='m', desc='max length')
        self.add_input('W', 40, units='m', desc='max width')
        self.add_input('H', 8, units='m', desc='max height')
        self.add_input('cell_w', 0.059*2, units='m', desc='cell width (2"*2 Amprius)')
        self.add_input('cell_l', 0.0571, units='m' , desc='cell length (2.0" Amprius)')
        self.add_input('cell_h', 0.00635, units='m' , desc='cell thickness (0.25" Amprius)')
        self.add_input('cell_s_w', 0.003, units='m', desc='cell spacing, width')
        self.add_input('cell_s_h', 0.001, units='m', desc='cell spacing, height')
        self.add_input('cell_s_l', 0.001, units='m', desc='cell spacing, height')
        self.add_input('v_n_c', 3.4, units='V', desc='nominal cell voltage')
        self.add_input('q_max', 3.5*2, units='A*h', desc='nominal cell amp-hr capacity')
        

        self.add_output('cell_area', units='m**2', desc='cell area')
        self.add_output('n_modules', desc='number of modules')
        self.add_output('n_stacks', desc='number of stacks')
        self.add_output('n_cpm', desc='number of cells per module')
        self.add_output('n_cells', desc='number of cells')
        self.add_output('energy', desc='nominal total pack energy')


    def compute(self, i, o):

        o['cell_area'] = i['cell_w'] * i['cell_l']
        o['n_cpm'] = i['L']/(i['cell_w']+i['cell_s_w'])  # modules have cells side-by-side, across their width
        o['n_modules'] = i['H']/(i['cell_h']+i['cell_s_h']) # stacks have vertically concatenated modules, driven by cell height
        o['n_stacks'] = i['W']/(i['cell_l']+i['cell_s_l']) # packs have stacks, each the width of a cell length
        o['n_cells'] = o['n_modules']*o['n_cpm']*o['n_stacks']
        o['energy'] = o['n_cells'] * i['q_max'] * i['v_n_c']

    def compute_partials(self, inputs, J):
        pass #ToDo once calculations are complete


class pcmSize(om.ExplicitComponent):
    """ Sizing the Phase Change Material Pads"""
    def initialize(self):
        self.options.declare('num_nodes', types=int) #argument for eventual dymos transient model

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('frame_mass', 10, units='g', desc='frame mass per cell')
        self.add_input('LH_PCM', 190., units='J/g', desc='Latent Heat of the Phase change material')
        self.add_input('rho_PCM', 190., units='kg/m**3', desc='Density of the Phase change material')
        self.add_input('n_cells', 320, desc='number of cells')
        self.add_input('ext_cooling', 0, units='W', desc='external cooling')
        self.add_input('missionJ', 1200, units='J', desc='Energy rejected in a normal mission')
        self.add_input('frac_absorb', 1.0, desc='fraction of energy absorbed during runaway')
        self.add_input('runawayJ', 48000, units='J', desc='Runaway heat of a 18650 battery')
        self.add_input('cell_area', 0.059*2*0.0571, units='m**2', desc='cell area')
        self.add_input('n_modules', 1, desc='number of modules')
        self.add_input('n_cpm', 8, desc='number of cells per module')
        self.add_input('n_stacks', 40, desc='number of stacks')

        self.add_output('t_PCM', desc='PCM thickness')
        self.add_output('k_PCM', desc='PCM pad conductivity')
        self.add_output('mass_PCM', units='kg', desc='Total Pack PCM mass')


    def compute(self, inputs, outputs):

        n_modules = inputs['n_modules']
        n_cpm = inputs['n_cpm']
        n_stacks = inputs['n_stacks']
        n_cells = inputs['n_cells']
        rho_PCM = inputs['rho_PCM']
        runawayJ = inputs['runawayJ']
        missionJ = inputs['missionJ']
        ext_cooling = inputs['ext_cooling']
        LH = inputs['LH_PCM']
        frac_absorb = inputs['frac_absorb']
        cell_area = inputs['cell_area']


        outputs['mass_PCM'] = n_modules*n_stacks*runawayJ*frac_absorb/LH + missionJ/LH - ext_cooling
        outputs['t_PCM'] = outputs['mass_PCM']/(rho_PCM*n_cells*cell_area)


    def compute_partials(self, inputs, J):
        pass #ToDo once calculations are complete


class ohpSize(om.ExplicitComponent):
    """ Sizing the Heat Pipe """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # inputs
        self.add_input('frame_mass', 10, units='g', desc='frame mass per cell')
        self.add_input('cell_area', 0.059*2*0.0571, units='m**2', desc='cell area')
        self.add_input('runawayJ', 48000, units='J', desc='Runaway heat of a 18650 battery')
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

        self.add_input('mass_PCM', units='kg', desc='total pack PCM mass') 
        self.add_input('mass_OHP', units='kg', desc='total pack OHP mass')
        self.add_input('frame_mass', units='kg', desc='frame mass per cell')
        self.add_input('n_cells', desc='number of cells')
        self.add_input('cell_mass', units='kg', desc='individual cell mass')
        self.add_input('ext_cool_mass', units='kg', desc='mass from external cooling')

        self.add_output('p_mass', desc='inactive pack mass')
        self.add_output('tot_mass', desc='total pack mass')
        self.add_output('mass_frac', desc='fraction of mass not fromt the battery cells')

    def compute(self,i,o):

        o['p_mass'] = i['mass_PCM'] + i['mass_OHP'] + i['frame_mass']*i['n_cells'] + i['ext_cool_mass']
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

    model.add_subsystem('sizing', SizingGroup(num_nodes=nn))
    #model.add_design_var('sizing.L', lower=-1, upper=1)
    #model.add_objective('OD1.Eff')

    p.setup(force_alloc_complex=True)

    #p.set_val('DESIGN.rot_ir' , 60)

    p.run_model()

    # p.run_driver()
    # p.cleanup()

    model.list_outputs(prom_name=True)
    # p.check_partials(compact_print=True, method='cs')

    #from terminal run:
    #openmdao n2 PCM_size.py



# # Starting with copper foam and Eicosane
# n_modules = 1 # arches
# n_cpm = 8 # cells per module
# n_stacks = 40 # stacks
# n_stacks_show = 3
# s_h = 1.1 # horizontal spacing
# s_v = 1.3 # vertical spacing
# s_h2 = ((s_h-1)/2 +1)

# n_cells = n_modules*n_cpm*n_stacks*4 # number of prismatic cells
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

# mass_PCM = n_modules*n_stacks*runawayJ*frac_absorb/LH + missionJ/LH - ext_cooling
# t_PCM = mass_PCM/(rho_pcm*n_cells*cell_area)
# flux = runawayJ/dur #heat flux during runaway
# Areal_weight = 0.3866*flux+1.7442 # NH3   kg/m^2
# # Areal_weight = 0.4922*flux+1.5599 # Acetone
# # Areal_weight = 0.7805*flux+1.9131 # C4H10
# mass_OHP = Areal_weight*cell_area*n_cells/2

# p_mass = mass_PCM + mass_OHP + frame_mass*n_cells + ext_cool_mass
# tot_mass = p_mass + cell_mass*n_cells 

# mass_frac = p_mass/tot_mass
# energy = n_cells * q_max * v_n_c


# # Render in OpenSCAD using the OpenPySCAD library
# import openpyscad as ops

# t_PCM_m = t_PCM/1000

# cell = ops.Cube([cell_w, cell_l, cell_h]) # Amprius Cell
# pcm = ops.Cube([cell_w, cell_l, t_PCM_m]).color("Orange") # Phase Change Material
# ohp = ops.Cube([cell_w, cell_l*n_cpm*s_h, cell_h]).color("Gray") # Oscillating Heat Pipe Straight-away
# ohp_turn = ops.Cylinder(h=cell_h, r=cell_w*s_h2, _fn=100).color("Gray") # OHP End Turn
# ohp_turn_d = ops.Cube([cell_w*2*s_h,cell_w*s_h,cell_h+0.02]) # Make it a semi-circle by subtracting a rectangle, plus a little height to avoid green
# pack = ops.Union()
# module = ops.Union()
# d = ops.Difference()
# d2 = ops.Difference()
# insulation = ops.Cube([cell_w*2,cell_l*s_h*n_cpm*1.1,cell_h*s_v]).color("Blue")

# stack_h = cell_h*2 + t_PCM_m*2

# for b in range(n_cpm):
#     # Cell Array
#     module.append(cell.translate([0, cell_l*s_h*b, 0])) # first row cell
#     module.append(cell.translate([0, cell_l*s_h*b, stack_h])) # second column, first row
#     module.append(cell.translate([cell_w*s_h, cell_l*s_h*b, 0])) # second row cell
#     module.append(cell.translate([cell_w*s_h, cell_l*s_h*b, stack_h])) # second column, second row cell
#     # PCM Array
#     module.append(pcm.translate([0, cell_l*s_h*b, cell_h])) # first row PCM
#     module.append(pcm.translate([0, cell_l*s_h*b, stack_h-t_PCM_m])) # second column, first row
#     module.append(pcm.translate([cell_w*s_h, cell_l*s_h*b, cell_h])) # second row cell
#     module.append(pcm.translate([cell_w*s_h, cell_l*s_h*b, stack_h-t_PCM_m])) # second column, second row cell

# # OHP
# module.append(ohp.translate([0,0,cell_h+t_PCM_m]))
# module.append(ohp.translate([cell_w*s_h,0,cell_h+t_PCM_m]))
# d.append(ohp_turn.translate([cell_w*s_h2,0,stack_h/2]))
# d.append(ohp_turn_d.translate([0,0,stack_h/2-0.01]))
# module.append(d)

# # Insulation
# d2.append(insulation.translate([0,-0.03,0.017]))
# for b in range(n_cpm): #subtract cells
#     d2.append(cell.translate([-0.005, cell_l*s_h*b, stack_h*1.3])) # first row cell
#     d2.append(cell.translate([-0.005+cell_w*s_h, cell_l*s_h*b, stack_h*1.3])) # second column, first row

# for s in range(n_stacks_show):
#     # Stack Array
#     for m in range(n_modules):
#         # Module Array
#         pack.append(module.translate([cell_w*s_h*2*m, 0, stack_h*s_v*s]))
# pack.append(d2.translate([-0.4,0,0]))
# pack.write("PCM.scad")
# d2.write("Insulation.scad")

# print("num of cells: ", n_cells)
# print("flux: ", flux)
# print("PCM mass: ", mass_PCM)
# print("PCM thickness (mm): ", t_PCM)
# print("OHP mass: ", mass_OHP)
# print("packaging mass: ", p_mass)
# print("total mass: ", tot_mass)
# print("package mass fraction: ", mass_frac)
# print("pack energy density: ", energy/(tot_mass/1000.))
# print("cell energy density: ", (q_max * v_n_c) / (cell_mass/1000.))
# print("pack energy (kWh): ", energy/1000.)
# print("pack cost ($K): ", n_cells*0.4)
# print("overall pack dimensions: %.3f  ft x %.3f ft x %.3f ft" % (cell_l*s_h*n_cpm*3.28, stack_h*s_v*n_stacks*3.28, cell_w*s_h*2*3.28))


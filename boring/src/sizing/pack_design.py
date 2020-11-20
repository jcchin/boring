"""
This file sizes the battery pack structure

Author: Jeff Chin
"""

import openmdao.api as om
from openmdao.api import ArmijoGoldsteinLS, DirectSolver, NewtonSolver
from boring.src.sizing.mass.mass import packMass
from boring.src.sizing.heat_pipe import OHP
import numpy as np

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
                           subsys=OHP(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name='mass',
                           subsys=packMass(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])


        self.set_input_defaults('frame_mass', 0.01, units='kg')

        self.nonlinear_solver = NewtonSolver(maxiter=30, atol=1e-10, rtol=1e-100)
        self.nonlinear_solver.options['solve_subsystems'] = True
        self.nonlinear_solver.options['max_sub_solves'] = 500
        self.nonlinear_solver.linesearch = ArmijoGoldsteinLS()
        self.linear_solver = DirectSolver()
        self.nonlinear_solver.options['err_on_non_converge'] = True

class packSize(om.ExplicitComponent):
    """ Sizing the Overall Size of the Battery Pack"""
    def initialize(self):
        self.options.declare('num_nodes', types=int) #argument for eventual dymos transient model

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('L', 2*np.ones(nn), units='m', desc='max length')
        self.add_input('W', 0.4*np.ones(nn), units='m', desc='max width')
        self.add_input('energy', 70000.*np.ones(nn), units='W*h', desc='nominal total pack energy')
        self.add_input('cell_l', 0.102*np.ones(nn), units='m', desc='cell width (4" Amprius)')
        self.add_input('cell_w', 0.0571*np.ones(nn), units='m' , desc='cell length (2.0" Amprius)')
        self.add_input('cell_h', 0.00635*np.ones(nn), units='m' , desc='cell thickness (0.25" Amprius)')
        self.add_input('cell_s_w', 0.003*np.ones(nn), units='m', desc='cell spacing, width')
        self.add_input('cell_s_h', 0.001*np.ones(nn), units='m', desc='cell spacing, height')
        self.add_input('cell_s_l', 0.001*np.ones(nn), units='m', desc='cell spacing, height')
        self.add_input('v_n_c', 3.4*np.ones(nn), units='V', desc='nominal cell voltage')
        self.add_input('q_max', 7.*np.ones(nn), units='A*h', desc='nominal cell amp-hr capacity')
        self.add_input('t_PCM', 0.006*np.ones(nn), units='m', desc='PCM pad thickness')
        self.add_input('t_HP', 0.006*np.ones(nn), units='m', desc='OHP thickness')
        

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
        o['H'] = o['n_stacks']*(i['cell_h']*2.+i['cell_s_h']+i['t_PCM']*2.+i['t_HP']) # stacks have vertically stacked bars, driven by cell, pad, and OHP height/thickness


    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')


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


    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')




if __name__ == "__main__":

    pass
    
    #print("overall pack dimensions: %.3f  ft x %.3f ft x %.3f ft" % (cell_l*s_h*n_cpb*3.28, stack_h*s_v*n_stacks*3.28, cell_w*s_h*2*3.28))

    #from terminal run:
    #openmdao n2 PCM_size.py



# # Starting with copper foam and Eicosane
# #mass_OHP = 26 # g,  #2.5mm x 50mm x 250mm (< 270W) 0.2C/W
# #https://amecthermasol.co.uk/datasheets/MHP-2550A-250A.pdf


"""
Author: Dustin Hall

Assumptions:
    1) Pouch cells
    2) Aerogel as insulation
    3) insulation thickness surrounding cells in all direction is equal to LW:L_adiabatic
    4) Amprius large pouch dimensions
    5) layout of the pouch cells are as seen below:
         ________________
        |  _   _   _   _  |
        | |_| |_| |_| |_| |
        |_________________|
    
"""

import openmdao.api as om


class insulationMass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('num_cells', 4, desc='number of cells in system')
        self.add_input('num_stacks', 1, desc='number of parallel stacks in system')
        self.add_input('batt_l', .10599, units='m', desc='length of the battery')
        self.add_input('L_flux', 0.04902, units='m', desc='width of the active battery material (no lip)')
        self.add_input('batt_cutout_w', 0.050038, units='m', desc='width of the aerogel cutout for the cell')
        self.add_input('batt_h', 0.00635, units='m', desc='height (thickness) of the battery')
        self.add_input('ins_density', 100, units='kg/m**3', desc='density of the insulation material')
        self.add_input('LW:L_adiabatic', 0.002, units='m', desc='spacing between batteries = thickness of insulation')
        self.add_input('A_pad', 0.00259781, units='m**2', desc='area of the pcm pad')
        self.add_input('ins_pcm_layer_t',0.002, units='m', desc='thickness of the pcm insulation layer')
        self.add_input('LW:L_flux_flat', 0.025, units='m', desc='width of the heatpipe')
        self.add_input('XS:H_hp', 0.005, units='m', desc='height of the heatpipe')

        self.add_output('L_ins', 0.1, units='m', desc='length of the insulation plane')
        self.add_output('W_ins', 0.05, units='m', desc='width of the insulation plane')
        self.add_output('ins_cell_tray_area', 0.0250, units='m**2', desc='area of the cell tray insulation')
        self.add_output('ins_cell_tray_volume', 0.0250, units='m**3', desc='volume cell tray insulation')
        self.add_output('ins_cell_tray_mass', 0.0250, units='kg', desc='mass cell tray insulation')
        self.add_output('ins_pcm_layer_area', 0.0250, units='m**2', desc='area of the insulation layer parallel with the pcm')
        self.add_output('ins_pcm_layer_volume', 0.0250, units='m**3', desc='volume of the insulation layer parallel with the pcm')
        self.add_output('ins_pcm_layer_mass', 0.0250, units='kg', desc='mass of the insulation layer parallel with the pcm')
        self.add_output('ins_hp_layer_area', 0.0250, units='m**2', desc='area of the insulation layer parallel with the heatpipe')
        self.add_output('ins_hp_layer_volume', 0.0250, units='m**3', desc='volume of the insulation layer parallel with the heatpipe')
        self.add_output('ins_hp_layer_mass', 0.0250, units='kg', desc='mass of the insulation layer parallel with the heatpipe')
        self.add_output('ins_tot_A', 0.05, units='m**2', desc='total area of insulation')
        self.add_output('ins_tot_volume', 0.05, units='m**3', desc='total volume of insulation')
        self.add_output('ins_tot_mass', 1, units='kg', desc='total mass of insulation')
        

        self.declare_partials('*', '*', method='cs')
    
    def compute(self, inputs, outputs):
        num_cells = inputs['num_cells']
        num_stacks = inputs['num_stacks']
        batt_l = inputs['batt_l']
        L_flux = inputs['L_flux']
        batt_h = inputs['batt_h']
        ins_density = inputs['ins_density']
        L_adiabatic = inputs['LW:L_adiabatic']
        A_pad = inputs['A_pad']
        ins_pcm_layer_t = inputs['ins_pcm_layer_t']
        hp_w = inputs['LW:L_flux_flat']
        hp_h = inputs['XS:H_hp']
        batt_cutout_w = inputs['batt_cutout_w']

        outputs['L_ins'] = ((num_cells+1) * L_adiabatic) + (num_cells*batt_cutout_w)
        outputs['W_ins'] = 2*L_adiabatic + batt_l

        outputs['ins_cell_tray_area'] = outputs['L_ins'] * outputs['W_ins'] * num_stacks
        outputs['ins_cell_tray_volume'] = (outputs['ins_cell_tray_area'] * (batt_h + L_adiabatic) - (batt_h*batt_cutout_w*batt_l*num_cells)) * num_stacks
        outputs['ins_cell_tray_mass'] =outputs['ins_cell_tray_volume']  * ins_density * num_stacks

        outputs['ins_pcm_layer_area'] = (outputs['ins_cell_tray_area'] - (num_cells*A_pad)) * num_stacks
        outputs['ins_pcm_layer_volume'] = outputs['ins_pcm_layer_area'] * ins_pcm_layer_t * num_stacks
        outputs['ins_pcm_layer_mass'] = outputs['ins_pcm_layer_volume'] * ins_density * num_stacks

        outputs['ins_hp_layer_area'] = (outputs['ins_cell_tray_area'] - (hp_w  * outputs['L_ins'])) * num_stacks
        outputs['ins_hp_layer_volume'] = outputs['ins_hp_layer_area'] * (0.5*hp_h) * num_stacks
        outputs['ins_hp_layer_mass'] = outputs['ins_hp_layer_volume'] * ins_density * num_stacks

        outputs['ins_tot_A'] = (outputs['ins_cell_tray_area'] + outputs['ins_pcm_layer_area'] + outputs['ins_hp_layer_area']) * num_stacks
        outputs['ins_tot_volume'] = (outputs['ins_cell_tray_volume'] + outputs['ins_pcm_layer_volume'] + outputs['ins_hp_layer_volume']) * num_stacks
        outputs['ins_tot_mass'] = (outputs['ins_cell_tray_mass'] + outputs['ins_pcm_layer_mass'] + outputs['ins_hp_layer_mass']) * num_stacks




if __name__ == "__main__":
    from openmdao.api import Problem

    nn = 1
    prob = Problem()

    prob.model.add_subsystem('ins_mass', insulationMass(num_nodes=nn), promotes=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()


    print('ins_cell_tray_mass = ', prob.get_val('ins_mass.ins_cell_tray_mass'))
    print('ins_pcm_layer_mass = ', prob.get_val('ins_mass.ins_pcm_layer_mass'))
    print('ins_hp_layer_mass = ', prob.get_val('ins_mass.ins_hp_layer_mass'))
    print('Tot mass = ', prob.get_val('ins_mass.ins_tot_mass'))
    print('Tot volume = ', prob.get_val('ins_mass.ins_tot_volume'))


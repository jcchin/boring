"""
Author: Dustin Hall

Assumptions:
    1) Pouch cells
    2) Aerogel as insulation
    3) 2mm is min width of aerogel
    4) spacing between batteries is 2mm.
    5) Amprius large pouch dimensions
    6) layout of the battery design is 4 horizontal cells with insulation on backside, in between, on edges
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
        self.add_input('batt_l', 106.0, units='mm', desc='length of the battery')
        self.add_input('L_flux', 50.0, units='mm', desc='width of the battery')
        self.add_input('batt_h', 6.4, units='mm', desc='height (thickness) of the battery')
        self.add_input('ins_density', 1.6e-7, units='kg/mm**3', desc='density of the insulation material')
        self.add_input('ins_thickness', 2, units='mm', desc='height (thickness) of the insulation, equal to d')
        self.add_input('LW:L_adiabatic', 2, units='mm', desc='spacing between batteries in the horizontal')
        self.add_input('batt_end_sep', 2, units='mm', desc='spacing at the ends of the batteries in the vertical')
        self.add_input('stack_ins_t', 2, units='mm', desc='insulation thickness in the stack direction')
        self.add_input('A_pad', 5, units='mm**2', desc='area of the pcm pad')
        self.add_input('ins_pcm_layer_t', units='mm', desc='thickness of the pcm insulation layer')
        self.add_input('LW:L_flux', units='mm', desc='width of the heatpipe')

        self.add_output('ins_volume', 50, units='mm**3', desc='volume of the insulation')
        self.add_output('cell_tray_area', 250, units='mm**2', desc='area of the insulation on the back of the batts')
        self.add_output('A', 50, units='mm**2', desc='side area of the battery')
        self.add_output('ins_side_sep_area', 250, units='mm**2', desc='area of the ins between the batts')
        self.add_output('ins_end_sep_area', 250, units='mm**2', desc='area of the ins at the vertical ends of the batts')
        self.add_output('ins_mass', 0.5, units='kg', desc='total mass of the insulation')

        self.declare_partials('*', '*', method='cs')
    
    def compute(self, inputs, outputs):
        num_cells = inputs['num_cells']
        num_stacks = inputs['num_stacks']
        batt_l = inputs['batt_l']
        L_flux = inputs['L_flux']
        batt_h = inputs['batt_h']
        ins_density = inputs['ins_density']
        ins_thickness = inputs['ins_thickness']
        batt_side_sep = inputs['LW:L_adiabatic']
        batt_end_sep = inputs['batt_end_sep']
        stack_ins_t = inputs['stack_ins_t']
        A_pad = inputs['A_pad']
        ins_pcm_layer_t = inputs['ins_pcm_layer_t']
        hp_w = inputs['LW:L_flux']

        outputs['cell_tray_area'] = (num_cells*batt_l*L_flux) + (batt_side_sep*batt_l*(num_cells+1)) + ( batt_end_sep*(num_stacks+1) * ((num_cells*L_flux)+(batt_side_sep*(num_cells+1)))  )
        outputs['cell_tray_thickness'] = batt_h + stack_ins_t
        outputs['cell_tray_mass'] =((outputs['cell_tray_area']*outputs['cell_tray_thickness']) - (batt_h*L_flux*batt_l*num_cells)) * ins_density

        outputs['ins_pcm_layer_area'] = outputs['cell_tray_area'] - (num_cells*A_pad)
        outputs['ins_pcm_layer_volume'] = outputs['ins_pcm_layer_area'] * ins_pcm_layer_t
        outputs['ins_pcm_layer_mass'] = outputs['cell_tray_area'] * outputs['ins_pcm_layer_area'] * ins_density

        outputs['ins_hp_layer_area'] = outputs['cell_tray_area'] - hp_w
        outputs['ins_hp_layer_volume'] = outputs['ins_hp_layer_area'] * hp_t

        outputs['A'] = batt_l*batt_h
        outputs['ins_side_sep_area'] = outputs['A']*num_stacks*(num_cells+1)
        outputs['ins_end_sep_area'] = batt_h * ((num_cells*L_flux) + (batt_side_sep*(num_cells+1))) * (num_stacks+1)
        outputs['ins_volume'] = (outputs['cell_tray_area'] + outputs['ins_side_sep_area'] + outputs['ins_end_sep_area']) * ins_thickness
        outputs['ins_mass'] = outputs['ins_volume'] * ins_density




if __name__ == "__main__":
    from openmdao.api import Problem

    nn = 1
    prob = Problem()

    prob.model.add_subsystem('ins_mass', insulationMass(num_nodes=nn), promotes=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()

    print('cell_tray_area = ', prob.get_val('ins_mass.cell_tray_area'))
    print('ins_side_sep_area = ', prob.get_val('ins_mass.ins_side_sep_area'))
    print('ins_end_sep_area = ', prob.get_val('ins_mass.ins_end_sep_area'))
    print('ins_volume = ', prob.get_val('ins_mass.ins_volume'))
    print('ins_mass = ', prob.get_val('ins_mass.ins_mass'))


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
        self.add_input('num_rows', 1, desc='number of cells in system')
        self.add_input('batt_l', 106.0, units='mm', desc='length of the battery')
        self.add_input('batt_w', 50.0, units='mm', desc='width of the battery')
        self.add_input('batt_h', 6.4, units='mm', desc='height (thickness) of the battery')
        self.add_input('ins_density', 1.6e-7, units='kg/mm**3', desc='density of the insulation material')
        self.add_input('ins_thickness', 2, units='mm', desc='height (thickness) of the insulation, equal to d')
        self.add_input('batt_side_sep', 2, units='mm', desc='spacing between batteries in the horizontal')
        self.add_input('batt_end_sep', 2, units='mm', desc='spacing at the ends of the batteries in the vertical')

        self.add_output('ins_volume', 50, units='mm**3', desc='volume of the insulation')
        self.add_output('ins_backing_area', 250, units='mm**2', desc='area of the insulation on the back of the batts')
        self.add_output('ins_side_sep_area', 250, units='mm**2', desc='area of the ins between the batts')
        self.add_output('ins_end_sep_area', 250, units='mm**2', desc='area of the ins at the vertical ends of the batts')
        self.add_output('ins_mass', .5, units='kg', desc='total mass of the insulation')
    
    def compute(self, inputs, outputs):
        num_cells = inputs['num_cells']
        num_rows = inputs['num_rows']
        batt_l = inputs['batt_l']
        batt_w = inputs['batt_w']
        batt_h = inputs['batt_h']
        ins_density = inputs['ins_density']
        ins_thickness = inputs['ins_thickness']
        batt_side_sep = inputs['batt_side_sep']
        batt_end_sep = inputs['batt_end_sep']

        outputs['ins_backing_area'] = (num_cells*batt_l*batt_w) + (batt_side_sep*(num_cells+1)) + (batt_end_sep*(num_rows+1))
        outputs['ins_side_sep_area'] = batt_l*batt_h*num_rows*(num_cells+1)
        outputs['ins_end_sep_area'] = batt_h * ((num_cells*batt_w) + (batt_side_sep*(num_cells+1))) * (num_rows+1)
        outputs['ins_volume'] = (outputs['ins_backing_area'] + outputs['ins_side_sep_area'] + outputs['ins_end_sep_area']) * ins_thickness
        outputs['ins_mass'] = outputs['ins_volume'] * ins_density

def setup_partials(inputs, outputs, J):
    self.declare_partials('*', '*', method='cs')



if __name__ == "__main__":
    from openmdao.api import Problem

    nn = 1
    prob = Problem()

    prob.model.add_subsystem('ins_mass', insulationMass(num_nodes=nn), promotes=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()

    print('ins_backing_area = ', prob.get_val('ins_mass.ins_backing_area'))
    print('ins_side_sep_area = ', prob.get_val('ins_mass.ins_side_sep_area'))
    print('ins_end_sep_area = ', prob.get_val('ins_mass.ins_end_sep_area'))
    print('ins_volume = ', prob.get_val('ins_mass.ins_volume'))
    print('ins_mass = ', prob.get_val('ins_mass.ins_mass'))


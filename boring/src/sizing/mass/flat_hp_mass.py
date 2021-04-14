"""
Author: Dustin Hall

REFERENCES
1) Sintered Powder Heatpipe denstiy/porosity/thermal conductivity: https://arc.aiaa.org/doi/pdf/10.2514/3.50

"""
import openmdao.api as om
import numpy as np

class flatHPmass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('length_hp', 0.200 * np.ones(nn), units='m', desc='length of the hp')
        self.add_input('width_hp', 0.030 * np.ones(nn), units='m', desc='width of the hp, battery contact surface')
        self.add_input('height_hp', 0.020 * np.ones(nn), units='m', desc='height of the hp')
        self.add_input('wick_t', 0.003 * np.ones(nn), units='m', desc='thickness of the wick')
        self.add_input('wall_t', 0.003 * np.ones(nn), units='m', desc='wall thickness')
        self.add_input('wall_density', 2710 * np.ones(nn), units='kg/m**3', desc='density of the Aluminium wall material')
        self.add_input('wick_density', 8940. * np.ones(nn), units='kg/m**3', desc='density of Cu102 (see ref 1)')
        self.add_input('wick_porosity', 0.52 * np.ones(nn), desc="porosity of the wick (0.32, 0.52, 0.59: see ref 1), 0=solid wick")
        self.add_input('fluid_density', 997. * np.ones(nn), units='kg/m**3', desc='density of the working fluid')
        self.add_input('fluid_fill', 0.5 * np.ones(nn), desc='fill factor of the fluid')
        
        self.add_output('volume_wall', .005 * np.ones(nn), units='m**3', desc='volume of the wall')
        self.add_output('volume_wick', .005 * np.ones(nn), units='m**3', desc='volume of the wick')
        self.add_output('volume_fluid', .005 * np.ones(nn), units='m**3', desc='volume of the fluid')
        self.add_output('mass_flat_hp', .05 * np.ones(nn), units='kg', desc='mass of the flat hp')

    def setup_partials(self):
        nn=self.options['num_nodes']
        ar = np.arange(nn) 

        self.declare_partials('volume_wall', ['width_hp', 'height_hp', 'wall_t', 'length_hp'], rows=ar, cols=ar)
        self.declare_partials('volume_wick', ['width_hp', 'wall_t', 'height_hp', 'wick_t', 'wick_porosity', 'length_hp'], rows=ar, cols=ar)
        self.declare_partials('volume_fluid', ['width_hp', 'wall_t', 'wick_t', 'height_hp', 'length_hp', 'fluid_fill'], rows=ar, cols=ar)
        self.declare_partials('mass_flat_hp', ['wall_density', 'wick_density', 'fluid_density', 'width_hp', 'height_hp', 'wall_t', 'length_hp', 'wick_porosity', 'fluid_fill', 'wick_t'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        length_hp =     inputs['length_hp']
        width_hp =      inputs['width_hp']
        height_hp =     inputs['height_hp']
        wick_t =        inputs['wick_t']
        wall_t =        inputs['wall_t']
        wall_density =  inputs['wall_density']
        wick_density =  inputs['wick_density']
        wick_porosity = inputs['wick_porosity']
        fluid_density = inputs['fluid_density']
        fluid_fill = inputs['fluid_fill']

        outputs['volume_wall'] = ( (width_hp*height_hp) - ((width_hp-2*wall_t)*(height_hp-2*wall_t)) ) * (length_hp)
        outputs['volume_wick'] = ((((width_hp-2*wall_t)*(height_hp-2*wall_t)) - \
                                 ((width_hp-2*wall_t - 2*wick_t)*(height_hp-2*wall_t - 2*wick_t)))*(1-wick_porosity)) * (length_hp)
        outputs['volume_fluid'] = ((width_hp-2*wall_t - 2*wick_t)*(height_hp-2*wall_t - 2*wick_t)) * (length_hp) * (fluid_fill)
        outputs['mass_flat_hp'] = wall_density*outputs['volume_wall'] + wick_density*outputs['volume_wick']+ fluid_density*outputs['volume_fluid']


    def compute_partials(self, inputs, J):
        length_hp =     inputs['length_hp']
        cross_length_hp= inputs['cross_length_hp']
        width_hp =      inputs['width_hp']
        height_hp =     inputs['height_hp']
        wick_t =        inputs['wick_t']
        wall_t =        inputs['wall_t']
        wall_density =  inputs['wall_density']
        wick_density =  inputs['wick_density']
        wick_porosity = inputs['wick_porosity']
        fluid_density = inputs['fluid_density']
        fluid_fill =    inputs['fluid_fill']

        volume_wall = ( (width_hp*height_hp) - ((width_hp-2*wall_t)*(height_hp-2*wall_t)) ) * (length_hp)
        volume_wick = ((((width_hp-2*wall_t)*(height_hp-2*wall_t)) - ((width_hp-2*wall_t - 2*wick_t)*(height_hp-2*wall_t - 2*wick_t)))*(1-wick_porosity)) * (length_hp)
        volume_fluid = ((width_hp-2*wall_t - 2*wick_t)*(height_hp-2*wall_t - 2*wick_t)) * (length_hp) * (fluid_fill)
        mass_flat_hp = wall_density*volume_wall + wick_density*volume_wick+ fluid_density*volume_fluid

        alpha = (width_hp-2*wall_t - 2*wick_t)*(height_hp-2*wall_t - 2*wick_t)*length_hp
        d_alpha__d_width_hp = (height_hp-2*wall_t - 2*wick_t)*length_hp
        d_alpha__d_height_hp = (width_hp-2*wall_t - 2*wick_t)*length_hp
        d_alpha__d_wall_t = (-2*width_hp + 8*wall_t + 8*wick_t - 2*height_hp) * length_hp
        d_alpha__d_wick_t = (-2*width_hp + 8*wall_t + 8*wick_t - 2*height_hp) * length_hp
        d_alpha__d_length_hp = (width_hp-2*wall_t - 2*wick_t)*(height_hp-2*wall_t - 2*wick_t)

        d_volume_wall__d_width_hp = 2*length_hp*wall_t
        d_volume_wall__d_height_hp = 2*length_hp*wall_t
        d_volume_wall__d_wall_t = -length_hp*(-2*width_hp - 2*height_hp + 8*wall_t )
        d_volume_wall__d_length_hp = ( (width_hp*height_hp) - ((width_hp-2*wall_t)*(height_hp-2*wall_t)) )

        d_volume_wick__d_width_hp = (length_hp*(height_hp-2*wall_t) - d_alpha__d_width_hp) * (1-wick_porosity)
        d_volume_wick__d_wall_t = (length_hp*(-2*width_hp - 2*height_hp + 8*wall_t) - d_alpha__d_wall_t) * (1-wick_porosity)
        d_volume_wick__d_height_hp = (length_hp*(width_hp-2*wall_t) - d_alpha__d_height_hp) * (1-wick_porosity)
        d_volume_wick__d_wick_t = -d_alpha__d_wick_t * (1-wick_porosity)
        d_volume_wick__d_wick_porosity = -(((width_hp-2*wall_t)*(height_hp-2*wall_t)) - ((width_hp-2*wall_t - 2*wick_t)*(height_hp-2*wall_t - 2*wick_t))) * (length_hp)
        d_volume_wick__d_length_hp = ((((width_hp-2*wall_t)*(height_hp-2*wall_t)) - ((width_hp-2*wall_t - 2*wick_t)*(height_hp-2*wall_t - 2*wick_t)))*(1-wick_porosity))

        d_volume_fluid__d_width_hp = d_alpha__d_width_hp * (fluid_fill)
        d_volume_fluid__d_wall_t = d_alpha__d_wall_t * (fluid_fill)
        d_volume_fluid__d_wick_t = d_alpha__d_wick_t * (fluid_fill)
        d_volume_fluid__d_height_hp = d_alpha__d_height_hp * (fluid_fill)
        d_volume_fluid__d_length_hp = d_alpha__d_length_hp * (fluid_fill)
        d_volume_fluid__d_fluid_fill = ((width_hp-2*wall_t - 2*wick_t)*(height_hp-2*wall_t - 2*wick_t)) * (length_hp)

        d_mass__d_wall_density = volume_wall
        d_mass__d_wick_density = volume_wick
        d_mass__d_fluid_density = volume_fluid
        d_mass__d_width_hp = wall_density*d_volume_wall__d_width_hp + wick_density*d_volume_wick__d_width_hp + fluid_density*d_volume_fluid__d_width_hp
        d_mass__d_height_hp = wall_density*d_volume_wall__d_height_hp + wick_density*d_volume_wick__d_height_hp + fluid_density*d_volume_fluid__d_height_hp
        d_mass__d_wall_t = wall_density*d_volume_wall__d_wall_t + wick_density*d_volume_wick__d_wall_t + fluid_density*d_volume_fluid__d_wall_t
        d_mass__d_length_hp  = wall_density*d_volume_wall__d_length_hp + wick_density*d_volume_wick__d_length_hp + fluid_density*d_volume_fluid__d_length_hp
        d_mass__d_wick_porosity  =  wick_density * d_volume_wick__d_wick_porosity
        d_mass__d_fluid_fill  = fluid_density*d_volume_fluid__d_fluid_fill
        d_mass__d_wick_t = wick_density* d_volume_wick__d_wick_t + fluid_density * d_volume_fluid__d_wick_t

        J['volume_wall', 'width_hp'] = d_volume_wall__d_width_hp
        J['volume_wall', 'height_hp'] = d_volume_wall__d_height_hp
        J['volume_wall', 'wall_t'] = d_volume_wall__d_wall_t
        J['volume_wall', 'length_hp'] = d_volume_wall__d_length_hp

        J['volume_wick', 'width_hp'] = d_volume_wick__d_width_hp
        J['volume_wick', 'wall_t'] = d_volume_wick__d_wall_t
        J['volume_wick', 'height_hp'] = d_volume_wick__d_height_hp
        J['volume_wick', 'wick_t'] = d_volume_wick__d_wick_t
        J['volume_wick', 'wick_porosity'] = d_volume_wick__d_wick_porosity
        J['volume_wick', 'length_hp'] = d_volume_wick__d_length_hp

        J['volume_fluid', 'width_hp' ] = d_volume_fluid__d_width_hp
        J['volume_fluid', 'wall_t' ] = d_volume_fluid__d_wall_t
        J['volume_fluid', 'wick_t' ] = d_volume_fluid__d_wick_t
        J['volume_fluid', 'height_hp' ] = d_volume_fluid__d_height_hp
        J['volume_fluid', 'length_hp' ] = d_volume_fluid__d_length_hp
        J['volume_fluid', 'fluid_fill' ] = d_volume_fluid__d_fluid_fill

        J['mass_flat_hp', 'wall_density'] = d_mass__d_wall_density
        J['mass_flat_hp', 'wick_density'] = d_mass__d_wick_density
        J['mass_flat_hp', 'fluid_density'] = d_mass__d_fluid_density
        J['mass_flat_hp', 'width_hp'] = d_mass__d_width_hp
        J['mass_flat_hp', 'height_hp'] = d_mass__d_height_hp
        J['mass_flat_hp', 'wall_t'] = d_mass__d_wall_t
        J['mass_flat_hp', 'length_hp'] = d_mass__d_length_hp
        J['mass_flat_hp', 'wick_porosity'] = d_mass__d_wick_porosity
        J['mass_flat_hp', 'fluid_fill'] = d_mass__d_fluid_fill
        J['mass_flat_hp', 'wick_t'] = d_mass__d_wick_t



if __name__ == "__main__":
    prob = om.Problem(model=om.Group())  
    nn=1  

    prob.model.add_subsystem('comp1', flatHPmass(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)

    print('Volume of wall: ', prob.get_val('volume_wall'))
    print('Volume of wick: ', prob.get_val('volume_wick'))
    print('Volume of fluid: ', prob.get_val('volume_fluid'))
    print('Mass of flat hp: ', prob.get_val('mass_flat_hp'))
import numpy as np
import dymos as dm
import openmdao.api as om
# Boring Imports
# Geometry Imports
# from boring.src.sizing.geometry.pcm_geom import pcmGeom
from boring.src.sizing.geometry.insulation_geom import get_cond_phase
from boring.src.sizing.geometry.hp_geom import HPgeom
# Mass Imports
from boring.src.sizing.mass.pcm_mass import pcmMass
from boring.src.sizing.mass.insulation_mass import insulationMass
from boring.src.sizing.mass.flat_hp_mass import flatHPmass
from boring.src.sizing.mass.round_hp_mass import roundHPmass

class StaticSizing(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_cells', types=int, default=3)
        self.options.declare('pcm_bool', types=bool, default=False)
        self.options.declare('geom', values=['round', 'flat'], default='flat')


    def setup(self):
        nn = self.options['num_nodes']
        geom = self.options['geom']
        num_cells = self.options['num_cells']
        pcm_bool = self.options['pcm_bool']


        # Set inputs/outputs based on geometry
        if geom == 'round':
            inpts = ['LW:L_flux', 'LW:L_adiabatic', 'XS:t_w', 'XS:t_wk', 'XS:D_v']
            outpts = ['XS:D_od','XS:r_i', 'LW:A_flux', 'LW:A_inter']
        elif geom == 'flat':
            inpts = ['LW:L_flux', 'LW:L_adiabatic', 'XS:t_w', 'XS:t_wk']
            outpts = ['LW:A_flux', 'LW:A_inter', 'XS:W_hp']
        
        # Size the pack components
        self.add_subsystem(name = 'size',
                           subsys = HPgeom(num_nodes=nn, geom=geom),
                           promotes_inputs=inpts,
                           promotes_outputs=outpts) 

        # Calculate total mass
        self.add_subsystem(name='massPCM',
                           subsys = pcmMass(num_nodes=nn),
                           promotes_inputs=['t_pad', 'A_pad', 'porosity'],
                           promotes_outputs=['mass_pcm'])
        self.add_subsystem(name='massInsulation',
                           subsys = insulationMass(num_nodes=nn),
                           promotes_inputs=['num_cells','batt_l','L_flux','batt_h','ins_thickness'],
                           promotes_outputs=['ins_mass'])
        if geom == 'flat':
            self.add_subsystem(name='massHP',
                               subsys = flatHPmass(num_nodes=nn),
                               promotes_inputs=['length_hp', 'width_hp', 'wick_t', 'wall_t','wick_porosity'],
                               promotes_outputs=['mass_hp'])
        if geom == 'round':
            self.add_subsystem(name='massHP',
                               subsys = roundHPmass(num_nodes=nn),
                               promotes_inputs=['D_od_hp', 'wick_t', 'wall_t','wick_porosity'],
                               promotes_outputs=['mass_hp'])
        adder = om.AddSubtractComp()
        adder.add_equation('mass_total',
                           input_names=['mass_pcm','mass_hp','ins_mass','mass_battery'],
                           vec_size=nn, units='kg')
        self.add_subsystem(name='mass_comp',
                           subsys = adder,
                           promotes_inputs=['mass_pcm','mass_hp','ins_mass','mass_battery'],
                           promotes_outputs=['mass_total'])


if __name__ == '__main__':
    p = om.Problem()
    model = p.model
    nn = 1
    p.driver = om.ScipyOptimizeDriver()
    p.driver = om.pyOptSparseDriver(optimizer='SLSQP')

    p.driver.declare_coloring()

    traj = dm.Trajectory()
    phase = get_cond_phase()
    traj.add_phase('phase0', phase)
    p.model.add_subsystem(name='traj', subsys=traj,
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

    phase.add_objective('d', loc='final', ref=1)

    p.model.linear_solver = om.DirectSolver()

    p.model.add_subsystem('size', subsys=StaticSizing(num_nodes=1))

    #p.model.connect('traj.phase0.parameters:d','size.ins_thickness')
    p.setup(force_alloc_complex=True)


    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = 45
    p['traj.phase0.states:T'] = phase.interpolate(ys=[293.15, 333.15], nodes='state_input')
    p['traj.phase0.parameters:d'] = 0.001


    p.set_val('size.ins_thickness', 2.5) # mm
    p.set_val('size.t_pad', 0.0016) # m 
    p.set_val('size.A_pad', 0.0015) # m^2
    p.set_val('size.porosity', 0.96)
    p.set_val('size.wick_t', 0.0005) # m 
    p.set_val('size.wall_t', 0.0005) # m 
    p.set_val('size.length_hp', 0.051*4) # m 
    p.set_val('size.width_hp', 0.02) # m 
    p.set_val('size.massHP.fluid_fill', 0.25) # https://www.sciencedirect.com/science/article/abs/pii/S1359431120338151
    p.set_val('size.massHP.wick_density', 2710) # kg/m^3 
    p.set_val('size.wick_porosity', 0.52)
    p.set_val('size.mass_battery',0.0316) # kg

    p.run_model()

    num_cells = 4

    print("ins frac: ",p.get_val('size.ins_mass')/(p.get_val('size.mass_battery')*num_cells))
    print("hp frac: ",p.get_val('size.mass_hp')/(p.get_val('size.mass_battery')*num_cells*2))
    print("pcm frac: ",p.get_val('size.mass_pcm')/p.get_val('size.mass_battery'))
    print("total mass: ", p.get_val('size.mass_total'))

import unittest

import numpy as np
import dymos as dm
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
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

class SizeStuff(om.Group):
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


        # Size the pack components
        if geom == 'round':
            inpts = ['LW:L_flux', 'LW:L_adiabatic', 'XS:t_w', 'XS:t_wk', 'XS:D_v']
            outpts = ['XS:D_od','XS:r_i', 'LW:A_flux', 'LW:A_inter']
        elif geom == 'flat':
            inpts = ['LW:L_flux', 'LW:L_adiabatic', 'XS:t_w', 'XS:t_wk']
            outpts = ['LW:A_flux', 'LW:A_inter', 'XS:W_hp']
        
        self.add_subsystem(name = 'size',
                           subsys = HPgeom(num_nodes=nn, geom=geom),
                           promotes_inputs=inpts,
                           promotes_outputs=outpts) 
        
        # self.add_subsystem(name='sizeHP',
        #                    subsys = HPgeom(num_nodes=nn, geom=geom),
        #                    promotes_inputs=['LW:L_flux', 'LW:L_adiabatic', 'XS:t_w', 'XS:t_wk', 'XS:D_v'],
        #                    promotes_outputs=['XS:D_od','XS:r_i', 'LW:A_flux', 'LW:A_inter'])
        
        # self.add_subsystem(name='sizeInsulation',
        #                    subsys= calcThickness(),
        #                    promotes_inputs=['temp_limit'],
        #                    promotes_outputs=['ins_thickness'])



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

    p.model.add_subsystem('size', subsys=SizeStuff(num_nodes=1))

    #p.model.connect('traj.phase0.parameters:d','size.ins_thickness')
    p.setup(force_alloc_complex=True)


    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = 45
    p['traj.phase0.states:T'] = phase.interpolate(ys=[293.15, 333.15], nodes='state_input')
    p['traj.phase0.parameters:d'] = 0.001

    p.set_val('size.mass_battery',0.0316)

    p.run_model()

    num_cells = 4

    print("ins frac: ",p.get_val('size.ins_mass')/p.get_val('size.mass_battery')/num_cells)
    print("hp frac: ",p.get_val('size.mass_hp')/p.get_val('size.mass_battery')/num_cells)
    print("pcm frac: ",p.get_val('size.mass_pcm')/p.get_val('size.mass_battery')/num_cells)
    print("total mass: ", p.get_val('size.mass_total'))

class TestSizing(unittest.TestCase):
    """ Check general sizing groups for insulation and heatpipe, similar to the first two
    subsystems of build_pack.py"""

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('size', subsys=SizeStuff(num_nodes=1))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_tot_mass(self):  # calculation regression test
    # Cells weigh 31.6g

        print(self.prob.get_val('size.mass_total'))


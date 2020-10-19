from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om
from thermo.chemical import Chemical


class FluidPropertiesComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)


    def setup(self):
        nn=self.options['num_nodes']

        self.add_input('Q_hp', 1,  desc='')
        self.add_input('A_cond', 1,  desc='')
        self.add_input('h_c', 1200,  desc='')
        self.add_input('T_coolant', 293,  desc='')

        self.add_output('R_g', 1, desc='')
        self.add_output('P_v', 1, desc='')
        self.add_output('T_cond', 1, desc='')
        self.add_output('T_hp', 1, desc='')
        self.add_output('rho_v', 1, desc='')
        self.add_output('mu_v', 1, desc='')
        self.add_output('h_fg', 1,  desc='')
        self.add_output('hp_fluid_T_hp', 1, desc='')
        self.add_output('hp_fluid_T_hp__P_v', 1, desc='')


    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')


    def compute(self, inputs, outputs):
        hp_fluid = Chemical('water')

        Q_hp = inputs['Q_hp']
        A_cond = inputs['A_cond']
        h_c = inputs['h_c']
        T_coolant = inputs['T_coolant']

        outputs['T_cond'] = Q_hp/(A_cond*h_c)+T_coolant
        outputs['T_hp'] = outputs['T_cond']
        outputs['hp_fluid_T_hp'] = hp_fluid.calculate(outputs['T_hp'])

        outputs['P_v'] = hp_fluid.Psat
        outputs['hp_fluid_T_hp__P_v'] = hp_fluid.calculate(outputs['T_hp'],outputs['P_v'])


        outputs['h_fg'] = hp_fluid.Hvap
        outputs['mu_v'] = hp_fluid.mug
        outputs['rho_v'] = hp_fluid.rhog
        
        
        outputs['R_g'] = outputs['P_v']/(outputs['T_hp']*outputs['rho_v'])

    # def compute_partials(self, inputs, J):
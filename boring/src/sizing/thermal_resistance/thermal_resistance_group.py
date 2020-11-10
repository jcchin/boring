"""
Author: Dustin Hall
"""

from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om

from boring.src.sizing.thermal_resistance.axial_thermal_resistance import AxialThermalResistance
from boring.src.sizing.thermal_resistance.vapor_thermal_resistance import VaporThermalResistance
from boring.src.sizing.thermal_resistance.radial_thermal_resistance import RadialThermalResistance

# from total_resistance import TotalThermalResistance


class ThermalResistanceGroup(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='axial',
                           subsys=AxialThermalResistance(num_nodes=nn),
                           promotes_inputs=['epsilon', 'k_w', 'k_l', 'L_adiabatic', 'A_w', 'A_wk'],
                           promotes_outputs=['k_wk', 'R_aw', 'R_awk'])


        self.add_subsystem(name='vapor',
                           subsys=VaporThermalResistance(num_nodes=nn),
                           promotes_inputs=['D_v', 'R_g', 'mu_v', 'T_hp', 'h_fg', 'P_v', 'rho_v', 'L_eff'],
                           promotes_outputs=['r_h', 'R_v'])

        self.add_subsystem(name='radial',
                           subsys=RadialThermalResistance(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])


        # self.set_input_defaults('L_evap', 6, units='m')
        # self.set_input_defaults('L_cond', 5, units='m')
        # self.set_input_defaults('D_v', 0.5, units='m')
        # self.set_input_defaults('D_od', 2, units='m')
        # self.set_input_defaults('t_w', 0.01, units='m')
        # self.set_input_defaults('L_adiabatic', 0.01, units='m')
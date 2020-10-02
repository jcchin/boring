"""
Battery Performance Modeling on Maxwell X-57
Jeffrey C. Chin, Sydney L. Schnulo, Thomas B. Miller,
Kevin Prokopius, and Justin Gray
http://openmdao.org/pubs/chin_battery_performance_x57_2019.pdf

Thevenin voltage equation based on paper
"Evaluation of Lithium Ion Battery Equivalent Circuit Models
for State of Charge Estimation by an Experimental Approach"
Hongwen H, Rui Xiong, Jinxin Fan

Equivalent Circuit component values derived from
"High Fidelity Electrical Model with Thermal Dependence for
 Characterization and Simulation of High Power Lithium Battery Cells"
Tarun Huria, Massimo Ceraolo, Javier Gazzarri, Robyn Jackey
"""

from openmdao.api import Group
from boring.src.model.cell_comp import CellComp
from boring.src.model.reg_thevenin_interp_group import RegTheveninInterpGroup

import numpy as np


class BatteryGroup(Group):
    """Assembly to connect subcomponents of the Thevenin Battery Equivalent
    Circuit Model From Interpolated Performance Maps
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('cell_type', types=str, default='AMPRIUS') #'18650','AMPRIUS'
        self.options.declare('BATTERY_TF', types=bool)

    def setup(self):
        n = self.options['num_nodes']
        cell_type=self.options['cell_type']

        self.add_subsystem(name='interp_group',
                           subsys=RegTheveninInterpGroup(num_nodes=n, cell_type=cell_type),  
                           promotes_outputs=['U_oc', 'C_Th', 'R_Th', 'R_0'])

        self.add_subsystem(name='cell',
                           subsys=CellComp(num_nodes=n),
                           promotes_inputs=['SOC', 'U_oc', 'C_Th', 'R_Th', 'R_0'],
                           promotes_outputs=['pack_eta'])

        #self.connect('des_vars.Q_max','cell.Q_max')
        #promote inputs to match XDSM markdown spec
        self.promotes('interp_group', inputs=[('T_batt', 'T_{batt}')])
        self.promotes('interp_group', inputs=['SOC'])
        self.promotes('cell', inputs=[('U_Th', 'V_{thev}'), 
                                     ('I_pack', 'I_{batt}'),
                                     ('n_series','n_{series}'), 
                                     ('n_parallel', 'n_{parallel}'),
                                     ('Q_max','Q_{max}')])

        #promote outputs to match XDSM markdown spec
        self.promotes('cell', outputs=[('U_pack', 'V_{batt,actual}'),
                                       ('dXdt:U_Th', 'dXdt:V_{thev}'),
                                       ('dXdt:SOC', 'dXdt:SOC'),
                                       ('Q_pack', 'Q_{batt}')])

        self.set_input_defaults('SOC', val='0.5', units=None)


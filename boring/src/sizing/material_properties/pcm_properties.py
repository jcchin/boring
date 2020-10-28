from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om


class PCM_props(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int) # parallel execution


    def setup(self):
        nn=self.options['num_nodes']

        # conductive foam properties
        self.add_input('porosity', 1.,  desc='percentage porosity, 1 = completely void, 0 = solid')
        self.add_input('k_foam', 1., desc='thermal conductivity of the foam')
        self.add_input('rho_foam', 1., desc='intrinsic density of the foam material (unrelated to porosity)')
        self.add_input('t_pad', 1., desc='PCM pad thickness')
        # phase change material properties


        self.add_output('R_g', 1, units='J/kg/K', desc='gas constant of the vapor')



    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')


    def compute(self, inputs, outputs):
        
        #add calculations here


    # def compute_partials(self, inputs, J):

        # add partial derivatives here
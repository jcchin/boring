"""
Construct thermal network, then solve for flux and equivalent resistance

Assume all flux connection directions are pointed down and right

Author: Jeff Chin
"""

import openmdao.api as om

from boring.src.sizing.material_properties.fluid_properties import FluidPropertiesComp
from boring.src.sizing.thermal_resistance.radial_thermal_resistance import RadialThermalResistance
from boring.src.sizing.thermal_resistance.axial_thermal_resistance import AxialThermalResistance
from boring.src.sizing.thermal_resistance.vapor_thermal_resistance import VaporThermalResistance


class Resistor(om.ExplicitComponent):
    """Computes flux across a resistor using Ohm's law."""

    # def initialize(self):
    #     self.options.declare('R', default=1., desc='Thermal Resistance in m*k/W')

    def setup(self):
        self.add_input('T_in', units='K')
        self.add_input('T_out', units='K')
        self.add_input('R', 10., units='K/W')
        self.add_output('q', units='W')

        self.declare_partials('q', 'T_in')
        self.declare_partials('q', 'T_out')
        self.declare_partials('q', 'R')

    def compute(self, inputs, outputs):
        deltaT = inputs['T_in'] - inputs['T_out']
        outputs['q'] = deltaT / inputs['R']

    def compute_partials(self, inputs, J):
        J['q','T_in'] = 1./inputs['R']
        J['q','T_out'] = -1./inputs['R']
        J['q','R'] = -(inputs['T_in'] - inputs['T_out'])/inputs['R']**2

class Node(om.ImplicitComponent):
    """Computes temperature residual across a node based on incoming and outgoing flux."""

    def initialize(self):
        self.options.declare('n_in', default=1, types=int, desc='number of connections with + assumed in')
        self.options.declare('n_out', default=1, types=int, desc='number of current connections + assumed out')

    def setup(self):
        self.add_output('T', val=5., units='K',lower=1e-5)

        for i in range(self.options['n_in']):
            q_name = 'q_in:{}'.format(i)
            self.add_input(q_name, 0., units='W')

        for i in range(self.options['n_out']):
            q_name = 'q_out:{}'.format(i)
            self.add_input(q_name, 0., units='W')

    def setup_partials(self):
        #note: we don't declare any partials wrt `T` here,
        #      because the residual doesn't directly depend on it
        self.declare_partials('T', 'q*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['T'] = 0.
        for q_conn in range(self.options['n_in']):
            residuals['T'] += inputs['q_in:{}'.format(q_conn)]
        for q_conn in range(self.options['n_out']):
            residuals['T'] -= inputs['q_out:{}'.format(q_conn)]

    def linearize(self, inputs, outputs, partials):

        for q_conn in range(self.options['n_in']):
            partials['T','q_in:{}'.format(q_conn)] = 1.
        for q_conn in range(self.options['n_out']):
            partials['T','q_out:{}'.format(q_conn)] = -1.   

class Radial_Stack(om.Group):
    """ Used for both Condensor and Evaporator Thermal Resistance Stacks"""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)  
        self.options.declare('n_in', types=int, default=0)  # middle = 2, end = 1
        self.options.declare('n_out', types=int, default=0)  # middle = 2, end = 1

    def setup(self):
        n_in = self.options['n_in']
        n_out = self.options['n_out']
        nn = self.options['num_nodes']


        # Calculate Fluid Properties
        self.add_subsystem(name = 'fluids',
                           subsys = FluidPropertiesComp(num_nodes=nn),
                           promotes_inputs=['Q_hp', 'A_cond', 'h_c', 'T_coolant'],
                           promotes_outputs=['R_g', 'P_v', 'T_hp', 'rho_v', 'mu_v', 'h_fg'])

        # Calculate Resistances
        self.add_subsystem(name='radial',
                           subsys=RadialThermalResistance(num_nodes=nn),
                           promotes_inputs=['T_hp','v_fg','D_od','R_g','P_v','k_wk','A_inter','k_w','L_cond','r_i','D_v','h_fg','alpha'],
                           promotes_outputs=['R_w','R_wk','R_inter'])


        # Define Resistors
        self.add_subsystem('Rex', Resistor())
        self.add_subsystem('Rw', Resistor())#, promotes_inputs=[('T_in', 'T_hot')]) # evaporator wall
        self.add_subsystem('Rwk', Resistor()) # evaporator wick
        self.add_subsystem('Rinter', Resistor())

        self.add_subsystem('n1', Node(n_in=1+n_in, n_out=1+n_out))  # 1, 2 out
        self.add_subsystem('n2', Node(n_in=1+n_in, n_out=1+n_out))  # 1, 2 out
        self.add_subsystem('n3', Node(n_in=1, n_out=1))  # 1
        self.add_subsystem('n4', Node(n_in=1+n_in, n_out=1+n_out))  # 1

        # node 1 (6 connections, 1 in, 2 out)
        self.connect('n1.T', ['Rex.T_out','Rw.T_in'])
        self.connect('Rex.q','n1.q_in:0')
        self.connect('Rw.q','n1.q_out:0')

        # node 2 (6 connections, 1 in, 2 out)
        self.connect('n2.T', ['Rw.T_out', 'Rwk.T_in'])
        self.connect('Rw.q', 'n2.q_in:0')
        self.connect('Rwk.q', 'n2.q_out:0')

        # node 3 (4 connections)
        self.connect('n3.T', ['Rwk.T_out','Rinter.T_in'])
        self.connect('Rwk.q', 'n3.q_in:0')
        self.connect('Rinter.q', 'n3.q_out:0')

        # node 4 (4 connections)
        self.connect('n4.T', ['Rinter.T_out']) 
        self.connect('Rinter.q', 'n4.q_in:0')

        # connect resistances
        self.connect('R_w','Rw.R')
        self.connect('R_wk','Rwk.R')
        self.connect('R_inter','Rinter.R')


class Bridge(om.Group):
    """ Bridge between evaporator or condensors """
    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)  
    def setup(self):
        nn = self.options['num_nodes']

        # Compute Resistances
        self.add_subsystem(name='axial',
                           subsys=AxialThermalResistance(num_nodes=nn),
                           promotes_inputs=['epsilon', 'k_w', 'k_l', 'L_adiabatic', 'A_w', 'A_wk'])
                           #promotes_outputs=['k_wk', 'R_aw', 'R_awk']

        self.add_subsystem(name='vapor',
                           subsys=VaporThermalResistance(num_nodes=nn),
                           promotes_inputs=['D_v', 'R_g', 'mu_v', 'T_hp', 'h_fg', 'P_v', 'rho_v', 'L_eff'])
                           #promotes_outputs=['r_h', 'R_v']

        # Define Axial Resistors
        self.add_subsystem('Rv', Resistor()) # vapor channel
        self.add_subsystem('Rwka', Resistor()) # wick axial
        self.add_subsystem('Rwa', Resistor()) # wall 

        # connect
        self.connect('vapor.R_v','Rv.R')
        self.connect('axial.R_aw','Rwa.R')
        self.connect('axial.R_awk', 'Rwka.R')

def thermal_link(model, l_comp, r_comp):
    l_name = l_comp
    r_name = r_comp

    b_name = '{}_bridge'.format(l_comp)
    model.add_subsystem(b_name, Bridge(),
                        promotes_inputs=['L_adiabatic','A_w','A_wk','D_v','R_g','mu_v','h_fg','P_v','rho_v','L_eff','k_w','k_l'])

    #determine connection number

    # left node 1
    model.connect('{}.n1.T'.format(l_name),'{}.Rwa.T_in'.format(b_name))
    model.connect('{}.Rwa.q'.format(b_name),'{}.n1.q_out:1'.format(l_name))
    # left node 2
    model.connect('{}.n2.T'.format(l_name),'{}.Rwka.T_in'.format(b_name))
    model.connect('{}.Rwka.q'.format(b_name),'{}.n2.q_out:1'.format(l_name))
    # left node 4
    model.connect('{}.n4.T'.format(l_name),'{}.Rv.T_in'.format(b_name))
    model.connect('{}.Rv.q'.format(b_name),'{}.n4.q_out:0'.format(l_name))

    # right node 1
    model.connect('{}.n1.T'.format(r_name),'{}.Rwa.T_out'.format(b_name))
    model.connect('{}.Rwa.q'.format(b_name),'{}.n1.q_in:1'.format(r_name))
    # right node 2
    model.connect('{}.n2.T'.format(r_name),'{}.Rwka.T_out'.format(b_name))
    model.connect('{}.Rwka.q'.format(b_name),'{}.n2.q_in:1'.format(r_name))
    # right node 4
    model.connect('{}.n4.T'.format(r_name),'{}.Rv.T_out'.format(b_name))
    model.connect('{}.Rv.q'.format(b_name),'{}.n4.q_in:1'.format(r_name))


    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 20
    model.linear_solver = om.DirectSolver()
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
    model.nonlinear_solver.linesearch.options['maxiter'] = 10
    model.nonlinear_solver.linesearch.options['iprint'] = 2


class Circuit(om.Group):
    """ Full thermal equivalent circuit from one evaporator to one condensor"""
    def setup(self):

        # Evaporator
        self.add_subsystem('Rex_e', Resistor())
        self.add_subsystem('Rwe', Resistor())#, promotes_inputs=[('T_in', 'T_hot')]) # evaporator wall
        self.add_subsystem('Rwke', Resistor()) # evaporator wick
        self.add_subsystem('Rinter_e', Resistor())
        # Axial
        self.add_subsystem('Rv', Resistor()) # vapor
        self.add_subsystem('Rwka', Resistor()) # wick adiabatic
        self.add_subsystem('Rwa', Resistor()) # wall adiabatic
        # Condensor
        self.add_subsystem('Rinter_c', Resistor()) #
        self.add_subsystem('Rwkc', Resistor()) # condensor wick
        self.add_subsystem('Rwc', Resistor())#, promotes_inputs=[('T_out', 'T_cold')]) #condensor wall
        self.add_subsystem('Rex_c', Resistor())

        self.add_subsystem('n1', Node(n_in=1, n_out=2))  # 1, 2 out
        self.add_subsystem('n2', Node(n_in=1, n_out=2))  # 1, 2 out
        self.add_subsystem('n3', Node(n_in=1, n_out=1))  # 1
        self.add_subsystem('n4', Node(n_in=1, n_out=1))  # 1
        self.add_subsystem('n5', Node(n_in=1, n_out=1))  # 1
        self.add_subsystem('n6', Node(n_in=1, n_out=1))  # 1
        self.add_subsystem('n7', Node(n_in=2, n_out=1))  # 2 in, 1
        self.add_subsystem('n8', Node(n_in=2, n_out=1))  # 2 in, 1

        # node 1 (6 connections, 1 in, 2 out)
        self.connect('n1.T', ['Rex_e.T_out','Rwe.T_in', 'Rwa.T_in']) # define temperature node as resitor inputs
        self.connect('Rex_e.q','n1.q_in:0')
        self.connect('Rwe.q','n1.q_out:0')
        self.connect('Rwa.q','n1.q_out:1')

        # node 2 (6 connections, 1 in, 2 out)
        self.connect('n2.T', ['Rwe.T_out', 'Rwka.T_in','Rwke.T_in'])
        self.connect('Rwe.q', 'n2.q_in:0')
        self.connect('Rwka.q', 'n2.q_out:0') 
        self.connect('Rwke.q', 'n2.q_out:1')

        # node 3 (4 connections)
        self.connect('n3.T', ['Rwke.T_out','Rinter_e.T_in'])
        self.connect('Rwke.q', 'n3.q_in:0')
        self.connect('Rinter_e.q', 'n3.q_out:0')

        # node 4 (4 connections)
        self.connect('n4.T', ['Rinter_e.T_out','Rv.T_in'])
        self.connect('Rinter_e.q', 'n4.q_in:0')
        self.connect('Rv.q', 'n4.q_out:0')

        # node 5 (4 connections)
        self.connect('n5.T', ['Rv.T_out','Rinter_c.T_in'])
        self.connect('Rv.q', 'n5.q_in:0')
        self.connect('Rinter_c.q', 'n5.q_out:0')

        # node 6 (4 connections)
        self.connect('n6.T', ['Rinter_c.T_out','Rwkc.T_in'])
        self.connect('Rinter_c.q', 'n6.q_in:0')
        self.connect('Rwkc.q', 'n6.q_out:0')
        
        # node 7 (4 connections, 2 in, 1 out)
        self.connect('n7.T', ['Rwka.T_out','Rwkc.T_out','Rwc.T_in'])
        self.connect('Rwka.q', 'n7.q_in:0')
        self.connect('Rwkc.q', 'n7.q_in:1')
        self.connect('Rwc.q', 'n7.q_out:0')

        # node 8 (6 connections, 2 in, 1 out)
        self.connect('n8.T',['Rwa.T_out','Rwc.T_out','Rex_c.T_in'])
        self.connect('Rwa.q','n8.q_in:0')
        self.connect('Rwc.q','n8.q_in:1')
        self.connect('Rex_c.q','n8.q_out:0')



        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 20
        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.linesearch.options['maxiter'] = 10
        self.nonlinear_solver.linesearch.options['iprint'] = 2

    

if __name__ == "__main__":

    p = om.Problem()
    model = p.model

    # # Circuit
    # model.add_subsystem('circuit', Circuit())
    # p.setup()
    # p['circuit.Rex_e.T_in'] = 100.
    # p['circuit.Rex_c.T_out'] = 20.

    # # Simple Circuit
    # model.add_subsystem('evap', Radial_Stack(n_in=0, n_out=1))
    # model.add_subsystem('cond', Radial_Stack(n_in=1, n_out=0))
    # thermal_link(model,'evap','cond')

    # # CEC
    # model.add_subsystem('cond', Radial_Stack(n_in=0, n_out=1))
    # model.add_subsystem('evap', Radial_Stack(n_in=1, n_out=1))
    # model.add_subsystem('cond2', Radial_Stack(n_in=1, n_out=0))
    # thermal_link(model,'cond','evap')
    # thermal_link(model,'evap','cond2')

    # ECC
    model.add_subsystem('evap', Radial_Stack(n_in=0, n_out=1))
    model.add_subsystem('cond', Radial_Stack(n_in=1, n_out=1))
    model.add_subsystem('cond2', Radial_Stack(n_in=1, n_out=0))
    thermal_link(model,'evap','cond')
    thermal_link(model,'cond','cond2')

    p.setup()

    p.set_val('evap.Rex.T_in', 100.)
    p.set_val('cond.Rex.T_in', 20.)
    p.set_val('cond2.Rex.T_in', 20.)

    #p.check_partials(compact_print=True)

    p.run_model() 
    #om.n2(p)

    p.model.list_inputs(values=True, prom_name=True)   
    p.model.list_outputs(values=True, prom_name=True)   

    #print(p.get_val('cond.Rex.T_in'))


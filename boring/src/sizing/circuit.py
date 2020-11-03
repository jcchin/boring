"""
Construct thermal network, then solve for flux and equivalent resistance

Author: Jeff Chin
"""

import openmdao.api as om


class Resistor(om.ExplicitComponent):
    """Computes flux across a resistor using Ohm's law."""

    # def initialize(self):
    #     self.options.declare('R', default=1., desc='Thermal Resistance in m*k/W')

    def setup(self):
        self.add_input('T_in', units='K')
        self.add_input('T_out', units='K')
        self.add_input('R', 10., units='m*K/W')
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
            self.add_input(q_name, units='W')

        for i in range(self.options['n_out']):
            q_name = 'q_out:{}'.format(i)
            self.add_input(q_name, units='W')

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

class Evaporator(om.Group):
    """ Evaporator Stack """
    def setup(self):

        # Evaporator
        self.add_subsystem('Rex_e', Resistor())
        self.add_subsystem('Rwe', Resistor())#, promotes_inputs=[('T_in', 'T_hot')]) # evaporator wall
        self.add_subsystem('Rwke', Resistor()) # evaporator wick
        self.add_subsystem('Rinter_e', Resistor())

        self.add_subsystem('n1', Node(n_in=1, n_out=2))  # 1, 2 out
        self.add_subsystem('n2', Node(n_in=1, n_out=2))  # 1, 2 out
        self.add_subsystem('n3', Node(n_in=1, n_out=1))  # 1
        self.add_subsystem('n4', Node(n_in=1, n_out=1))  # 1

        # node 1 (6 connections, 1 in, 2 out)
        self.connect('n1.T', ['Rex_e.T_out','Rwe.T_in'])
        self.connect('Rex_e.q','n1.q_in:0')
        self.connect('Rwe.q','n1.q_out:0')

        # node 2 (6 connections, 1 in, 2 out)
        self.connect('n2.T', ['Rwe.T_out', 'Rwke.T_in'])
        self.connect('Rwe.q', 'n2.q_in:0')
        self.connect('Rwke.q', 'n2.q_out:1')

        # node 3 (4 connections)
        self.connect('n3.T', ['Rwke.T_out','Rinter_e.T_in'])
        self.connect('Rwke.q', 'n3.q_in:0')
        self.connect('Rinter_e.q', 'n3.q_out:0')

        # node 4 (4 connections)
        self.connect('n4.T', ['Rinter_e.T_out']) 
        self.connect('Rinter_e.q', 'n4.q_in:0')


class Condensor(om.Group):
    """ Condensor Stack """
    def setup(self):

        # Condensor
        self.add_subsystem('Rinter_c', Resistor()) #
        self.add_subsystem('Rwkc', Resistor()) # condensor wick
        self.add_subsystem('Rwc', Resistor())#, promotes_inputs=[('T_out', 'T_cold')]) #condensor wall
        self.add_subsystem('Rex_c', Resistor())

        self.add_subsystem('n5', Node(n_in=1, n_out=1))  # 1
        self.add_subsystem('n6', Node(n_in=1, n_out=1))  # 1
        self.add_subsystem('n7', Node(n_in=2, n_out=1))  # 2 in, 1
        self.add_subsystem('n8', Node(n_in=2, n_out=1))  # 2 in, 1

        # node 5 (4 connections)
        self.connect('n4.T', ['Rinter_c.T_in']) 
        self.connect('Rinter_c.q', 'n4.q_out:0')

        # node 6 (4 connections)
        self.connect('n3.T', ['Rinter_c.T_out','Rwkc.T_in'])
        self.connect('Rinter_c.q', 'n3.q_in:0')
        self.connect('Rwkc.q', 'n3.q_out:0')
        
        # node 7 (4 connections, 2 in, 1 out)
        self.connect('n2.T', ['Rwkc.T_out','Rwc.T_in']) 
        self.connect('Rwkc.q', 'n2.q_in:1')
        self.connect('Rwc.q', 'n2.q_out:0')

        # node 8 (6 connections, 2 in, 1 out)
        self.connect('n1.T',['Rwc.T_out','Rex_c.T_in']) 
        self.connect('Rwc.q','n1.q_in:1')
        self.connect('Rex_c.q','n1.q_out:0')

class Bridge(om.Group):
    """ Bridge between evaporator or condensors """
    def setup(self):

        # Axial
        self.add_subsystem('Rv', Resistor()) # vapor
        self.add_subsystem('Rwka', Resistor()) # wick adiabatic
        self.add_subsystem('Rwa', Resistor()) # wall adiabatic

def thermal_link(l_comp, r_comp, type):
    l_name = l_comp.name
    r_name = r_comp.name

    #determine connection number

    if type('VA'): # connect a vertical (condensor or evaportor) to axial (bridge) component
        # node 1
        self.connect('{}.n1.T'.format(l_name),'{}.Rwa.T_in'.format(b_name))
        self.connect('{}.Rwa.q'.format(b_name),'{}.n1.q_out:1'.format(l_name))
        # node 2
        self.connect('{}.n2.T'.format(l_name),'{}.Rwka.T_in'.format(r_name))
        self.connect('{}.Rwka.q'.format(b_name),'{}.n2.q_out:0'.format(l_name))
        # node 4
        self.connect('{}.n4.T'.format(l_name),'{}.Rv.T_in'.format(b_name))
        self.connect('{}.Rv.q'.format(b_name),'{}.n4.q_out:0'.format(l_name))

    elif type('AV'): # connect a axial (bridge) component to a vertical component
        # node 1 (8)
        self.connect('{}.n1.T'.format(r_name),'{}.Rwa.T_out'.format(b_name))
        self.connect('{}.Rwa.q'.format(b_name),'{}.n1.q_in:1'.format(r_name))
        # node 2 (7)
        self.connect('{}.n2.T'.format(r_name),'{}.Rwka.T_out'.format(b_name))
        self.connect('{}.Rwka.q'.format(b_name),'{}.n2.q_in:0'.format(r_name))
        # node 4 (5)
        self.connect('{}.n4.T'.format(r_name),'{}.Rv.T_out'.format(b_name))
        self.connect('{}.Rv.q'.format(b_name),'{}.n4.q_in:0'.format(r_name))

    
    else:   
        print("invalid connection type")



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

    # model.add_subsystem('T_hot', om.IndepVarComp('T', 500., units='K'))
    # model.add_subsystem('T_cold', om.IndepVarComp('T', 60, units='K'))
    #model.add_subsystem('q_hot', om.IndepVarComp('q', 70., units='W'))
    #model.add_subsystem('q_cold', om.IndepVarComp('q', 50, units='W'))
    model.add_subsystem('circuit', Circuit())

    # model.connect('T_hot.T', 'circuit.n1.T')
    # model.connect('T_cold.T', 'circuit.n2.T')
    # model.connect('q_hot.q', 'circuit.n1.q_in:0')
    # model.connect('q_cold.q', 'circuit.n2.q_out:0')

    p.setup()

    p.set_val('circuit.Rex_e.T_in', 100.)
    p.set_val('circuit.Rex_c.T_out', 20.)

    #p.check_partials(compact_print=True)
    #om.n2(p)

    # set some initial guesses

    # p['circuit.n3.T'] = 300.
    # p['circuit.n4.T'] = 250.
    # p['circuit.n5.T'] = 200.
    # p['circuit.n6.T'] = 150.
    # p['circuit.n7.T'] = 100.
    # p['circuit.n8.T'] = 60.

    p.run_model() 

    p.model.list_outputs()   
    #om.n2(p)    
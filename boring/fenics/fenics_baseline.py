"""
High Fidelity Baseline Model

See Wiki for Install Requirements
(https://github.com/jcchin/boring/wiki)

Author: Jeff Chin
"""

from dolfin import *
from mshr import *

import openmdao.api as om
import matplotlib.pyplot as plt;
from matplotlib.animation import FuncAnimation

#from IPython.display import clear_output, display; import time; import dolfin.common.plotting as fenicsplot 
import time
import numpy as np
import os, sys, shutil

# Classes to define different material properties

class Conductivity(UserExpression):
    def __init__(self, markers, **kwargs):
        super(Conductivity, self).__init__(**kwargs)
        self.markers = markers
    def eval_cell(self, values, x, cell):
        if self.markers[cell.index] == 0:
            values[0] = 204      # aluminum
        else:
            values[0] = 1.3      # batteries

class Density(UserExpression):
    def __init__(self, markers, **kwargs):
        super(Density, self).__init__(**kwargs)
        self.markers = markers
    def eval_cell(self, values, x, cell):
        if self.markers[cell.index] == 0:
            values[0] = 2700      # aluminum
        else:
            values[0] = 1460      # batteries

class SpecificHeat(UserExpression):
    def __init__(self, markers, **kwargs):
        super(SpecificHeat, self).__init__(**kwargs)
        self.markers = markers
    def eval_cell(self, values, x, cell):
        if self.markers[cell.index] == 0:
            values[0] = 883      # aluminum
        else:
            values[0] = 880      # batteries


class FenicsBaseline(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)


    def setup(self):
        nn = self.options['num_nodes']

        #self.add_input('cell_d', 0.018, units='m', desc='diameter of an 18650 cell')
        self.add_input('extra',  1.1516, desc='extra spacing along the diagonal')
        self.add_input('ratio', 0.7381, desc='cell radius to cutout radius')
        self.add_input('energy', 12, units='kJ', desc='energy in heat load')
        #self.add_input('al_density', 2700, units='kg/m**3', desc='density of aluminum')
        #self.add_input('n_cells', 4,  desc='cell array deminsion')

        #self.add_output('temp1', 100, units='degC', desc='runaway cell temp')
        self.add_output('temp2_data', 100, desc='neighboring cell temp')
        self.add_output('temp3_data', 100, desc='diagonal neighboring cell temp')


    def setup_partials(self):
        self.declare_partials('*','extra', method='fd', step=0.001)
        #self.declare_partials('*','ratio', method='fd', step=0.01)


    def compute(self,i,o):
        # Define domain and mesh
        cell_d = 0.018#i['cell_d']
        extra = i['extra']  # extra space along the diagonal
        ratio = i['ratio']  # cell diameter/cutout diameter
        energy = float(i['energy'])
        n_cells = 4#int(i['n_cells'])  # number of cells
        
        side = cell_d*extra*n_cells;  # square side length
        hole_r = ratio*0.5*cell_d*((2**0.5*extra)-1)
        offset = cell_d*extra

        XMIN, XMAX = 0, side; 
        YMIN, YMAX = 0, side; 
        G = [XMIN, XMAX, YMIN, YMAX];
        mresolution = 30; # number of cells     Vary mesh density and capture CPU time - ACTION ITEM NOW

        # Define 2D geometry
        holes = [Circle(Point(offset*i,offset*j), hole_r) for j in range(n_cells+1) for i in range(n_cells+1)] # nested list comprehension creates a size (n+1)*(n+1) array

        domain = Rectangle(Point(G[0], G[2]), Point(G[1], G[3]))
        for hole in holes:
          domain = domain - hole

        off2 = cell_d/2+ cell_d*(extra-1)/2
        batts = [Circle(Point(off2+offset*i,off2+offset*j), cell_d/2) for j in range(n_cells) for i in range(n_cells)]

        for (i, batt) in enumerate(batts):
          domain.set_subdomain(1+i, batt) # add cells to domain as a sub-domain
        mesh1 = generate_mesh(domain, mresolution)

        # Mark the sub-domains
        markers = MeshFunction('size_t', mesh1, 2, mesh1.domains())  # size_t = non-negative integers
        dx = Measure('dx', domain=mesh1, subdomain_data=markers)
        # Extract facets to apply boundary conditions for internal boundaries
        boundaries = MeshFunction('size_t', mesh1, mesh1.topology().dim()-1)
  
        # Materials
        k = Conductivity(markers, degree=1)  
        rho = Density(markers, degree=1)  
        cp = SpecificHeat(markers, degree=1)     

        # Define finite element function space
        degree = 1;
        V = FunctionSpace(mesh1, "CG", degree);

        # Finite element functions
        v = TestFunction(V); 
        u = Function(V);

        # Time parameters
        theta = 1.0 # Implicit Euler
        dt = 0.5; # Time step
        t, T = 0., 5.; # Start and end time


        # Boundray Conditions           
        tol = 1E-14
        for f in facets(mesh1):
            domains = []
            for c in cells(f):
                domains.append(markers[c])
            domains = list(set(domains))
            # Domains with [0,i] refer to each boundary within the "copper" plate
            # Domains 1-9 refer to the domains of the batteries
            boundaries[f] = int(domains[0])


        u1= Constant(298.0)
        ue = Constant(325.0)
        u0 = u1;

        q = Constant((energy*1000/2)*62500) # Neumann Heat Flux
        R = Constant("333")       # Inverse Contact resistance - [W/(m^2*K)]

        # Inititalize time stepping
        stepcounter = 0; 

        # Define new measures associated with the interior boundaries
        dS = Measure('dS', domain=mesh1, subdomain_data=boundaries)
        # Time-stepping loop
        start_time = time.time()

        T2_max = 0
        T3_max = 0

        def maxsub(f, subdomains, subd_id):
            '''Minimum of f over subdomains cells marked with subd_id'''
            
            V = f.function_space()
            dm = V.dofmap()

            subd_dofs = np.unique(np.hstack(
                [dm.cell_dofs(c.index()) for c in SubsetIterator(subdomains, subd_id)]))
            
            return np.max(f.vector().get_local()[subd_dofs])

        # Time-stepping loop
        while t < T: 
            # Time scheme
            um = theta*u + (1.0-theta)*u0 
            # Heat for first 2 seconds only
            if t > 2:
                q = Constant("0")

            terms = rho*cp*u*v*dx + dt*inner(k*grad(u), grad(v))*dx - rho*cp*u0*v*dx -dt*q*v*dx(1)
            lhs = []
            for i in range(n_cells**2):
              lhs.append(dt*R*u('+')*v('+')*dS(i+1) - dt*R*u1*v('+')*dS(i+1))
            a = terms + sum(lhs)
            # a = rho*cp*u*v*dx + dt*inner(k*grad(u), grad(v))*dx - rho*cp*u0*v*dx - dt*q*v*dx(1) + dt*R*u('+')*v('+')*dS(1) - dt*R*u1*v('+')*dS(1)
            # + dt*R*u('+')*v('+')*dS(2) - dt*R*u1*v('+')*dS(2) + dt*R*u('+')*v('+')*dS(3) - dt*R*u1*v('+')*dS(3) + dt*R*u('+')*v('+')*dS(4) - dt*R*u1*v('+')*dS(4)
            # + dt*R*u('+')*v('+')*dS(5) - dt*R*u1*v('+')*dS(5) + dt*R*u('+')*v('+')*dS(6) - dt*R*u1*v('+')*dS(6) + dt*R*u('+')*v('+')*dS(7) - dt*R*u1*v('+')*dS(7)
            # + dt*R*u('+')*v('+')*dS(8) - dt*R*u1*v('+')*dS(8) + dt*R*u('+')*v('+')*dS(9) - dt*R*u1*v('+')*dS(9)
            # Solve the Heat equation (one timestep)
            solve(a==0, u)  
            # Shift to next timestep
            t += dt; u0 = project(u, V); 
            stepcounter += 1

            T2_max = max(T2_max,maxsub(u,markers,2))
            Td_max = max(T3_max,maxsub(u,markers,n_cells+2))


        # Evaluate Temperature of neighboring batteries
        # cells are numbered from the bottom, left to right, then up a row
        vol = assemble(Constant('1.0')*dx(1)) # Compute the area/volume of 1 battery cell
        T_1 = assemble(u*dx(1))/vol           # Compute area-average T over cell 1
        T_2 = maxsub(u,markers,2)           # Compute area-max T over cell 2
        
        print("********")
        print("Extra: ", extra)
        print("Ratio: ", ratio)
        print("Energy: ", energy)

        print("Average Final T_1[K] = ", T_1)
        print("Max Overall T_2[K] = ", T2_max)
        print("Max Final T_2[K] = ", T_2)
        print("Max Overall Td [K] = ", Td_max)
        print("Temp Ratio = ", T2_max/Td_max)
        print("Maximum Final T[K] = ", u.vector().max())
        print("Minimum Final T[K] = ", u.vector().min())

        #o['temp1'] = u.vector().max()
        o['temp2_data'] = T2_max
        o['temp3_data'] = Td_max

if __name__ == '__main__':
    from openmdao.api import Problem

    nn = 1
    prob = Problem()

    prob.model.add_subsystem('baseline_temp', FenicsBaseline(num_nodes=nn), promotes=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()

    #print(prob.get_val('temp1'))
    print(prob.get_val('temp2_data'))
    print(prob.get_val('temp3_data'))

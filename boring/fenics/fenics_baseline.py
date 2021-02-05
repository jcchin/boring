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

from IPython.display import clear_output, display; import time; import dolfin.common.plotting as fenicsplot 
import time

import os, sys, shutil

class Conductivity(UserExpression):
    def __init__(self, markers, **kwargs):
        super(Conductivity, self).__init__(**kwargs)
        self.markers = markers
    def eval_cell(self, values, x, cell):
        if self.markers[cell.index] == 0:
            values[0] = 11.4 # copper
        else:
            values[0] = 1      # aluminum


class FenicsBaseline(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)


    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('cell_d', 18, units='mm', desc='diameter of an 18650 cell')
        self.add_input('extra',  1, units='mm', desc='extra spacing along the diagonal')
        self.add_input('ratio', 1., desc='cell radius to cutout radius')
        self.add_input('al_density', 2.7e-6, units='kg/mm**3', desc='density of aluminum')
        self.add_input('n_cells', 4,  desc='cell array deminsion')

        self.add_output('temp1', 100, units='degC', desc='neighboring cell temp')

    def setup_partials(self):
        self.declare_partials('*','*', method='fd')

    def compute(self,i,o):
        # Define domain and mesh
        cell_d = i['cell_d']
        extra = i['extra']  # extra space along the diagonal
        ratio = i['ratio']  # cell diameter/cutout diameter
        n_cells = int(i['n_cells'])  # number of cells
        diagonal = (n_cells*cell_d)+((n_cells)*(cell_d/ratio))*extra;  # diagonal distance from corner to corner
        side = diagonal/(2**0.5);  # square side length

        G = [0, side, 0, side];
        mresolution = 30; # number of cells

        # Define 2D geometry

        offset = side/n_cells
        holes = [Circle(Point(offset*i,offset*j), cell_d/(2*ratio)) for j in range(n_cells+1) for i in range(n_cells+1)] # nested list comprehension creates a size (n+1)*(n+1) array

        domain = Rectangle(Point(G[0], G[2]), Point(G[1], G[3]))
        for hole in holes:
          domain = domain - hole

        offset2 = side/n_cells
        batts = [Circle(Point(side/(2*n_cells)+offset2*i,side/(2*n_cells)+offset2*j), cell_d/2) for j in range(n_cells) for i in range(n_cells)]

        for (i, batt) in enumerate(batts):
          domain.set_subdomain(1+i, batt) # add cells to domain as a sub-domain
        mesh = generate_mesh(domain, mresolution)

        markers = MeshFunction('size_t', mesh, 2, mesh.domains())  # size_t = non-negative integers
        dx = Measure('dx', domain=mesh, subdomain_data=markers)
        boundaries = MeshFunction('size_t', mesh, 1, mesh.domains())
  
        #k_coeff = 0.5
        k_coeff = Conductivity(markers, degree=1)

        # Define finite element function space
        degree = 1;
        V = FunctionSpace(mesh, "CG", degree);

        # Finite element functions
        v = TestFunction(V); 
        u = Function(V);

        # Time parameters
        theta = 1.0 # Implicit Euler
        k = 0.05; # Time step
        t, T = 0., 5.; # Start and end time

        # Exact solution
        kappa = 1e-1
        rho = 1.0;

        # Boundray Conditions (Ezra)
        tol = 1E-14
        for f in facets(mesh):
            domains = []
            for c in cells(f):
                domains.append(markers[c])

            domains = list(set(domains))
            # Domains with [0,i] refer to each boundary within the "copper" plate
            # Domains 1-9 refer to the domains of the batteries
            if domains == [1]:
                boundaries[f] = 2

        u1= Constant(298.0)
        ue = Constant(325.0)
        ue.t = k/10.;
        u0 = u1;

        Q = Constant(71000)
        L_N = Q*v*dx(2)

        bc_D = DirichletBC(V, ue, boundaries, 2)

        # Inititalize time stepping
        pl, ax = None, None; 
        stepcounter = 0; 
        timer0 = time.time()

        # Time-stepping loop
        while t < T: 
            # Time scheme
            um = theta*u + (1.0-theta)*u0 
            # Weak form of the heat equation in residual form
            r = (u - u0)/k*v*dx + k_coeff*inner(grad(um), grad(v))*dx 
            # Solve the Heat equation (one timestep)
            solve(r==0, u, bc_D)  
            # Shift to next timestep
            t += k; u0 = project(u, V); 
            ue.t = t;
            stepcounter += 1 


        W = VectorFunctionSpace(mesh, 'P', degree)
        flux_u = project(-k_coeff*grad(u),W)

        print(flux_u.vector().max())

        # Evaluate Temperature of neighboring batteries
        vol = assemble(Constant('1.0')*dx(1)) # Compute the area/volume of 1 battery cell
        T_1 = assemble(u*dx(1))/vol           # Compute area-average T over cell 1
        T_2 = assemble(u*dx(2))/vol           # Compute area-average T over cell 2
        T_4 = assemble(u*dx(4))/vol           # Compute area-average T over cell 4
        T_5 = assemble(u*dx(5))/vol           # Compute area-average T over cell 5
        print("Average T_1[K] = ", T_1)
        print("Average T_2[K] = ", T_2)
        print("Average T_4[K] = ", T_4)
        print("Average T_5[K] = ", T_5)
        print("Maximum T[K] = ", u.vector().max())
        print("Minimum T[K] = ", u.vector().min())

        o['temp1'] = u.vector().max()

if __name__ == '__main__':
    from openmdao.api import Problem

    nn = 1
    prob = Problem()

    prob.model.add_subsystem('baseline_temp', FenicsBaseline(num_nodes=nn), promotes=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()

    print(prob.get_val('temp1'))

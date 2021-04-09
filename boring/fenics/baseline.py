"""
High Fidelity Baseline Model

See Wiki for Install Requirements
(https://github.com/jcchin/boring/wiki)

Author: Jeff Chin
"""

from dolfin import *
from mshr import *

import matplotlib.pyplot as plt;
from matplotlib.animation import FuncAnimation

from IPython.display import clear_output, display; import time; import dolfin.common.plotting as fenicsplot 
import time

import os, sys, shutil

start_time = time.time()

# Define domain and mesh
cell_d = 1.8; # diameter of the cell
extra = 1.5;  # extra space along the diagonal
ratio = 2.;  # cell diameter/cutout diameter
n_cells = 7;  # number of cells
diagonal = (n_cells*cell_d)+((n_cells)*(cell_d/ratio))*extra;  # diagonal distance from corner to corner
side = diagonal/(2**0.5);  # square side length

XMIN, XMAX = 0, side; 
YMIN, YMAX = 0, side; 
G = [XMIN, XMAX, YMIN, YMAX];
mresolution = 40; # number of cells

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


#mesh = generate_mesh(Rectangle(Point(G[0], G[2]), Point(G[1], G[3])), mresolution)
#mesh = generate_mesh(Circle(Point(0.0), 2.0), mresolution)

markers = MeshFunction('size_t', mesh, 2, mesh.domains())  # size_t = non-negative integers
dx = Measure('dx', domain=mesh, subdomain_data=markers)
boundaries = MeshFunction('size_t', mesh, 1, mesh.domains())


# plot(markers,"Subdomains")

# plt.show()


def plot_compact(u, t, stepcounter, QQ, pl, ax): # Compact plot utility function
  if stepcounter == 0:
    pl, ax = plt.subplots(); display(pl); clear_output(); # Plotting setup
  if stepcounter % 5 == 0:
    uEuclidnorm = project(sqrt(inner(u, u)), QQ);
    #plt.clf();
    fig = plt.gcf();
    fig.set_size_inches(16, 4)
    plt.subplot(1, 2, 1); pp = plot(uEuclidnorm, cmap="coolwarm"); plt.title("Solution at t=%f" % (t)) # Plot norm of velocity
    if t == 0.: plt.axis(G); # plt.colorbar(pp, shrink=0.5); 
    plt.subplot(1, 2, 2);
    #if t == 0.: plot(QQ.mesh()); plt.title("Mesh") # Plot mesh
    vol = assemble(Constant('1.0')*dx(1)) # Compute the area/volume of 1 battery cell
    T_2 = assemble(u*dx(2))/vol           # Compute area-average T over cell 2
    plt.plot(t,T_2,marker='o',color='k')
    plt.xlabel('Time[s]')
    plt.ylabel('Temperature [K]')
    plt.tight_layout(); dpl = display(pl, display_id="test");
  
  return (pl, ax)

class Conductivity(UserExpression):
    def __init__(self, markers, **kwargs):
        super(Conductivity, self).__init__(**kwargs)
        self.markers = markers
    def eval_cell(self, values, x, cell):
        if self.markers[cell.index] == 0:
            values[0] = 201./(900*2700) # aluminum
        else:
            values[0] = 1./(800*2800)      # battery


  
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

Q = Constant(12000.*(1/((pi*(9^2)*65)/1000000000))/10.)
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
    
    # Plot all quantities (see implementation above)
    #plt.clf()
    

    #pl, ax=plot_compact(u, t, stepcounter, V, pl, ax)
    

    # ani = FuncAnimation(plt.figure(), plot_compact, blit=True)
    # plt.show()
    # Shift to next timestep
    t += k; u0 = project(u, V); 
    ue.t = t;
    stepcounter += 1 


# W = VectorFunctionSpace(mesh, 'P', degree)
# flux_u = project(-k_coeff*grad(u),W)
# plot(flux_u, title='Flux Field')
# plt.show()
# print(flux_u.vector().max())

# Evaluate Temperature of neighboring batteries
vol = assemble(Constant('1.0')*dx(1)) # Compute the area/volume of 1 battery cell
T_1 = assemble(u*dx(1))/vol           # Compute area-average T over cell 1
T_2 = assemble(u*dx(2))/vol           # Compute area-average T over cell 2
T_4 = assemble(u*dx(4))/vol           # Compute area-average T over cell 4
T_5 = assemble(u*dx(5))/vol           # Compute area-average T over cell 4
print("Average T_1[K] = ", T_1)
print("Average T_2[K] = ", T_2)
print("Average T_4[K] = ", T_4)
print("Average T_5[K] = ", T_5)
print("Maximum T[K] = ", u.vector().max())
print("Minimum T[K] = ", u.vector().min())
# c = plot(u,cmap='jet');
# plt.colorbar(c);
# plt.title("Solution at t=" +str(T));
# plt.show()

# # Scaled variables
# L = 1; W = 1
# mu = 1
# rho = 1
# delta = W/L
# gamma = 0.4*delta**2
# beta = 1.25
# lambda_ = beta
# G_force = 20
# g = gamma

# # Define function space
# V = VectorFunctionSpace(mesh, 'P', 1);

# # Define boundary conditions
# tol = 1E-14

# def clamped_boundary(x, on_boundary):
#     return on_boundary and near(x[1], 0, tol)

# bc = DirichletBC(V, Constant((0,0)), clamped_boundary)

# # Define strain and stress
# def epsilon(u):
#     return 0.5*(nabla_grad(u) + nabla_grad(u).T)

# def sigma(u):
#     return lambda_*nabla_grad(u)*Identity(d) + 2*mu*epsilon(u)

# # Define variational problem
# u = TrialFunction(V)
# d = u.geometric_dimension()  # space dimension
# v = TestFunction(V)
# f = Constant((G_force*g, -G_force*g))   # Body force per unit volume
# T = Constant((0, 0))                    # Traction forces (tension/compression)
# a = inner(sigma(u), epsilon(v))*dx
# L = dot(f, v)*dx + dot(T, v)*ds

# # Compute solution
# u = Function(V)
# solve(a == L, u, bc)

# # Plot stress
# s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d)  # deviatoric stress
# von_Mises = sqrt(3./2*inner(s, s))
# V = FunctionSpace(mesh, 'P', 1)
# von_Mises = project(von_Mises, V)
# c = plot(von_Mises,cmap='jet');
# plt.colorbar(c);
# plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
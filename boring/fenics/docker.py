from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import time 

tfile = File("output/Thermal.pvd")
sfile = File("output/Structural.pvd")

# Define domain and mesh
cell_d = 0.018; # diameter of the cell
extra = 1.1516;  # extra space along the diagonal
ratio = 0.7381;  # cell diameter/cutout diameter
n_cells = 3;  # number of cells     VARY THIS AS WELL 


def make_mesh():

    side = cell_d*extra*n_cells;  # square side length
    hole_r = ratio*0.5*cell_d*((2**0.5*extra)-1)
    offset = cell_d*extra

    XMIN, XMAX = 0, side; 
    YMIN, YMAX = 0, side; 
    G = [XMIN, XMAX, YMIN, YMAX];
    mresolution = 50; # number of cells     Vary mesh density and capture CPU time - ACTION ITEM NOW

    # Define 2D geometry
    holes = [Circle(Point(offset*i,offset*j), hole_r) for j in range(n_cells+1) for i in range(n_cells+1)] # nested list comprehension creates a size (n+1)*(n+1) array

    domain = Rectangle(Point(G[0], G[2]), Point(G[1], G[3]))
    for hole in holes:
      domain = domain - hole

    off2 = cell_d/2+ cell_d*(extra-1)/2
    batts = [Circle(Point(off2+offset*i,off2+offset*j), cell_d/2) for j in range(n_cells) for i in range(n_cells)]

    for (i, batt) in enumerate(batts):
      domain.set_subdomain(1+i, batt) # add cells to domain as a sub-domain
    mesh = generate_mesh(domain, mresolution)

    return mesh

mesh1 = make_mesh()
# Mark the sub-domains
markers = MeshFunction('size_t', mesh1, 2, mesh1.domains())  # size_t = non-negative int
dx = Measure('dx', domain=mesh1, subdomain_data=markers)
# Extract facets to apply boundary conditions for internal boundaries
boundaries = MeshFunction('size_t', mesh1, mesh1.topology().dim()-1)

########################################################## Heat Transfer Solver #####################################################################
# Conductivity class to allow for different materials
class Conductivity(UserExpression):
    def __init__(self, markers, **kwargs):
        super(Conductivity, self).__init__(**kwargs)
        self.markers = markers
    def eval_cell(self, values, x, cell):
        if self.markers[cell.index] == 0:
            values[0] = 204 # aluminum
        else:
            values[0] = 1.3      # batteries
k = Conductivity(markers, degree=1)    
# Density class to allow for different materials
class Density(UserExpression):
    def __init__(self, markers, **kwargs):
        super(Density, self).__init__(**kwargs)
        self.markers = markers
    def eval_cell(self, values, x, cell):
        if self.markers[cell.index] == 0:
            values[0] = 2700 # aluminum
        else:
            values[0] = 1460      # batteries
rho = Density(markers, degree=1)     
# Specific Heat class to allow for different materials
class SpecificHeat(UserExpression):
    def __init__(self, markers, **kwargs):
        super(SpecificHeat, self).__init__(**kwargs)
        self.markers = markers
    def eval_cell(self, values, x, cell):
        if self.markers[cell.index] == 0:
            values[0] = 883 # aluminum
        else:
            values[0] = 880      # batteries
cp = SpecificHeat(markers, degree=1)     
# Define finite element function space
degree = 1;
V = FunctionSpace(mesh1, "CG", degree);

# Finite element functions
v = TestFunction(V); 
u = Function(V);

# Time parameters
theta = 1.0 # Implicit Euler
dt = 0.05; # Time step
t, T = 0., 5.; # Start and end time  PERFORM STUDY ON THIS AS WELL, VARY END TIME
def set_bcs():
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
        # if domains == [1]:
        #   boundaries[f] = 1
        # elif domains == [2]:
        #   boundaries[f] = 2
        # elif domains == [3]:
        #   boundaries[f] = 3
        # elif domains == [4]:
        #   boundaries[f] = 4
        # elif domains == [5]:
        #   boundaries[f] = 5
        # elif domains == [6]:
        #   boundaries[f] = 6
        # elif domains == [7]:
        #   boundaries[f] = 7
        # elif domains == [8]:
        #   boundaries[f] = 8
        # elif domains == [9]:
        #   boundaries[f] = 9
        # else:
        #   boundaries[f] = 0

set_bcs()

u1= Constant(298.0)
ue = Constant(325)
u0 = u1;

q = Constant("750000000") # Neumann Heat Flux

#R = Constant("0.003")   # Contact resistance - [K*m^2/W]
R = Constant("333")       # Inverse Contact resistance - [W/(m^2*K)]

# Inititalize time stepping
stepcounter = 0; 

# Define new measures associated with the interior boundaries
dS = Measure('dS', domain=mesh1, subdomain_data=boundaries)
vol = assemble(Constant('1.0')*dx(1))
# Time-stepping loop
start_time = time.time()

terms = [rho*cp*u*v*dx + dt*inner(k*grad(u), grad(v))*dx - rho*cp*u0*v*dx -dt*q*v*dx(1)]
for i in range(9):
    terms.extend([dt*R*u('+')*v('+')*dS(i+1) - dt*R*u1*v('+')*dS(i+1)])

while t < T: 
    # Time scheme
    um = theta*u + (1.0-theta)*u0 
    # Weak form of the heat equation in residual form
    #r = (u - u0)/k*v*dx + k_coeff*inner(grad(um), grad(v))*dx - q*v*dx(1) - R*um('+')*v('+')*dS(1) + R*u1*v('+')*dS(1)   # With Neumann BCs -f*v('+')*dS(1)
    if t > 2:
        q = Constant("0")
    a = sum(terms)
    a = rho*cp*u*v*dx + dt*inner(k*grad(u), grad(v))*dx - rho*cp*u0*v*dx - dt*q*v*dx(1) + dt*R*u('+')*v('+')*dS(1) - dt*R*u1*v('+')*dS(1)
    + dt*R*u('+')*v('+')*dS(2) - dt*R*u1*v('+')*dS(2) + dt*R*u('+')*v('+')*dS(3) - dt*R*u1*v('+')*dS(3) + dt*R*u('+')*v('+')*dS(4) - dt*R*u1*v('+')*dS(4)
    + dt*R*u('+')*v('+')*dS(5) - dt*R*u1*v('+')*dS(5) + dt*R*u('+')*v('+')*dS(6) - dt*R*u1*v('+')*dS(6) + dt*R*u('+')*v('+')*dS(7) - dt*R*u1*v('+')*dS(7)
    + dt*R*u('+')*v('+')*dS(8) - dt*R*u1*v('+')*dS(8) + dt*R*u('+')*v('+')*dS(9) - dt*R*u1*v('+')*dS(9)
    # Solve the Heat equation (one timestep)
    solve(a==0, u)      # Add bcs to this if using Dirichlet BCs
    # Shift to next timestep
    t += dt; u0 = project(u, V); 
    #ue.t = t;
    stepcounter += 1 
    # Output solution to the VTK file
    tfile << u, t

####################################################################### Structural Solver ########################################################
# Scaled variables
L = 1; W = 1
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
G_force = 20
g = gamma

# Define function space
V = VectorFunctionSpace(mesh1, 'P', 1);

# Define boundary conditions
tol = 1E-14

def clamped_boundary(x, on_boundary):
    return on_boundary and near(x[1], 0, tol)

bc = DirichletBC(V, Constant((0,0)), clamped_boundary)

# Define strain and stress
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return lambda_*nabla_grad(u)*Identity(d) + 2*mu*epsilon(u)

# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension()  # space dimension
v = TestFunction(V)
f = Constant((G_force*g, -G_force*g))   # Body force per unit volume
T = Constant((0, 0))                    # Traction forces (tension/compression)
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d)  # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s))
V = FunctionSpace(mesh1, 'P', 1)
von_Mises = project(von_Mises, V)
sfile << von_Mises

print("--- %s seconds ---" % (time.time() - start_time))
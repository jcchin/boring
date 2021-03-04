from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import time 

tfile = File("output/Thermal.pvd")
sfile = File("output/Structural.pvd")

# Define domain and mesh
cell_d = 1.8; # diameter of the cell
extra = 1.5;  # extra space along the diagonal
ratio = 2.;  # cell diameter/cutout diameter
n_cells = 3;  # number of cells     VARY THIS AS WELL 
diagonal = (n_cells*cell_d)+((n_cells)*(cell_d/ratio))+extra;  # diagonal distance from corner to corner
side = diagonal/(2**0.5);  # square side length

XMIN, XMAX = 0, side; 
YMIN, YMAX = 0, side; 
G = [XMIN, XMAX, YMIN, YMAX];
mresolution = 50; # number of cells     Vary mesh density and capture CPU time - ACTION ITEM NOW

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

# Mark the sub-domains
markers = MeshFunction('size_t', mesh, 2, mesh.domains())  # size_t = non-negative integers
dx = Measure('dx', domain=mesh, subdomain_data=markers)
# Extract facets to apply boundary conditions for internal boundaries
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)

########################################################## Heat Transfer Solver #####################################################################
# Conductivity class to allow for different materials
class Conductivity(UserExpression):
    def __init__(self, markers, **kwargs):
        super(Conductivity, self).__init__(**kwargs)
        self.markers = markers
    def eval_cell(self, values, x, cell):
        if self.markers[cell.index] == 0:
            values[0] = 11.4 # copper
        else:
            values[0] = 1      # aluminum
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
t, T = 0., 5.; # Start and end time  PERFORM STUDY ON THIS AS WELL, VARY END TIME

# Boundray Conditions           
tol = 1E-14
for f in facets(mesh):
    domains = []
    for c in cells(f):
        domains.append(markers[c])
    domains = list(set(domains))
    # Domains with [0,i] refer to each boundary within the "copper" plate
    # Domains 1-9 refer to the domains of the batteries
    if domains == [1]:
      boundaries[f] = 1
    elif domains == [2]:
      boundaries[f] = 2
    elif domains == [3]:
      boundaries[f] = 3
    elif domains == [4]:
      boundaries[f] = 4
    elif domains == [5]:
      boundaries[f] = 5
    elif domains == [6]:
      boundaries[f] = 6
    elif domains == [7]:
      boundaries[f] = 7
    elif domains == [8]:
      boundaries[f] = 8
    elif domains == [9]:
      boundaries[f] = 9
    else:
      boundaries[f] = 0

u1= Constant(298.0)
ue = Constant(325)
#ue.t = k/10.;
u0 = u1;
# def boundary(x, on_boundary):
#     return on_boundary and near(x[0], 0, tol)
q = Constant("1000") # Neumann Heat Flux

R = Constant("0.0003")    # Contact resistance

# Inititalize time stepping
stepcounter = 0; 
timer0 = time.time()

# Define new measures associated with the interior boundaries
dS = Measure('dS', domain=mesh, subdomain_data=boundaries)
vol = assemble(Constant('1.0')*dx(1))
# Time-stepping loop
start_time = time.time()
while t < T: 
    # Time scheme
    um = theta*u + (1.0-theta)*u0 
    T_1 = assemble(u*dx(1))/vol         # Compute area-average T over cell 1

    if stepcounter == 0:
      T_cr = u1
    else:
      T_cr = T_1 + R*q
    bcs = DirichletBC(V, T_cr, boundaries,1)

    # Weak form of the heat equation in residual form
    r = (u - u0)/k*v*dx + k_coeff*inner(grad(um), grad(v))*dx - q*v*dx(1)       # With Neumann BCs -f*v('+')*dS(1)

    # Solve the Heat equation (one timestep)
    solve(r==0, u, bcs)      # Add bcs to this if using Dirichlet BCs
    # Shift to next timestep
    t += k; u0 = project(u, V); 
    ue.t = t;
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
V = VectorFunctionSpace(mesh, 'P', 1);

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
V = FunctionSpace(mesh, 'P', 1)
von_Mises = project(von_Mises, V)
sfile << von_Mises

print("--- %s seconds ---" % (time.time() - start_time))
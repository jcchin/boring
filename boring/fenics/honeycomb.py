"""
High Fidelity Honeycomb Model

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
# Define domain and mesh
cell_d = 1.8; # diameter of the cell
extra = 1.5;  # extra space along the diagonal
n_cells = 4;  # number of cells
width = cell_d * extra;
length = width * (n_cells-1)  # square side length

G = [0, width, 0, length];
mresolution = 100; # number of cells

xstep1 = width/2*3**0.5
xstep2 = width*3**0.5
ystep = width/2

# Define 2D geometry
domain1a = Rectangle(Point(G[0], G[2]), Point(G[1], G[3])) # column 1
arc1a = Circle(Point(width/2,0),width/2)
arc2a = Circle(Point(width/2,length),width/2)
domain1a = domain1a + arc1a + arc2a

domain1b = Rectangle(Point(G[0]+xstep2, G[2]), Point(G[1]+xstep2, G[3])) # column 3
arc1b = Circle(Point(width/2+xstep2,0),width/2)
arc2b = Circle(Point(width/2+xstep2,length),width/2)
domain1b = domain1b + arc1b + arc2b

domain2a = Rectangle(Point(G[0]+xstep1, G[2]+ystep), Point(G[1]+xstep1, G[3]+ystep)) # column 2
arc3a = Circle(Point(width/2+xstep1,ystep),width/2)
arc4a = Circle(Point(width/2+xstep1,length+ystep),width/2)
domain2a = domain2a + arc3a + arc4a

domain2b = Rectangle(Point(G[0]+xstep1+xstep2, G[2]+ystep), Point(G[1]+xstep1+xstep2, G[3]+ystep)) # column 4
arc3b = Circle(Point(width/2+xstep1+xstep2,ystep),width/2)
arc4b = Circle(Point(width/2+xstep1+xstep2,length+ystep),width/2)
domain2b = domain2b + arc3b + arc4b

domain = domain1a + domain1b + domain2a + domain2b

xoffset = length/(n_cells-1)

batts = [Circle(Point(width/2+xstep2*j,xoffset*i), cell_d/2) for j in range(int(n_cells/2)) for i in range(n_cells)] # first and third column
batts2 = [Circle(Point(width/2+xstep1+xstep2*j,ystep+xoffset*i), cell_d/2) for j in range(int(n_cells/2)) for i in range(n_cells)] # second and fourth column (shifted over and up)

for (i, batt) in enumerate(batts):
  domain.set_subdomain(1+i, batt) # add cells to domain as a sub-domain
for (i, batt) in enumerate(batts2):
  domain.set_subdomain(1+i+len(batts), batt) # add shifted cells to domain as additional sub-domains

mesh = generate_mesh(domain, mresolution)


markers = MeshFunction('size_t', mesh, 2, mesh.domains())  # size_t = non-negative integers
dx = Measure('dx', domain=mesh, subdomain_data=markers)

plot(markers,"Subdomains")

plt.show()
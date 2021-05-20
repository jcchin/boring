import numpy as np  
from dolfin import *
from mshr import * 
import matplotlib.pyplot as plt


def make_mesh(extra,ratio,wfile=False):  
    cell_d = 0.018#i['cell_d']
    n_cells = 4#int(i['n_cells'])  # number of cells
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
    mesh1 = generate_mesh(domain, mresolution)
    # Mark the sub-domains
    markers = MeshFunction('size_t', mesh1, 2, mesh1.domains())  # size_t = non-negative integers
    # Extract facets to apply boundary conditions for internal boundaries
    boundaries = MeshFunction('size_t', mesh1, mesh1.topology().dim()-1)
    if wfile:
        File("meshes/mesh_e{:.2f}_r{:.2f}.xml".format(extra,ratio)) << mesh1
        File("meshes/mmf_e{:.2f}_r{:.2f}.xml".format(extra,ratio)) << markers
        File("meshes/bmf_e{:.2f}_r{:.2f}.xml".format(extra,ratio)) << boundaries

    return mesh1,markers,boundaries

def read_mesh(extra,ratio):
    parameters['ghost_mode'] = 'shared_vertex'#'shared_facet'
    extra = float(extra)
    ratio = float(ratio)

    mesh1 = Mesh("meshes/mesh_e{:.2f}_r{:.2f}.xml".format(extra,ratio))
    markers = MeshFunction('size_t', mesh1, 2, mesh1.domains())  # size_t = non-negative integers
    boundaries =  MeshFunction('size_t', mesh1, mesh1.topology().dim()-1)
    #markers = MeshFunction('size_t',mesh1,"meshes/mmf_e{:.2f}_r{:.2f}.xml".format(extra,ratio))
    #boundaries = MeshFunction('size_t',mesh1,"meshes/bmf_e{:.2f}_r{:.2f}.xml".format(extra,ratio))

    return mesh1,markers,boundaries


if __name__ == '__main__':

    # mesh1,markers,boundaries=read_mesh(1.10,0.05)
    # plot(markers,"Subdomains")
    # plt.show()
    # quit()

    exs = np.linspace(1.05,1.5,3)
    rts =  np.linspace(0.05,0.95,3)

    for i in range(len(exs)):
        for j in range(len(rts)):
            print("{:.2f},{:.2f}".format(exs[i],rts[j]))
            make_mesh(exs[i],rts[j],wfile=True)
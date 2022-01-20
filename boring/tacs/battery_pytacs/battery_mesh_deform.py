import os
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

from tacs import functions, constitutive, elements, TACS, pyTACS

def update_points(Xpts0, indices, Xpts_cp, delta):
    # this function warps points using the displacements from curve projections
    # Xpts0: the original surface point coordinates
    # indices: indices of the independent nodes
    # Xpts_cp: original control point coordinates
    # delta: displacements of the control points

    Xnew = np.zeros(np.shape(Xpts0))
    Xnew[:, :] = Xpts0[:, :]


    for i in indices:

        # point coordinates with the baseline design
        # this is the point we will warp
        xpts_i = Xpts0[i]

        # the vectorized point-based warping we had from older versions.
        rr = xpts_i - Xpts_cp
        LdefoDist = (rr[:, 0]**2 + rr[:, 1]**2 + 1e-16)**-0.5
        LdefoDist3 = LdefoDist**1
        Wi = LdefoDist3
        den = np.sum(Wi)
        interp = np.zeros(2)
        for iDim in range(2):
            interp[iDim] = np.sum(Wi*delta[:, iDim])/den

        # finally, update the coord in place
        Xnew[i] = Xnew[i] + interp

    return Xnew


def get_battery_edge_nodes(Xpts, m=3, n=3, cell_d=0.018, extra=1.1516, ratio=0.7381, eps=1e-6):
    """
    Get the indexes of battery edges nodes

    Inputs:
    Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

    Outputs:
    battery_edge_idx: indexes in the Xpts array that are the battery edges
        - sorted in nested array of length 9 for each battery
    """

    w = cell_d*m*extra
    l = cell_d*n*extra

    battery_edge_idx = []
    for i in range(m):
            for j in range(n):
                x0 = (i + 0.5)*w/m
                y0 = (j + 0.5)*l/n
                battery_ij_edge_idx = []
                for k in range(len(Xpts)):
                    dist = ((Xpts[k, 0] - x0)**2 + (Xpts[k, 1] - y0)**2)**0.5
                    if np.absolute(dist - cell_d/2.0) <= eps:
                        battery_ij_edge_idx.append(k)
                battery_edge_idx.append(battery_ij_edge_idx)

    return battery_edge_idx

def get_hole_edge_nodes(Xpts, m=3, n=3, extra=1.1516, ratio=0.7381, eps=1e-6):
    """
    Get the indexes of battery edges nodes

    Inputs:
    Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

    Outputs:
    hole_edge_idx: indexes in the Xpts array that are the hole edges
        - sorted in nested array of length 9 for each battery
    """

    w = cell_d*m*extra
    l = cell_d*n*extra
    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

    hole_edge_idx = []
    for i in range(1, m):
        for j in range(1, n):
            x0 = i*w/m
            y0 = j*l/n
            hole_uv_edge_idx = []
            for k in range(len(Xpts)):
                dist = ((Xpts[k, 0] - x0)**2 + (Xpts[k, 1] - y0)**2)**0.5
                if np.absolute(dist - hole_r) <= eps:
                    hole_uv_edge_idx.append(k)
            hole_edge_idx.append(hole_uv_edge_idx)

    return hole_edge_idx

def get_border_nodes(Xpts, m=3, n=3, extra=1.1516, eps=1e-6):
    """
    Get the indexes of battery edges nodes

    Inputs:
    Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

    Outputs:
    border_idx: indexes in the Xpts array that are the pack edges
    """

    w = cell_d*m*extra
    l = cell_d*n*extra

    border_idx = []
    for i in range(len(Xpts)):

        if np.absolute(Xpts[i, 0]) <= eps:
            border_idx.append(i)
        elif np.absolute(Xpts[i, 1]) <= eps:
            border_idx.append(i)
        elif np.absolute(Xpts[i, 0] - w) <= eps:
            border_idx.append(i)
        elif np.absolute(Xpts[i, 1] - l) <= eps:
            border_idx.append(i)

    return border_idx

def get_edge_control_points(Xpts, m=3, n=3, cell_d=0.018, extra=1.1516, ratio=0.7381, eps=1e-6):
    """
    Get the indexes of battery edges control point nodes

    Note: this function negates the m,n option - assumes 3x3 square grid

    Inputs:
    Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

    Outputs:
    border_idx: indexes in the Xpts array that are the pack edge control points
    """

    w = cell_d*m*extra
    l = cell_d*n*extra
    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

    edge_cp_idx = []  # store a nested list of length 4: [[bottom edge cp nodes], [right edge ""], [top edge ""], [left edge ""]] 
    edge_uv = [[[0, 0], [1, 0], [2, 0], [3, 0]], 
               [[3, 0], [3, 1], [3, 2], [3, 3]], 
               [[3, 3], [2, 3], [1, 3], [0, 3]], 
               [[0, 3], [0, 2], [0, 1], [0, 0]]]  # u,v index of holes on each edge (bottom, right, top, left)
    pt_offsets = np.array([1, -1, 1, -1, 1, -1])
    for i in range(4):
        i_edge_cp_idx = []
        if i%2 == 0:
            dp = np.array([1, 0])  # move point in x-direction
        else:
            dp = np.array([0, 1])  # move point in y-direction
        for j in range(4):
            [u, v] = edge_uv[i][j]
            x = u*w/m + hole_r*pt_offsets*dp[0]  # array of x-points to find on this edge
            y = v*l/n + hole_r*pt_offsets*dp[1]  # array of y-points to find on this edge
            for k in range(len(Xpts)):
                d = ((x - Xpts[k, 0])**2 + (y - Xpts[k, 1])**2)**0.5
                if np.any(d < eps):
                    i_edge_cp_idx.append(k)
        edge_cp_idx.append(i_edge_cp_idx)

    return edge_cp_idx

def get_hole_deltas(Xpts0, hole_idx, dratio, m=3, n=3, extra=1.1516, ratio=0.7381):

    w = cell_d*m*extra
    l = cell_d*n*extra
    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

    for i in range(1, m):
        for j in range(1, n):
            x0 = i*w/m
            y0 = j*l/n
            for k in hole_idx[i][j]

    return hole_deltas

comm = MPI.COMM_WORLD

# Instantiate FEASolver
structOptions = {
    # Specify what type of elements we want in the f5
    'writeSolution': False,
    'outputElement': TACS.PLANE_STRESS_ELEMENT,
}

bdfFile = os.path.join(os.path.dirname(__file__), 'boring_pytacs.bdf')
FEAAssembler = pyTACS(bdfFile, comm, options=structOptions)

# Plate geometry
tplate = 0.065  # 1 cm

# Material properties
battery_rho = 1460.0  # density kg/m^3
battery_kappa = 1.3 # Thermal conductivity W/(m⋅K)
battery_cp = 880.0 # Specific heat J/(kg⋅K)

alum_rho = 2700.0  # density kg/m^3
alum_kappa = 204.0 # Thermal conductivity W/(m⋅K)
alum_cp = 883.0 # Specific heat J/(kg⋅K)

def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):

    # Setup property and constitutive objects
    if compDescript == 'Block':
        prop = constitutive.MaterialProperties(rho=alum_rho, kappa=alum_kappa, specific_heat=alum_cp)
    else:  # battery
        prop = constitutive.MaterialProperties(rho=battery_rho, kappa=battery_kappa, specific_heat=battery_cp)
    
    # Set one thickness dv for every component
    con = constitutive.PlaneStressConstitutive(prop, t=tplate, tNum=-1)

    # For each element type in this component,
    # pass back the appropriate tacs element object
    elemList = []
    model = elements.HeatConduction2D(con)
    for elemDescript in elemDescripts:
        if elemDescript in ['CQUAD4', 'CQUADR']:
            basis = elements.LinearQuadBasis()
        elif elemDescript in ['CTRIA3', 'CTRIAR']:
            basis = elements.LinearTriangleBasis()
        else:
            print("Uh oh, '%s' not recognized" % (elemDescript))
        elem = elements.Element2D(model, basis)
        elemList.append(elem)

    return elemList

# Set up constitutive objects and elements
FEAAssembler.initialize(elemCallBack)

# Get the mesh points
Xpts0 = FEAAssembler.getOrigNodes()

# Drop the z-values and reshape the vector
Xpts0 = np.delete(Xpts0, np.arange(2, Xpts0.size, 3))
Xpts0 = Xpts0.reshape((int(Xpts0.size/2), 2))

# Parametrize the geometry
m = 3  # number of rows of battery cells
n = 3  # number of columns of battery cells
cell_d = 0.018  # diameter of the cell
extra = 1.1516  # extra space along the diagonal, in [1, infty)
ratio = 0.7381  # cell diameter/cutout diamter, in [0, 1]

w = cell_d*m*extra # total width
l = cell_d*n*extra # total length
hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

# Get the control point node indexes
battery_edge_idx = get_battery_edge_nodes(Xpts0)
hole_edge_idx = get_hole_edge_nodes(Xpts0)
border_idx = get_border_nodes(Xpts0)
edge_cp_idx = get_edge_control_points(Xpts0)

# ------- Try deforming only the hole radius ---------

# Get independent node indices
indep_idx = []
for i in range(len(Xpts0)):
    if i not in hole_edge_idx:
        indep_idx.append(i)



# Plot the initial nodes
# fig = plt.figure()
# ax = plt.subplot(1,1,1, aspect=1)
# ax.scatter(Xpts0[:, 0], Xpts0[:, 1], s=2, color="tab:blue")
# for i in range(len(battery_edge_idx)):
#     ax.scatter(Xpts0[battery_edge_idx[i], 0], Xpts0[battery_edge_idx[i], 1], s=2, color="tab:red")

# for i in range(len(hole_edge_idx)):
#     ax.scatter(Xpts0[hole_edge_idx[i], 0], Xpts0[hole_edge_idx[i], 1], s=2, color="tab:green")

# colors = ["C1", "black", "C3", "C4"]
# for i in range(len(edge_cp_idx)):
#     ax.scatter(Xpts0[edge_cp_idx[i], 0], Xpts0[edge_cp_idx[i], 1], s=2, color=colors[i])

# plt.show()
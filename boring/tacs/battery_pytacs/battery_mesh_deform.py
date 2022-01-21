import os
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

from tacs import functions, constitutive, elements, TACS, pyTACS

def update_points(Xpts0, indices, Xpts_cp, delta):
    # this function warps points using the displacements from curve projections
    # Xpts0: the original surface point coordinates
    # indices: indices of the dependent nodes
    # Xpts_cp: original control point coordinates
    # delta: displacements of the control points

    Xnew = np.zeros(np.shape(Xpts0))
    Xnew[:, :] = Xpts0[:, :]
    max_update = 0.0

    for i in indices:

        # point coordinates with the baseline design
        # this is the point we will warp
        xpts_i = Xpts0[i]

        # the vectorized point-based warping we had from older versions.
        rr = xpts_i - Xpts_cp
        LdefoDist = (rr[:,0]**2 + rr[:,1]**2+1e-16)**-0.5
        LdefoDist3 = LdefoDist**3
        Wi = LdefoDist3
        den = np.sum(Wi)
        interp = np.zeros(2)
        for iDim in range(2):
            interp[iDim] = np.sum(Wi*delta[:, iDim])/den
        dx = ((interp[0])**2 + (interp[1])**2)**0.5
        if dx > max_update:
            max_update = dx

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
            for k in range(len(Xpts)):
                dist = ((Xpts[k, 0] - x0)**2 + (Xpts[k, 1] - y0)**2)**0.5
                if np.absolute(dist - cell_d/2.0) <= eps:
                    battery_edge_idx.append(k)

    return battery_edge_idx

def get_battery_internal_nodes(Xpts, m=3, n=3, cell_d=0.018, extra=1.1516, ratio=0.7381, eps=1e-6):
    """
    Get the indexes of internal battery nodes

    Inputs:
    Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

    Outputs:
    battery_internal_idx: indexes in the Xpts array that are the battery edges
        - sorted in nested array of length 9 for each battery
    """

    w = cell_d*m*extra
    l = cell_d*n*extra
    battery_internal_idx = []
    for i in range(m):
        for j in range(n):
            x0 = (i + 0.5)*w/m
            y0 = (j + 0.5)*l/n
            for k in range(len(Xpts)):
                dist = ((Xpts[k, 0] - x0)**2 + (Xpts[k, 1] - y0)**2)**0.5
                if dist < cell_d/2.0 - eps:
                    battery_internal_idx.append(k)

    return battery_internal_idx

def get_battery_nodes(Xpts, m=3, n=3, cell_d=0.018, extra=1.1516, ratio=0.7381, eps=1e-6):
    """
    Get the indexes of internal battery nodes

    Inputs:
    Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

    Outputs:
    battery_internal_idx: indexes in the Xpts array that are the battery edges
        - sorted in nested array of length 9 for each battery
    """

    w = cell_d*m*extra
    l = cell_d*n*extra
    battery_internal_idx = []
    for i in range(m):
        for j in range(n):
            x0 = (i + 0.5)*w/m
            y0 = (j + 0.5)*l/n
            for k in range(len(Xpts)):
                dist = ((Xpts[k, 0] - x0)**2 + (Xpts[k, 1] - y0)**2)**0.5
                if dist <= cell_d/2.0 + eps:
                    battery_internal_idx.append(k)

    return battery_internal_idx

def get_hole_edge_nodes(Xpts, m=3, n=3, extra=1.1516, ratio=0.7381, eps=1e-6):
    """
    Get the indexes of battery edges nodes

    Inputs:
    Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

    Outputs:
    hole_edge_idx: indexes in the Xpts array that are the hole edges
    """

    w = cell_d*m*extra
    l = cell_d*n*extra
    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

    hole_edge_idx = []
    for i in range(0, m+1):
        for j in range(0, n+1):
            x0 = i*w/m
            y0 = j*l/n
            for k in range(len(Xpts)):
                dist = ((Xpts[k, 0] - x0)**2 + (Xpts[k, 1] - y0)**2)**0.5
                if np.absolute(dist - hole_r) <= eps:
                    hole_edge_idx.append(k)

    return hole_edge_idx

def get_border_hole_nodes(Xpts, m=3, n=3, extra=1.1516, ratio=0.7381, eps=1e-6):
    """
    Get the indexes of border hole edges nodes

    Inputs:
    Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

    Outputs:
    border_hole_idx: indexes in the Xpts array that are the hole edges
    """

    w = cell_d*m*extra
    l = cell_d*n*extra
    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

    border_uv = [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2], 
                 [3, 3], [2, 3], [1, 3], [0, 3], [0, 2], [0, 1]]
    border_hole_idx = []
    for i, j in border_uv:
        x0 = i*w/m
        y0 = j*l/n
        for k in range(len(Xpts)):
            dist = ((Xpts[k, 0] - x0)**2 + (Xpts[k, 1] - y0)**2)**0.5
            if np.absolute(dist - hole_r) <= eps:
                border_hole_idx.append(k)

    return border_hole_idx

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

def get_hole_deltas(Xpts0, hole_idx, dratio, dextra, m=3, n=3, extra=1.1516, ratio=0.7381):

    w = cell_d*m*extra
    l = cell_d*n*extra
    dw = cell_d*m*dextra
    dl = cell_d*n*dextra

    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
    dhole_r = dratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

    x_holes = np.repeat(np.linspace(0.0, w, m+1), 4)
    y_holes = np.tile(np.linspace(0.0, l, n+1), 4)
    dx_holes = np.repeat(np.linspace(0.0, dw, m+1), 4)
    dy_holes = np.tile(np.linspace(0.0, dl, n+1), 4)

    hole_deltas = np.zeros((len(hole_idx), 2))
    for i, idx in enumerate(hole_idx):
        pt = Xpts0[idx]

        # Find the center of the hole that this point belongs to
        dist = ((pt[0] - x_holes)**2 + (pt[1] - y_holes)**2)**0.5 - hole_r
        which_hole = np.argmin(dist)
        x0 = x_holes[which_hole]
        y0 = y_holes[which_hole]

        # Get the angle of the point wrt the hole center
        theta = np.arctan2(pt[1]-y0, pt[0]-x0)

        # Compute the delta for this point
        hole_deltas[i, 0] = dhole_r*np.cos(theta) + dx_holes[which_hole]
        hole_deltas[i, 1] = dhole_r*np.sin(theta) + dy_holes[which_hole]

    return hole_deltas

def get_battery_deltas(Xpts0, battery_idx, dextra, m=3, n=3, extra=1.1516, ratio=0.7381):

    w = cell_d*m*extra
    l = cell_d*n*extra
    dw = cell_d*m*dextra
    dl = cell_d*n*dextra

    xb = np.repeat(np.linspace(0.5*w/m, 2.5*w/m, m), 3)
    yb = np.tile(np.linspace(0.5*l/n, 2.5*l/n, n), 3)
    dxb = np.repeat(np.linspace(0.5*dw/m, 2.5*dw/m, m), 3)
    dyb = np.tile(np.linspace(0.5*dl/n, 2.5*dl/m, n), 3)

    battery_deltas = np.zeros((len(battery_idx), 2))
    for i, idx in enumerate(battery_idx):
        pt = Xpts0[idx]

        # Find the center of the hole that this point belongs to
        dist = ((pt[0] - xb)**2 + (pt[1] - yb)**2)**0.5 - 0.5*cell_d
        which_battery = np.argmin(dist)

        # Compute the delta for this point
        battery_deltas[i, 0] = dxb[which_battery]
        battery_deltas[i, 1] = dyb[which_battery]

    return battery_deltas

def get_border_deltas(Xpts0, border_idx, dratio, dextra, m=3, n=3, extra=1.1516, ratio=0.7381, eps=1e-6):

    w = cell_d*m*extra
    l = cell_d*n*extra
    dw = cell_d*m*dextra
    dl = cell_d*n*dextra

    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
    dhole_r = dratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

    x_border_len = (w-2*m*hole_r)/m
    y_border_len = (l-2*n*hole_r)/n
    dx_border_len = (dw-2*m*dhole_r)/m
    dy_border_len = (dl-2*n*dhole_r)/n

    x_ranges = [hole_r+dhole_r, hole_r+dhole_r + (x_border_len+dx_border_len), 
                3*(hole_r+dhole_r) + (x_border_len+dx_border_len), 3*(hole_r+dhole_r) + 2*(x_border_len+dx_border_len), 
                5*(hole_r+dhole_r) + 2*(x_border_len+dx_border_len), 5*(hole_r+dhole_r) + 3*(x_border_len+dx_border_len)]
    y_ranges = [hole_r+dhole_r, hole_r+dhole_r + (y_border_len+dy_border_len), 
                3*(hole_r+dhole_r) + (y_border_len+dy_border_len), 3*(hole_r+dhole_r) + 2*(y_border_len+dy_border_len), 
                5*(hole_r+dhole_r) + 2*(y_border_len+dy_border_len), 5*(hole_r+dhole_r) + 3*(y_border_len+dy_border_len)]

    edge_cp_idx = get_edge_control_points(Xpts0)
    x_cp = Xpts0[edge_cp_idx[:], 0]
    y_cp = Xpts0[edge_cp_idx[:], 1]
    border_deltas = np.zeros((len(border_idx), 2))
    for i, idx in enumerate(border_idx):
        pt = Xpts0[idx]
        if np.absolute(pt[0]) < eps:  # left edge
            # Check if this is a control point
            if np.any(np.absolute(pt[1] - y_cp[3]) < eps):
                cp_idx = np.argmin(np.absolute(pt[1] - y_cp[3]))
                border_deltas[i, 1] = y_ranges[cp_idx] - pt[1]
            else:  
                # Get control points this node is between
                y1 = y_cp[3][np.argmax(y_cp[3] >= pt[1])-1]
                y2 = y_cp[3][np.argmax(y_cp[3] >= pt[1])]
                ynew1 = y_ranges[np.argmax(y_cp[3] >= pt[1])-1]
                ynew2 = y_ranges[np.argmax(y_cp[3] >= pt[1])]
                border_deltas[i, 1] = ynew1 + (pt[1] - y1)*(ynew2 - ynew1)/(y2 - y1) - pt[1]

        elif np.absolute(pt[1]) < eps:  # bottom edge
            # Check if this is a control point
            if np.any(np.absolute(pt[0] - x_cp[0]) < eps):
                cp_idx = np.argmin(np.absolute(pt[0] - x_cp[0]))
                border_deltas[i, 0] = x_ranges[cp_idx] - pt[0]
            else:  
                # Get control points this node is between
                x1 = x_cp[0][np.argmax(x_cp[0] >= pt[0])-1]
                x2 = x_cp[0][np.argmax(x_cp[0] >= pt[0])]
                xnew1 = y_ranges[np.argmax(x_cp[0] >= pt[0])-1]
                xnew2 = y_ranges[np.argmax(x_cp[0] >= pt[0])]
                border_deltas[i, 0] = xnew1 + (pt[0] - x1)*(xnew2 - xnew1)/(x2 - x1) - pt[0]
            
        elif np.absolute(pt[0] - w) < eps:  # right edge
            # Check if this is a control point
            if np.any(np.absolute(pt[1] - y_cp[1]) < eps):
                cp_idx = np.argmin(np.absolute(pt[1] - y_cp[1]))
                border_deltas[i, 1] = y_ranges[cp_idx] - pt[1]
            else:  
                # Get control points this node is between
                y1 = y_cp[1][np.argmax(y_cp[1] >= pt[1])-1]
                y2 = y_cp[1][np.argmax(y_cp[1] >= pt[1])]
                ynew1 = y_ranges[np.argmax(y_cp[1] >= pt[1])-1]
                ynew2 = y_ranges[np.argmax(y_cp[1] >= pt[1])]
                border_deltas[i, 1] = ynew1 + (pt[1] - y1)*(ynew2 - ynew1)/(y2 - y1) - pt[1]
            border_deltas[i, 0] = dw  # shift all right border nodes by dw

        elif np.absolute(pt[1] - l) < eps:  # top edge
            # Check if this is a control point
            if np.any(np.absolute(pt[0] - x_cp[2]) < eps):
                cp_idx = np.argmin(np.absolute(pt[0] - x_cp[2]))
                border_deltas[i, 0] = x_ranges[cp_idx] - pt[0]
            else:  
                # Get control points this node is between
                x1 = x_cp[2][np.argmax(x_cp[2] >= pt[0])-1]
                x2 = x_cp[2][np.argmax(x_cp[2] >= pt[0])]
                xnew1 = y_ranges[np.argmax(x_cp[2] >= pt[0])-1]
                xnew2 = y_ranges[np.argmax(x_cp[2] >= pt[0])]
                border_deltas[i, 0] = xnew1 + (pt[0] - x1)*(xnew2 - xnew1)/(x2 - x1) - pt[0]
            border_deltas[i, 1] = dl  # shift all top border nodes by dl

    return border_deltas

comm = MPI.COMM_WORLD

# Instantiate FEASolver
bdfFile = os.path.join(os.path.dirname(__file__), 'boring_pytacs.bdf')
FEAAssembler = pyTACS(bdfFile, comm)

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
# battery_edge_idx = get_battery_edge_nodes(Xpts0)
# battery_internal_idx = get_battery_internal_nodes(Xpts0)
battery_idx = get_battery_nodes(Xpts0)
hole_edge_idx = get_hole_edge_nodes(Xpts0)
border_idx = get_border_nodes(Xpts0)
#border_hole_idx = get_border_hole_nodes(Xpts0)
edge_cp_idx = get_edge_control_points(Xpts0)

# Get dependent node indices
dep_idx = []
for i in range(len(Xpts0)):
    if (i not in hole_edge_idx) and (i not in border_idx) and (i not in battery_idx):
        dep_idx.append(i)

# Get the node locations of the control points
Xpts0_cp_idx = hole_edge_idx + border_idx + battery_idx
Xpts0_cp = Xpts0[Xpts0_cp_idx[:]]

# Define the change in geometric parameters to be applied
dratio = 0.2
dextra = 0.2

# Compute the delta of the hole points
hole_deltas = get_hole_deltas(Xpts0, hole_edge_idx, dratio, dextra)
new_hole_pts = Xpts0[hole_edge_idx[:], :] + hole_deltas[:, :]

# Get the delta of the battery nodes
battery_deltas = get_battery_deltas(Xpts0, battery_idx, dextra)
new_battery_pts = Xpts0[battery_idx[:], :] + battery_deltas[:, :]

# Get the delta of the border edges
border_deltas = get_border_deltas(Xpts0, border_idx, dratio, dextra)
new_border_pts = Xpts0[border_idx[:], :] + border_deltas[:, :]

# Set the total delta array
delta = np.zeros((len(Xpts0_cp_idx), 2))
delta[0:len(hole_edge_idx), :] = hole_deltas[:, :]
delta[len(hole_edge_idx):len(hole_edge_idx)+len(border_idx), :] = border_deltas[:, :]
delta[len(hole_edge_idx)+len(border_idx):len(hole_edge_idx)+len(border_idx)+len(battery_idx), :] = battery_deltas[:, :]

# Get the updated locations of the dependent nodes
Xnew = update_points(Xpts0, dep_idx, Xpts0_cp, delta)

# Update the independent node locations
Xnew[Xpts0_cp_idx[:], :] += delta[:, :]

# Plot the initial and deformed meshes nodes
fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, constrained_layout=True)
ax0.scatter(Xpts0[:, 0], Xpts0[:, 1], s=2, color="tab:blue")
# ax0.scatter(new_border_pts[:, 0], new_border_pts[:, 1], s=2, color="tab:red")
# ax0.scatter(new_hole_pts[:, 0], new_hole_pts[:, 1], s=2, color="tab:red")
# ax0.scatter(new_battery_pts[:, 0], new_battery_pts[:, 1], s=2, color="tab:red")
ax1.scatter(Xnew[:, 0], Xnew[:, 1], s=2, color="tab:blue")

ax0.set_aspect("equal")
ax1.set_aspect("equal")
#ax0.axis("off")
#ax1.axis("off")
ax0.set_title("Original node locations")
ax1.set_title(r"Deformed mesh: $\Delta$ratio = {0}".format(dratio))

plt.show()
# plt.savefig(f"deformed_points_dratio_{dratio}.pdf")
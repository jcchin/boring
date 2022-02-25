import os
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

import openmdao.api as om
from tacs import functions, constitutive, elements, TACS, pyTACS

# TODO:
# - add MeshDeformation class to OM class
class MeshDeformation:
    def __init__(self, Xpts0, m=3, n=3, cell_d=0.018, extra=1.5, ratio=0.4, dextra=0.0, dratio=0.0):

        self.Xpts0 = Xpts0
        self.m = m
        self.n = n
        self.cell_d = cell_d
        self.extra = extra
        self.ratio = ratio
        self.dextra = dextra
        self.dratio = dratio
        self.eps = 1e-6
        
        # Evaluate the initial geometry objects
        self.eval_geometry()

        # Get the node indices of interest (only needs to be done once)
        self.get_battery_edge_nodes()
        self.get_battery_nodes()
        self.get_hole_nodes()
        self.get_border_nodes()
        self.get_edge_control_points()

        # Group the control point indices together
        self.Xpts0_cp_idx = self.hole_idx + self.battery_edge_idx + self.border_idx
        self.hole_start = 0
        self.hole_end = len(self.hole_idx)
        self.battery_start = len(self.hole_idx)
        self.battery_end = len(self.hole_idx) + len(self.battery_edge_idx)
        self.border_start = len(self.hole_idx) + len(self.battery_edge_idx)
        self.border_end = len(self.hole_idx) + len(self.battery_edge_idx) + len(self.border_idx)

        dep_idx = []
        for i in range(len(Xpts0)):
            if (i not in hole_idx) and (i not in battery_edge_idx) and (i not in border_idx):
                dep_idx.append(i)
        self.dep_idx = dep_idx

        return
    
    def eval_geometry(self):

        m = self.m
        n = self.n
        cell_d = self.cell_d
        extra = self.extra
        ratio = self.ratio
        dextra = self.dextra
        dextra = self.dratio

        self.w = cell_d*m*extra
        self.l = cell_d*n*extra
        self.dw = cell_d*m*dextra
        self.dl = cell_d*n*dextra

        self.xb = np.repeat(np.linspace(0.5*self.w/m, 2.5*self.w/m, m), 3)
        self.yb = np.tile(np.linspace(0.5*self.l/n, 2.5*self.l/n, n), 3).flatten()

        self.x_holes = np.repeat(np.linspace(0.0, self.w, m+1), 4)
        self.y_holes = np.tile(np.linspace(0.0, self.l, n+1), 4).flatten()

        self.dxb = np.repeat(np.linspace(0.5*self.dw/m, 2.5*self.dw/m, m), 3)
        self.dyb = np.tile(np.linspace(0.5*self.dl/n, 2.5*self.dl/n, n), 3).flatten(order="F")

        self.dx_holes = np.repeat(np.linspace(0.0, self.dw, m+1), 4)
        self.dy_holes = np.tile(np.linspace(0.0, self.dl, n+1), 4).flatten(order="F")

        self.hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
        self.dhole_r = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

        x_border_len = (self.w-2*m*self.hole_r)/m
        y_border_len = (self.l-2*n*self.hole_r)/n
        dx_border_len = cell_d*dextra-2*self.dhole_r
        dy_border_len = cell_d*dextra-2*self.dhole_r

        self.x_ranges = [self.hole_r+self.dhole_r, self.hole_r+self.dhole_r + (x_border_len+dx_border_len), 
                         3*(self.hole_r+self.dhole_r) + (x_border_len+dx_border_len), 3*(self.hole_r+self.dhole_r) + 2*(x_border_len+dx_border_len), 
                         5*(self.hole_r+self.dhole_r) + 2*(x_border_len+dx_border_len), 5*(self.hole_r+self.dhole_r) + 3*(x_border_len+dx_border_len)][:]
        self.y_ranges = [self.hole_r+self.dhole_r, self.hole_r+self.dhole_r + (y_border_len+dy_border_len), 
                         3*(self.hole_r+self.dhole_r) + (y_border_len+dy_border_len), 3*(self.hole_r+self.dhole_r) + 2*(y_border_len+dy_border_len), 
                         5*(self.hole_r+self.dhole_r) + 2*(y_border_len+dy_border_len), 5*(self.hole_r+self.dhole_r) + 3*(y_border_len+dy_border_len)][:]

        return

    def eval_geometry_partials(self):

        m = self.m
        n = self.n
        cell_d = self.cell_d
        extra = self.extra
        ratio = self.ratio
        dextra = self.dextra
        dextra = self.dratio

        self.ddw_ddextra = cell_d*m
        self.ddl_ddextra = cell_d*n

        self.ddxb_ddextra = np.repeat(np.linspace(0.5*self.ddw_ddextra/m, 2.5*self.ddw_ddextra/m, m), 3)
        self.ddyb_ddextra = np.tile(np.linspace(0.5*self.ddl_ddextra/n, 2.5*self.ddl_ddextra/n, n), 3).flatten(order="F")

        self.ddx_holes_ddextra = np.repeat(np.linspace(0.0, self.ddw_ddextra, m+1), 4)
        self.ddy_holes_ddextra = np.tile(np.linspace(0.0, self.ddl_ddextra, n+1), 4).flatten(order="F")

        self.ddr_ddextra = (np.log(2.0)/4.0)*(ratio+dratio)*cell_d*(2.0**0.5*(extra+dextra))
        self.ddr_ddratio = 0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0)

        ddx_ddextra = cell_d - 2.0*self.ddr_ddextra
        ddy_ddextra = cell_d - 2.0*self.ddr_ddextra
        ddx_ddratio = -2.0*self.ddr_ddratio
        ddy_ddratio = -2.0*self.ddr_ddratio

        self.dx_ranges_ddextra = [self.ddr_ddextra, self.ddr_ddextra + ddx_ddextra, 
                                  3.0*self.ddr_ddextra + ddx_ddextra, 3.0*self.ddr_ddextra + 2.0*ddx_ddextra, 
                                  5.0*self.ddr_ddextra + 2.0*ddx_ddextra, 5.0*self.ddr_ddextra + 3.0*ddx_ddextra][:]
        self.dy_ranges_ddextra = [self.ddr_ddextra, self.ddr_ddextra + ddy_ddextra, 
                                  3.0*self.ddr_ddextra + ddy_ddextra, 3.0*self.ddr_ddextra + 2.0*ddy_ddextra, 
                                  5.0*self.ddr_ddextra + 2.0*ddy_ddextra, 5.0*self.ddr_ddextra + 3.0*ddy_ddextra][:]
        self.dx_ranges_ddratio = [self.ddr_ddratio, self.ddr_ddratio + ddx_ddratio, 
                                  3.0*self.ddr_ddratio + ddx_ddratio, 3.0*self.ddr_ddratio + 2.0*ddx_ddratio, 
                                  5.0*self.ddr_ddratio + 2.0*ddx_ddratio, 5.0*self.ddr_ddratio + 3.0*ddx_ddratio][:]
        self.dy_ranges_ddratio = [self.ddr_ddratio, self.ddr_ddratio + ddy_ddratio, 
                                  3.0*self.ddr_ddratio + ddy_ddratio, 3.0*self.ddr_ddratio + 2.0*ddy_ddratio, 
                                  5.0*self.ddr_ddratio + 2.0*ddy_ddratio, 5.0*self.ddr_ddratio + 3.0*ddy_ddratio][:]

        return

    def get_battery_edge_nodes(self):
        """
        Get the indexes of battery edges nodes

        Inputs:
        Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

        Outputs:
        battery_edge_idx: indexes in the Xpts array that are the battery edges
            - sorted in nested array of length 9 for each battery
        """

        Xpts0 = self.Xpts0
        cell_d = self.cell_d
        eps = self.eps
        xb = self.xb
        yb = self.yb
        
        battery_edge_idx = []
        for i in range(len(Xpts0)):
            pt = Xpts0[i]

            # Find the center of the hole that this point belongs to
            dist = ((pt[0] - xb)**2 + (pt[1] - yb)**2)**0.5 - 0.5*cell_d
            if np.any(np.absolute(dist) < eps):
                battery_edge_idx.append(i)

        self.battery_edge_idx = battery_edge_idx

        return

    def get_battery_nodes(self):
        """
        Get the indexes of internal battery nodes

        Inputs:
        Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

        Outputs:
        battery_internal_idx: indexes in the Xpts array that are the battery edges
            - sorted in nested array of length 9 for each battery
        """

        Xpts0 = self.Xpts0
        cell_d = self.cell_d
        eps = self.eps
        xb = self.xb
        yb = self.yb

        battery_idx = []
        for i in range(len(Xpts0)):
            pt = Xpts0[i]

            # Find the center of the hole that this point belongs to
            dist = ((pt[0] - xb)**2 + (pt[1] - yb)**2)**0.5 - 0.5*cell_d
            if np.any(dist <= eps):
                battery_idx.append(i)

        self.battery_idx = battery_idx

        return

    def get_hole_nodes(self):
        """
        Get the indexes of battery edges nodes

        Inputs:
        Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

        Outputs:
        hole_idx: indexes in the Xpts array that are the hole edges
        """

        Xpts0 = self.Xpts0
        hole_r = self.hole_r
        eps = self.eps
        x_holes = self.x_holes
        y_holes = self.y_holes

        hole_idx = []
        for i in range(len(Xpts0)):
            pt = Xpts0[i]
            dist = ((pt[0] - x_holes)**2 + (pt[1] - y_holes)**2)**0.5 - hole_r
            if np.any(dist < eps):
                hole_idx.append(i)

        self.hole_idx = hole_idx

        return

    def get_border_nodes(self):
        """
        Get the indexes of battery edges nodes

        Inputs:
        Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

        Outputs:
        border_idx: indexes in the Xpts array that are the pack edges
        """

        Xpts0 = self.Xpts0
        eps = self.eps
        w = self.w
        l = self.l

        border_idx = []
        for i in range(len(Xpts0)):

            if np.absolute(Xpts0[i, 0]) <= eps:
                border_idx.append(i)
            elif np.absolute(Xpts0[i, 1]) <= eps:
                border_idx.append(i)
            elif np.absolute(Xpts0[i, 0] - w) <= eps:
                border_idx.append(i)
            elif np.absolute(Xpts0[i, 1] - l) <= eps:
                border_idx.append(i)

        self.border_idx = border_idx

        return

    def get_edge_control_points(self):
        """
        Get the indexes of battery edges control point nodes (end nodes of each straight section on border)

        Note: this function negates the m,n option - assumes 3x3 square grid

        Inputs:
        Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

        Outputs:
        border_idx: indexes in the Xpts array that are the pack edge control points
        """

        Xpts0 = self.Xpts0
        eps = self.eps
        w = self.w
        l = self.l
        hole_r = self.hole_r

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
                for k in range(len(Xpts0)):
                    d = ((x - Xpts0[k, 0])**2 + (y - Xpts0[k, 1])**2)**0.5
                    if np.any(d < eps):
                        i_edge_cp_idx.append(k)
            edge_cp_idx.append(i_edge_cp_idx)

        self.edge_cp_idx = edge_cp_idx

        return

    def get_hole_deltas(self):

        Xpts0 = self.Xpts0
        hole_r = self.hole_r
        dhole_r = self.dhole_r

        x_holes = self.x_holes
        y_holes = self.y_holes
        dx_holes = self.dx_holes
        dy_holes = self.dy_holes

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

        self.hole_deltas = hole_deltas

        return

    def get_battery_deltas(self):

        Xpts0 = self.Xpts0
        cell_d = self.cell_d
        xb = self.xb
        yb = self.yb
        dxb = self.dxb
        dyb = self.dyb

        battery_deltas = np.zeros((len(self.battery_edge_idx), 2))
        for i, idx in enumerate(self.battery_edge_idx):
            pt = Xpts0[idx]

            # Find the center of the hole that this point belongs to
            dist = ((pt[0] - xb)**2 + (pt[1] - yb)**2)**0.5 - 0.5*cell_d
            which_battery = np.argmin(dist)

            # Compute the delta for this point
            battery_deltas[i, 0] = dxb[which_battery]
            battery_deltas[i, 1] = dyb[which_battery]

        self.battery_deltas = battery_deltas

        return

    def get_border_deltas(self):

        Xpts0 = self.Xpts0
        eps = self.eps
        x_ranges = self.x_ranges
        y_ranges = self.y_ranges

        x_cp = np.sort(Xpts0[self.edge_cp_idx[:], 0])
        y_cp = np.sort(Xpts0[self.edge_cp_idx[:], 1])
        border_deltas = np.zeros((len(self.border_idx), 2))
        for i, idx in enumerate(self.border_idx):
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
                    xnew1 = x_ranges[np.argmax(x_cp[0] >= pt[0])-1]
                    xnew2 = x_ranges[np.argmax(x_cp[0] >= pt[0])]
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
                border_deltas[i, 0] = self.dw  # shift all right border nodes by dw

            elif np.absolute(pt[1] - l) < eps:  # top edge
                # Check if this is a control point
                if np.any(np.absolute(pt[0] - x_cp[2]) < eps):
                    cp_idx = np.argmin(np.absolute(pt[0] - x_cp[2]))
                    border_deltas[i, 0] = x_ranges[cp_idx] - pt[0]
                else:  
                    # Get control points this node is between
                    x1 = x_cp[2][np.argmax(x_cp[2] >= pt[0])-1]
                    x2 = x_cp[2][np.argmax(x_cp[2] >= pt[0])]
                    xnew1 = x_ranges[np.argmax(x_cp[2] >= pt[0])-1]
                    xnew2 = x_ranges[np.argmax(x_cp[2] >= pt[0])]
                    border_deltas[i, 0] = xnew1 + (pt[0] - x1)*(xnew2 - xnew1)/(x2 - x1) - pt[0]
                border_deltas[i, 1] = self.dl  # shift all top border nodes by dl

        self.border_deltas = border_deltas

        return

    def update_points(self):
        # this function warps points using the displacements from curve projections
        # Xpts0: the original surface point coordinates
        # indices: indices of the dependent nodes
        # Xpts_cp: original control point coordinates
        # delta: displacements of the control points

        Xpts_cp = self.Xpts0[self.Xpts0_cp_idx, :]
        Xnew = np.zeros(np.shape(self.Xpts0))
        Xnew[:, :] = self.Xpts0[:, :]

        for i in self.dep_idx:

            # point coordinates with the baseline design
            # this is the point we will warp
            xpts_i = self.Xpts0[i]

            # the vectorized point-based warping we had from older versions.
            rr = xpts_i - Xpts_cp
            LdefoDist = (rr[:,0]**2 + rr[:,1]**2+1e-16)**-0.5
            LdefoDist3 = LdefoDist**3
            Wi = LdefoDist3
            den = np.sum(Wi)
            interp = np.zeros(2)
            for iDim in range(2):
                interp[iDim] = np.sum(Wi*self.delta[:, iDim])/den

            # finally, update the coord in place
            Xnew[i] = Xnew[i] + interp

        return Xnew

    def deform_geometry(self):

        # Evaluate the initial geometry objects in case dratio/dextra changed
        self.eval_geometry()

        # Compute the deltas of the seed nodes
        self.get_hole_deltas()
        self.get_border_deltas()
        self.get_battery_deltas()

        delta = np.zeros((len(self.Xpts0_cp_idx), 2))
        delta[self.hole_start:self.hole_end, :] = self.hole_deltas[:, :]
        delta[self.battery_start:self.battery_end, :] = self.battery_deltas[:, :]
        delta[self.border_start:self.border_end, :] = self.border_deltas[:, :]
        self.delta = delta

        Xnew = self.update_points()
        Xnew[self.Xpts0_cp_idx[:], :] += delta[:, :]

        return Xnew

    def get_hole_delta_derivs(self):

        Xpts0 = self.Xpts0
        hole_r = self.hole_r

        hole_idx = self.hole_idx
        x_holes = self.x_holes
        y_holes = self.y_holes

        ddr_ddratio = self.ddr_ddratio
        ddr_ddextra = self.ddr_ddextra

        ddx_ddextra = self.ddx_holes_ddextra
        ddy_ddextra = self.ddy_holes_ddextra

        ddelta_ddratio = np.zeros((len(hole_idx), 2))
        ddelta_ddextra = np.zeros((len(hole_idx), 2))
        for i, idx in enumerate(hole_idx):
            pt = Xpts0[idx]

            # Find the center of the hole that this point belongs to
            dist = ((pt[0] - x_holes)**2 + (pt[1] - y_holes)**2)**0.5 - hole_r
            which_hole = np.argmin(dist)
            x0 = x_holes[which_hole]
            y0 = y_holes[which_hole]

            # Get the angle of the point wrt the hole center
            theta = np.arctan2(pt[1]-y0, pt[0]-x0)

            # Compute the delta  derivatives for this point
            ddelta_ddratio[i, 0] = np.cos(theta)*ddr_ddratio
            ddelta_ddratio[i, 1] = np.sin(theta)*ddr_ddratio

            ddelta_ddextra[i, 0] = np.cos(theta)*ddr_ddextra + ddx_ddextra[which_hole]
            ddelta_ddextra[i, 1] = np.sin(theta)*ddr_ddextra + ddy_ddextra[which_hole]

        return ddelta_ddratio, ddelta_ddextra

    def get_battery_delta_derivs(self):

        Xpts0 = self.Xpts0
        cell_d = self.cell_d
        xb = self.xb
        yb = self.yb

        ddxb_ddextra = self.ddxb_ddextra
        ddyb_ddextra = self.ddyb_ddextra

        ddelta_ddratio = np.zeros((len(battery_edge_idx), 2))
        ddelta_ddextra = np.zeros((len(battery_edge_idx), 2))
        for i, idx in enumerate(battery_edge_idx):
            pt = Xpts0[idx]

            # Find the center of the hole that this point belongs to
            dist = ((pt[0] - xb)**2 + (pt[1] - yb)**2)**0.5 - 0.5*cell_d
            which_battery = np.argmin(dist)

            # Compute the delta for this point
            ddelta_ddextra[i, 0] = ddxb_ddextra[which_battery]
            ddelta_ddextra[i, 1] = ddyb_ddextra[which_battery]

        return ddelta_ddratio, ddelta_ddextra

    def get_border_delta_derivs(self, eps=1e-6):

        Xpts0 = self.Xpts0
        w = self.w
        l = self.l
        
        dx_ranges_ddextra = self.dx_ranges_ddextra
        dy_ranges_ddextra = self.dy_ranges_ddextra
        dx_ranges_ddratio = self.dx_ranges_ddratio
        dy_ranges_ddratio = self.dy_ranges_ddratio

        edge_cp_idx = self.edge_cp_idx
        x_cp = np.sort(Xpts0[edge_cp_idx[:], 0])
        y_cp = np.sort(Xpts0[edge_cp_idx[:], 1])

        ddelta_ddratio = np.zeros((len(border_idx), 2))
        ddelta_ddextra = np.zeros((len(border_idx), 2))
        for i, idx in enumerate(border_idx):
            pt = Xpts0[idx]
            if np.absolute(pt[0]) < eps:  # left edge
                # Check if this is a control point
                if np.any(np.absolute(pt[1] - y_cp[3]) < eps):
                    cp_idx = np.argmin(np.absolute(pt[1] - y_cp[3]))
                    ddelta_ddratio[i, 1] = dy_ranges_ddratio[cp_idx]
                    ddelta_ddextra[i, 1] = dy_ranges_ddextra[cp_idx]

                else:
                    # Get control points this node is between
                    y1 = y_cp[3][np.argmax(y_cp[3] >= pt[1])-1]
                    y2 = y_cp[3][np.argmax(y_cp[3] >= pt[1])]
                    dynew1_ddratio = dy_ranges_ddratio[np.argmax(y_cp[3] >= pt[1])-1]
                    dynew2_ddratio = dy_ranges_ddratio[np.argmax(y_cp[3] >= pt[1])]
                    dynew1_ddextra = dy_ranges_ddextra[np.argmax(y_cp[3] >= pt[1])-1]
                    dynew2_ddextra = dy_ranges_ddextra[np.argmax(y_cp[3] >= pt[1])]
                    ddelta_ddratio[i, 1] = dynew1_ddratio + (dynew2_ddratio - dynew1_ddratio)*(pt[1] - y1)/(y2 - y1)
                    ddelta_ddextra[i, 1] = dynew1_ddextra + (dynew2_ddextra - dynew1_ddextra)*(pt[1] - y1)/(y2 - y1)

            elif np.absolute(pt[1]) < eps:  # bottom edge
                # Check if this is a control point
                if np.any(np.absolute(pt[0] - x_cp[0]) < eps):
                    cp_idx = np.argmin(np.absolute(pt[0] - x_cp[0]))
                    ddelta_ddratio[i, 0] = dx_ranges_ddratio[cp_idx]
                    ddelta_ddextra[i, 0] = dx_ranges_ddextra[cp_idx]
                
                else:
                    # Get control points this node is between
                    x1 = x_cp[0][np.argmax(x_cp[0] >= pt[0])-1]
                    x2 = x_cp[0][np.argmax(x_cp[0] >= pt[0])]
                    dxnew1_ddratio = dx_ranges_ddratio[np.argmax(x_cp[0] >= pt[0])-1]
                    dxnew2_ddratio = dx_ranges_ddratio[np.argmax(x_cp[0] >= pt[0])]
                    dxnew1_ddextra = dx_ranges_ddextra[np.argmax(x_cp[0] >= pt[0])-1]
                    dxnew2_ddextra = dx_ranges_ddextra[np.argmax(x_cp[0] >= pt[0])]
                    ddelta_ddratio[i, 0] = dxnew1_ddratio + (dxnew2_ddratio - dxnew1_ddratio)*(pt[0] - x1)/(x2 - x1)
                    ddelta_ddextra[i, 0] = dxnew1_ddextra + (dxnew2_ddextra - dxnew1_ddextra)*(pt[0] - x1)/(x2 - x1)

            elif np.absolute(pt[0] - w) < eps:  # right edge
                # Check if this is a control point
                if np.any(np.absolute(pt[1] - y_cp[1]) < eps):
                    cp_idx = np.argmin(np.absolute(pt[1] - y_cp[1]))
                    ddelta_ddratio[i, 1] = dy_ranges_ddratio[cp_idx]
                    ddelta_ddextra[i, 1] = dy_ranges_ddextra[cp_idx]

                else:
                    # Get control points this node is between
                    y1 = y_cp[1][np.argmax(y_cp[1] >= pt[1])-1]
                    y2 = y_cp[1][np.argmax(y_cp[1] >= pt[1])]
                    dynew1_ddratio = dy_ranges_ddratio[np.argmax(y_cp[1] >= pt[1])-1]
                    dynew2_ddratio = dy_ranges_ddratio[np.argmax(y_cp[1] >= pt[1])]
                    dynew1_ddextra = dy_ranges_ddextra[np.argmax(y_cp[1] >= pt[1])-1]
                    dynew2_ddextra = dy_ranges_ddextra[np.argmax(y_cp[1] >= pt[1])]
                    ddelta_ddratio[i, 1] = dynew1_ddratio + (dynew2_ddratio - dynew1_ddratio)*(pt[1] - y1)/(y2 - y1)
                    ddelta_ddextra[i, 1] = dynew1_ddextra + (dynew2_ddextra - dynew1_ddextra)*(pt[1] - y1)/(y2 - y1)

            elif np.absolute(pt[1] - l) < eps:  # top edge
                # Check if this is a control point
                if np.any(np.absolute(pt[0] - x_cp[2]) < eps):
                    cp_idx = np.argmin(np.absolute(pt[0] - x_cp[2]))
                    ddelta_ddratio[i, 0] = dy_ranges_ddratio[cp_idx]
                    ddelta_ddextra[i, 0] = dy_ranges_ddextra[cp_idx]

                else:
                    # Get control points this node is between
                    x1 = x_cp[2][np.argmax(x_cp[2] >= pt[0])-1]
                    x2 = x_cp[2][np.argmax(x_cp[2] >= pt[0])]
                    dxnew1_ddratio = dx_ranges_ddratio[np.argmax(x_cp[2] >= pt[0])-1]
                    dxnew2_ddratio = dx_ranges_ddratio[np.argmax(x_cp[2] >= pt[0])]
                    dxnew1_ddextra = dx_ranges_ddextra[np.argmax(x_cp[2] >= pt[0])-1]
                    dxnew2_ddextra = dx_ranges_ddextra[np.argmax(x_cp[2] >= pt[0])]
                    ddelta_ddratio[i, 0] = dxnew1_ddratio + (dxnew2_ddratio - dxnew1_ddratio)*(pt[0] - x1)/(x2 - x1)
                    ddelta_ddextra[i, 0] = dxnew1_ddextra + (dxnew2_ddextra - dxnew1_ddextra)*(pt[0] - x1)/(x2 - x1)

        return ddelta_ddratio, ddelta_ddextra

    def eval_seed_node_derivs(self):

        self.eval_geometry_partials()

        hole_ddelta_ddratio, hole_ddelta_ddextra = self.get_hole_delta_derivs()
        battery_ddelta_ddratio, battery_ddelta_ddextra = self.get_battery_delta_derivs()
        border_ddelta_ddratio, border_ddelta_ddextra = self.get_border_delta_derivs()

        ddelta_ddratio = np.zeros((len(self.Xpts0_cp_idx), 2))
        ddelta_ddextra = np.zeros((len(self.Xpts0_cp_idx), 2))
        ddelta_ddratio[self.hole_start:self.hole_end, :] = hole_ddelta_ddratio[:, :]
        ddelta_ddextra[self.hole_start:self.hole_end, :] = hole_ddelta_ddextra[:, :]
        ddelta_ddratio[self.battery_start:self.battery_end, :] = battery_ddelta_ddratio[:, :]
        ddelta_ddextra[self.battery_start:self.battery_end, :] = battery_ddelta_ddextra[:, :]
        ddelta_ddratio[self.border_start:self.border_end, :] = border_ddelta_ddratio[:, :]
        ddelta_ddextra[self.border_start:self.border_end, :] = border_ddelta_ddextra[:, :]

        return ddelta_ddratio, ddelta_ddratio

    def eval_dep_node_derivs(self, seed_ddelta_dx):

        Xpts_cp = self.Xpts0[self.Xpts0_cp_idx, :]
        ddelta_dx = np.zeros(np.shape(self.Xpts0))

        for i in self.dep_idx:

            # point coordinates with the baseline design
            # this is the point we will warp
            xpts_i = self.Xpts0[i]

            # the vectorized point-based warping we had from older versions.
            rr = xpts_i - Xpts_cp
            LdefoDist = (rr[:,0]**2 + rr[:,1]**2+1e-16)**-0.5
            LdefoDist3 = LdefoDist**3
            Wi = LdefoDist3
            den = np.sum(Wi)
            for iDim in range(2):
                ddelta_dx[i, iDim] = np.sum(Wi*seed_ddelta_dx[:, iDim])/den

        return ddelta_dx

    def compute_partials(self):

        ddelta_ddratio = np.zeros(np.shape(self.Xpts0))
        ddelta_ddextra = np.zeros(np.shape(self.Xpts0))

        seed_ddelta_ddratio, seed_ddelta_ddextra = self.eval_seed_node_derivs()
        dep_ddelta_ddratio = self.eval_dep_node_derivs(seed_ddelta_ddratio)
        dep_ddelta_ddextra = self.eval_dep_node_derivs(seed_ddelta_ddextra)

        ddelta_ddratio[self.Xpts0_cp_idx[:], :] = seed_ddelta_ddratio[:, :]
        ddelta_ddextra[self.Xpts0_cp_idx[:], :] = seed_ddelta_ddextra[:, :]
        ddelta_ddratio[self.dep_idx[:], :] = dep_ddelta_ddratio[self.dep_idx[:], :]
        ddelta_ddextra[self.dep_idx[:], :] = dep_ddelta_ddextra[self.dep_idx[:], :]

        return ddelta_ddratio, ddelta_ddextra

### END OF MESH DEFORMATION CLASS
### -----------------------------

def update_points(Xpts0, indices, Xpts_cp, delta):
    # this function warps points using the displacements from curve projections
    # Xpts0: the original surface point coordinates
    # indices: indices of the dependent nodes
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
        LdefoDist = (rr[:,0]**2 + rr[:,1]**2+1e-16)**-0.5
        LdefoDist3 = LdefoDist**3
        Wi = LdefoDist3
        den = np.sum(Wi)
        interp = np.zeros(2)
        for iDim in range(2):
            interp[iDim] = np.sum(Wi*delta[:, iDim])/den

        # finally, update the coord in place
        Xnew[i] = Xnew[i] + interp

    return Xnew

def get_battery_edge_nodes(Xpts, m=3, n=3, cell_d=0.018, extra=1.5, ratio=0.4, eps=1e-6):
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

    xb = np.repeat(np.linspace(0.5*w/m, 2.5*w/m, m), 3)
    yb = np.tile(np.linspace(0.5*l/n, 2.5*l/n, n), 3).flatten()

    battery_edge_idx = []
    for i in range(len(Xpts)):
        pt = Xpts[i]

        # Find the center of the hole that this point belongs to
        dist = ((pt[0] - xb)**2 + (pt[1] - yb)**2)**0.5 - 0.5*cell_d
        if np.any(np.absolute(dist) < eps):
            battery_edge_idx.append(i)

    return battery_edge_idx

def get_battery_nodes(Xpts, m=3, n=3, cell_d=0.018, extra=1.5, ratio=0.4, eps=1e-6):
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

    xb = np.repeat(np.linspace(0.5*w/m, 2.5*w/m, m), 3)
    yb = np.tile(np.linspace(0.5*l/n, 2.5*l/n, n), 3).flatten()

    battery_idx = []
    for i in range(len(Xpts)):
        pt = Xpts[i]

        # Find the center of the hole that this point belongs to
        dist = ((pt[0] - xb)**2 + (pt[1] - yb)**2)**0.5 - 0.5*cell_d
        if np.any(dist <= eps):
            battery_idx.append(i)

    return battery_idx

def get_hole_nodes(Xpts, m=3, n=3, extra=1.5, ratio=0.4, eps=1e-6):
    """
    Get the indexes of battery edges nodes

    Inputs:
    Xpts: mesh node locations [[x0, y0], [x1, y1], .., [xn, yn]]

    Outputs:
    hole_idx: indexes in the Xpts array that are the hole edges
    """

    w = cell_d*m*extra
    l = cell_d*n*extra
    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

    x_holes = np.repeat(np.linspace(0.0, w, m+1), 4)
    y_holes = np.tile(np.linspace(0.0, l, n+1), 4).flatten()

    hole_idx = []
    for i in range(len(Xpts)):
        pt = Xpts[i]
        dist = ((pt[0] - x_holes)**2 + (pt[1] - y_holes)**2)**0.5 - hole_r
        if np.any(dist < eps):
            hole_idx.append(i)

    return hole_idx

def get_border_nodes(Xpts, m=3, n=3, extra=1.5, eps=1e-6):
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

def get_edge_control_points(Xpts, m=3, n=3, cell_d=0.018, extra=1.5, ratio=0.4, eps=1e-6):
    """
    Get the indexes of battery edges control point nodes (end nodes of each straight section on border)

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

def get_hole_deltas(Xpts0, hole_idx, dratio, dextra, m=3, n=3, cell_d=0.018, extra=1.5, ratio=0.4):

    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
    dhole_r = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

    x_holes = np.repeat(np.linspace(0.0, cell_d*m*extra, m+1), 4)
    y_holes = np.tile(np.linspace(0.0, cell_d*n*extra, n+1), 4).flatten()#order="F")
    dx_holes = np.repeat(np.linspace(0.0, cell_d*m*dextra, m+1), 4)
    dy_holes = np.tile(np.linspace(0.0, cell_d*n*dextra, n+1), 4).flatten()#order="F")

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

def get_battery_deltas(Xpts0, battery_idx, dextra, m=3, n=3, cell_d=0.018, extra=1.5, ratio=0.4):

    w = cell_d*m*extra
    l = cell_d*n*extra
    dw = cell_d*m*dextra
    dl = cell_d*n*dextra

    xb = np.repeat(np.linspace(0.5*w/m, 2.5*w/m, m), 3)
    yb = np.tile(np.linspace(0.5*l/n, 2.5*l/n, n), 3).flatten()
    dxb = np.repeat(np.linspace(0.5*dw/m, 2.5*dw/m, m), 3)
    dyb = np.tile(np.linspace(0.5*dl/n, 2.5*dl/m, n), 3).flatten()

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

def get_border_deltas(Xpts0, border_idx, dratio, dextra, m=3, n=3, cell_d=0.018, extra=1.5, ratio=0.4, eps=1e-6):

    w = cell_d*m*extra
    l = cell_d*n*extra
    dw = cell_d*m*dextra
    dl = cell_d*n*dextra

    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
    dhole_r = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

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
    x_cp = np.sort(Xpts0[edge_cp_idx[:], 0])
    y_cp = np.sort(Xpts0[edge_cp_idx[:], 1])
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
                xnew1 = x_ranges[np.argmax(x_cp[0] >= pt[0])-1]
                xnew2 = x_ranges[np.argmax(x_cp[0] >= pt[0])]
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
                xnew1 = x_ranges[np.argmax(x_cp[2] >= pt[0])-1]
                xnew2 = x_ranges[np.argmax(x_cp[2] >= pt[0])]
                border_deltas[i, 0] = xnew1 + (pt[0] - x1)*(xnew2 - xnew1)/(x2 - x1) - pt[0]
            border_deltas[i, 1] = dl  # shift all top border nodes by dl

    return border_deltas

def get_hole_delta_derivs(Xpts0, hole_idx, dratio, dextra, m=3, n=3, cell_d=0.018, extra=1.5, ratio=0.4):

    w = cell_d*m*extra
    l = cell_d*n*extra
    dw = cell_d*m*dextra
    dl = cell_d*n*dextra

    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
    dhole_r = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

    x_holes = np.repeat(np.linspace(0.0, cell_d*m*extra, m+1), 4)
    y_holes = np.tile(np.linspace(0.0, cell_d*n*extra, n+1), 4).flatten()#order="F")
    dx_holes = np.repeat(np.linspace(0.0, cell_d*m*dextra, m+1), 4)
    dy_holes = np.tile(np.linspace(0.0, cell_d*n*dextra, n+1), 4).flatten()#order="F")

    pert = 1e-10
    dr2 = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra+pert)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
    ddr_ddextra_fd = (dr2 - dhole_r)/pert

    ddr_ddratio = 0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0)
    ddr_ddextra = ddr_ddextra_fd
    ddx_ddextra = np.repeat(np.linspace(0.0, cell_d*m, m+1), 4)
    ddy_ddextra = np.tile(np.linspace(0.0, cell_d*n, n+1), 4).flatten()#order="F")

    ddelta_ddratio = np.zeros((len(hole_idx), 2))
    ddelta_ddextra = np.zeros((len(hole_idx), 2))
    for i, idx in enumerate(hole_idx):
        pt = Xpts0[idx]

        # Find the center of the hole that this point belongs to
        dist = ((pt[0] - x_holes)**2 + (pt[1] - y_holes)**2)**0.5 - hole_r
        which_hole = np.argmin(dist)
        x0 = x_holes[which_hole]
        y0 = y_holes[which_hole]

        # Get the angle of the point wrt the hole center
        theta = np.arctan2(pt[1]-y0, pt[0]-x0)

        # Compute the delta  derivatives for this point
        ddelta_ddratio[i, 0] = np.cos(theta)*ddr_ddratio
        ddelta_ddratio[i, 1] = np.sin(theta)*ddr_ddratio

        ddelta_ddextra[i, 0] = np.cos(theta)*ddr_ddextra + ddx_ddextra[which_hole]
        ddelta_ddextra[i, 1] = np.sin(theta)*ddr_ddextra + ddy_ddextra[which_hole]

    return ddelta_ddratio, ddelta_ddextra

def get_battery_delta_derivs(Xpts0, battery_idx, dextra, m=3, n=3, cell_d=0.018, extra=1.5, ratio=0.4):

    w = cell_d*m*extra
    l = cell_d*n*extra
    dw = cell_d*m*dextra
    dl = cell_d*n*dextra

    xb = np.repeat(np.linspace(0.5*w/m, 2.5*w/m, m), 3)
    yb = np.tile(np.linspace(0.5*l/n, 2.5*l/n, n), 3).flatten()
    ddxb_ddextra = np.repeat(np.linspace(0.5*cell_d, 2.5*cell_d, m), 3)
    ddyb_ddextra = np.tile(np.linspace(0.5*cell_d, 2.5*cell_d, n), 3).flatten()

    ddelta_ddratio = np.zeros((len(battery_idx), 2))
    ddelta_ddextra = np.zeros((len(battery_idx), 2))
    for i, idx in enumerate(battery_idx):
        pt = Xpts0[idx]

        # Find the center of the hole that this point belongs to
        dist = ((pt[0] - xb)**2 + (pt[1] - yb)**2)**0.5 - 0.5*cell_d
        which_battery = np.argmin(dist)

        # Compute the delta for this point
        ddelta_ddextra[i, 0] = ddxb_ddextra[which_battery]
        ddelta_ddextra[i, 1] = ddyb_ddextra[which_battery]

    return ddelta_ddratio, ddelta_ddextra

def get_border_delta_derivs(Xpts0, border_idx, dratio, dextra, m=3, n=3, cell_d=0.018, extra=1.5, ratio=0.4, eps=1e-6):

    w = cell_d*m*extra
    l = cell_d*n*extra
    dw = cell_d*m*dextra
    dl = cell_d*n*dextra

    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
    dhole_r = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

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

    # Define the intermediate derivatives
    ddw_ddextra = cell_d*m
    ddl_ddextra = cell_d*n

    pert = 1e-6
    dr1 = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
    dr2 = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra+pert)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
    ddr_ddextra_fd = (dr2 - dr1)/pert

    ddr_ddratio = 0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0)
    ddr_ddextra = ddr_ddextra_fd  #0.346574*(ratio+dratio)*0.5*cell_d*(2.0**0.5*(extra+dextra))

    ddx_ddratio = -2.0*ddr_ddratio
    ddx_ddextra = cell_d - 2.0*ddr_ddextra
    ddy_ddratio = -2.0*ddr_ddratio
    ddy_ddextra = cell_d - 2.0*ddr_ddextra

    dx_ranges_ddextra = [ddr_ddextra, ddr_ddextra + ddx_ddextra, 
                         3*ddr_ddextra + ddx_ddextra, 3*ddr_ddextra + 2*ddx_ddextra, 
                         5*ddr_ddextra + 2*ddx_ddextra, 5*ddr_ddextra + 3*ddx_ddextra]
    dx_ranges_ddratio = [ddr_ddratio, ddr_ddratio + ddx_ddratio, 
                         3*ddr_ddratio + ddx_ddratio, 3*ddr_ddratio + 2*ddx_ddratio, 
                         5*ddr_ddratio + 2*ddx_ddratio, 5*ddr_ddratio + 3*ddx_ddratio]
    dy_ranges_ddextra = [ddr_ddextra, ddr_ddextra + ddy_ddextra, 
                         3*ddr_ddextra + ddx_ddextra, 3*ddr_ddextra + 2*ddy_ddextra, 
                         5*ddr_ddextra + 2*ddy_ddextra, 5*ddr_ddextra + 3*ddy_ddextra]
    dy_ranges_ddratio = [ddr_ddratio, ddr_ddratio + ddy_ddratio, 
                         3*ddr_ddratio + ddy_ddratio, 3*ddr_ddratio + 2*ddy_ddratio, 
                         5*ddr_ddratio + 2*ddy_ddratio, 5*ddr_ddratio + 3*ddy_ddratio]

    edge_cp_idx = get_edge_control_points(Xpts0)
    x_cp = np.sort(Xpts0[edge_cp_idx[:], 0])
    y_cp = np.sort(Xpts0[edge_cp_idx[:], 1])

    ddelta_ddratio = np.zeros((len(border_idx), 2))
    ddelta_ddextra = np.zeros((len(border_idx), 2))
    for i, idx in enumerate(border_idx):
        pt = Xpts0[idx]
        if np.absolute(pt[0]) < eps:  # left edge
            # Check if this is a control point
            if np.any(np.absolute(pt[1] - y_cp[3]) < eps):
                cp_idx = np.argmin(np.absolute(pt[1] - y_cp[3]))
                ddelta_ddratio[i, 1] = dy_ranges_ddratio[cp_idx]
                ddelta_ddextra[i, 1] = dy_ranges_ddextra[cp_idx]

            else:
                # Get control points this node is between
                y1 = y_cp[3][np.argmax(y_cp[3] >= pt[1])-1]
                y2 = y_cp[3][np.argmax(y_cp[3] >= pt[1])]
                dynew1_ddratio = dy_ranges_ddratio[np.argmax(y_cp[3] >= pt[1])-1]
                dynew2_ddratio = dy_ranges_ddratio[np.argmax(y_cp[3] >= pt[1])]
                dynew1_ddextra = dy_ranges_ddextra[np.argmax(y_cp[3] >= pt[1])-1]
                dynew2_ddextra = dy_ranges_ddextra[np.argmax(y_cp[3] >= pt[1])]
                ddelta_ddratio[i, 1] = dynew1_ddratio + (dynew2_ddratio - dynew1_ddratio)*(pt[1] - y1)/(y2 - y1)
                ddelta_ddextra[i, 1] = dynew1_ddextra + (dynew2_ddextra - dynew1_ddextra)*(pt[1] - y1)/(y2 - y1)

        elif np.absolute(pt[1]) < eps:  # bottom edge
            # Check if this is a control point
            if np.any(np.absolute(pt[0] - x_cp[0]) < eps):
                cp_idx = np.argmin(np.absolute(pt[0] - x_cp[0]))
                ddelta_ddratio[i, 0] = dx_ranges_ddratio[cp_idx]
                ddelta_ddextra[i, 0] = dx_ranges_ddextra[cp_idx]
            
            else:
                # Get control points this node is between
                x1 = x_cp[0][np.argmax(x_cp[0] >= pt[0])-1]
                x2 = x_cp[0][np.argmax(x_cp[0] >= pt[0])]
                dxnew1_ddratio = dx_ranges_ddratio[np.argmax(x_cp[0] >= pt[0])-1]
                dxnew2_ddratio = dx_ranges_ddratio[np.argmax(x_cp[0] >= pt[0])]
                dxnew1_ddextra = dx_ranges_ddextra[np.argmax(x_cp[0] >= pt[0])-1]
                dxnew2_ddextra = dx_ranges_ddextra[np.argmax(x_cp[0] >= pt[0])]
                ddelta_ddratio[i, 0] = dxnew1_ddratio + (dxnew2_ddratio - dxnew1_ddratio)*(pt[0] - x1)/(x2 - x1)
                ddelta_ddextra[i, 0] = dxnew1_ddextra + (dxnew2_ddextra - dxnew1_ddextra)*(pt[0] - x1)/(x2 - x1)

        elif np.absolute(pt[0] - w) < eps:  # right edge
            # Check if this is a control point
            if np.any(np.absolute(pt[1] - y_cp[1]) < eps):
                cp_idx = np.argmin(np.absolute(pt[1] - y_cp[1]))
                ddelta_ddratio[i, 1] = dy_ranges_ddratio[cp_idx]
                ddelta_ddextra[i, 1] = dy_ranges_ddextra[cp_idx]

            else:
                # Get control points this node is between
                y1 = y_cp[1][np.argmax(y_cp[1] >= pt[1])-1]
                y2 = y_cp[1][np.argmax(y_cp[1] >= pt[1])]
                dynew1_ddratio = dy_ranges_ddratio[np.argmax(y_cp[1] >= pt[1])-1]
                dynew2_ddratio = dy_ranges_ddratio[np.argmax(y_cp[1] >= pt[1])]
                dynew1_ddextra = dy_ranges_ddextra[np.argmax(y_cp[1] >= pt[1])-1]
                dynew2_ddextra = dy_ranges_ddextra[np.argmax(y_cp[1] >= pt[1])]
                ddelta_ddratio[i, 1] = dynew1_ddratio + (dynew2_ddratio - dynew1_ddratio)*(pt[1] - y1)/(y2 - y1)
                ddelta_ddextra[i, 1] = dynew1_ddextra + (dynew2_ddextra - dynew1_ddextra)*(pt[1] - y1)/(y2 - y1)

        elif np.absolute(pt[1] - l) < eps:  # top edge
            # Check if this is a control point
            if np.any(np.absolute(pt[0] - x_cp[2]) < eps):
                cp_idx = np.argmin(np.absolute(pt[0] - x_cp[2]))
                ddelta_ddratio[i, 0] = dy_ranges_ddratio[cp_idx]
                ddelta_ddextra[i, 0] = dy_ranges_ddextra[cp_idx]

            else:
                # Get control points this node is between
                x1 = x_cp[2][np.argmax(x_cp[2] >= pt[0])-1]
                x2 = x_cp[2][np.argmax(x_cp[2] >= pt[0])]
                dxnew1_ddratio = dx_ranges_ddratio[np.argmax(x_cp[2] >= pt[0])-1]
                dxnew2_ddratio = dx_ranges_ddratio[np.argmax(x_cp[2] >= pt[0])]
                dxnew1_ddextra = dx_ranges_ddextra[np.argmax(x_cp[2] >= pt[0])-1]
                dxnew2_ddextra = dx_ranges_ddextra[np.argmax(x_cp[2] >= pt[0])]
                ddelta_ddratio[i, 0] = dxnew1_ddratio + (dxnew2_ddratio - dxnew1_ddratio)*(pt[0] - x1)/(x2 - x1)
                ddelta_ddextra[i, 0] = dxnew1_ddextra + (dxnew2_ddextra - dxnew1_ddextra)*(pt[0] - x1)/(x2 - x1)

    return ddelta_ddratio, ddelta_ddextra

def make_ghost_nodes_for_holes(h, dratio, dextra, m=3, n=3, cell_d=0.018, extra=1.5, ratio=0.4):
    # h: spacing between nodes (1 per distance h)

    w = cell_d*m*extra
    l = cell_d*n*extra
    dw = cell_d*m*dextra
    dl = cell_d*n*dextra

    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
    dhole_r = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

    x_holes = np.repeat(np.linspace(0.0, w, m+1), 4)
    y_holes = np.tile(np.linspace(0.0, l, n+1), 4).flatten()
    dx_holes = np.repeat(np.linspace(0.0, dw, m+1), 4)
    dy_holes = np.tile(np.linspace(0.0, dl, n+1), 4).flatten()

    num_pts = int(2.0*np.pi*hole_r/h)
    theta = np.linspace(0.0, 2.0*np.pi, num_pts)
    ghost_xpts_hole = np.array([], dtype=np.float64, shape=(0, 2))
    ghost_deltas_hole = np.array([], dtype=np.float64, shape=(0, 2))
    for i in range(len(x_holes)):

        # Set the locations of the undeformed ghost nodes
        xi = x_holes[i] + hole_r*np.cos(theta)
        yi = y_holes[i] + hole_r*np.sin(theta)

        xi = np.reshape(xi, (len(xi), 1))
        yi = np.reshape(yi, (len(yi), 1))
        xpts_i = np.concatenate(xi, yi, axis=1)
        ghost_xpts_hole = np.concatenate((ghost_xpts_hole, xpts_i))

        # Set the delta for each node


    return ghost_xpts_hole, ghost_deltas_hole

class Intermediates(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("m", types=int, default=3, desc="number of battery columns in the pack")
        self.options.declare("n", types=int, default=3, desc="number of battery rows in the pack")
        self.options.declare("cell_d", types=float, default=0.018, desc="battery cell diameter (m)")
        self.options.declare("extra", types=float, default=1.5, desc="parametrized spacing between cells")
        self.options.declare("ratio", types=float, default=0.4, desc="parametrized hole size: ratio of hole size to max available space")

    def setup(self):

        self.add_input("dratio", val=0.0, units=None, desc="change in ratio parameter from the initial mesh")
        self.add_input("dextra", val=0.0, units=None, desc="change in extra parameter from the initial mesh")

        self.add_output("dw", val=np.zeros(16), units="m")
        self.add_output("dl", val=np.zeros(16), units="m")
        self.add_output("dxb", val=np.zeros(9), units="m")
        self.add_output("dyb", val=np.zeros(9), units="m")
        self.add_output("dx_holes", val=np.zeros(16), units="m")
        self.add_output("dy_holes", val=np.zeros(16), units="m")
        self.add_output("dhole_r", val=0.0, units="m")
        self.add_output("dx_border_len", val=0.0, units="m")
        self.add_output("dy_border_len", val=0.0, units="m")
        self.add_output("x_ranges", val=6*[0.0], units="m")
        self.add_output("y_ranges", val=6*[0.0], units="m")

        self.declare_partials(of=["dw", "dl", "dxb", "dyb", "dx_holes", "dy_holes",
                                  "dhole_r", "dx_border_len", "dy_border_len",
                                  "x_ranges", "y_ranges"],
                              wrt=["dratio", "dextra"])

    def compute(self, inputs, outputs):

        m = self.options["m"]
        n = self.options["n"]
        cell_d = self.options["cell_d"]
        extra = self.options["extra"]
        ratio = self.options["ratio"]

        dratio = inputs["dratio"]
        dextra = inputs["dextra"]
        
        dw = outputs["dw"]
        dl = outputs["dl"]
        dxb = outputs["dxb"]
        dyb = outputs["dyb"]
        dx_holes = outputs["dx_holes"]
        dy_holes = outputs["dy_holes"]
        dhole_r = outputs["dhole_r"]
        dx_border_len = outputs["dx_border_len"]
        dy_border_len = outputs["dy_border_len"]
        x_ranges = outputs["x_ranges"]
        y_ranges = outputs["y_ranges"]

        dw[:] = np.repeat(np.linspace(0.0, cell_d*m*dextra, m+1), 4)
        dl[:] = np.tile(np.linspace(0.0, cell_d*n*dextra, n+1), 4).flatten(order="F")

        dxb[:] = np.repeat(np.linspace(0.5*cell_d*m*dextra/m, 2.5*cell_d*m*dextra/m, m), 3)
        dyb[:] = np.tile(np.linspace(0.5*cell_d*n*dextra/n, 2.5*cell_d*n*dextra/m, n), 3).flatten(order="F")

        dx_holes[:] = np.repeat(np.linspace(0.0, cell_d*m*dextra, m+1), 4)
        dy_holes[:] = np.tile(np.linspace(0.0, cell_d*n*dextra, n+1), 4).flatten(order="F")

        hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
        dhole_r[:] = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

        x_border_len = (w-2*m*hole_r)/m
        y_border_len = (l-2*n*hole_r)/n
        dx_border_len[:] = cell_d*dextra-2*dhole_r
        dy_border_len[:] = cell_d*dextra-2*dhole_r

        x_ranges[:] = [hole_r+dhole_r, hole_r+dhole_r + (x_border_len+dx_border_len), 
                       3*(hole_r+dhole_r) + (x_border_len+dx_border_len), 3*(hole_r+dhole_r) + 2*(x_border_len+dx_border_len), 
                       5*(hole_r+dhole_r) + 2*(x_border_len+dx_border_len), 5*(hole_r+dhole_r) + 3*(x_border_len+dx_border_len)][:]
        y_ranges[:] = [hole_r+dhole_r, hole_r+dhole_r + (y_border_len+dy_border_len), 
                       3*(hole_r+dhole_r) + (y_border_len+dy_border_len), 3*(hole_r+dhole_r) + 2*(y_border_len+dy_border_len), 
                       5*(hole_r+dhole_r) + 2*(y_border_len+dy_border_len), 5*(hole_r+dhole_r) + 3*(y_border_len+dy_border_len)][:]

    def compute_partials(self, inputs, partials):

        m = self.options["m"]
        n = self.options["n"]
        cell_d = self.options["cell_d"]
        extra = self.options["extra"]
        ratio = self.options["ratio"]

        dratio = inputs["dratio"]
        dextra = inputs["dextra"]

        ddw_ddratio = partials["dw", "dratio"]
        ddl_ddratio = partials["dl", "dratio"]
        ddxb_ddratio = partials["dxb", "dratio"]
        ddyb_ddratio = partials["dyb", "dratio"]
        ddx_holes_ddratio = partials["dx_holes", "dratio"]
        ddy_holes_ddratio = partials["dy_holes", "dratio"]
        ddr_ddratio = partials["dhole_r", "dratio"]
        ddx_ddratio = partials["dx_border_len", "dratio"]
        ddy_ddratio = partials["dy_border_len", "dratio"]
        dx_ranges_ddratio = partials["x_ranges", "dratio"]
        dy_ranges_ddratio = partials["y_ranges", "dratio"]

        ddw_ddextra = partials["dw", "dextra"]
        ddl_ddextra = partials["dl", "dextra"]
        ddxb_ddextra = partials["dxb", "dextra"]
        ddyb_ddextra = partials["dyb", "dextra"]
        ddx_holes_ddextra = partials["dx_holes", "dextra"]
        ddy_holes_ddextra = partials["dy_holes", "dextra"]
        ddr_ddextra = partials["dhole_r", "dextra"]
        ddx_ddextra = partials["dx_border_len", "dextra"]
        ddy_ddextra = partials["dy_border_len", "dextra"]
        dx_ranges_ddextra = partials["x_ranges", "dextra"]
        dy_ranges_ddextra = partials["y_ranges", "dextra"]

        ddw_ddextra[:, :] = np.reshape(np.repeat(np.linspace(0.0, cell_d*m, m+1), 4), ddw_ddextra.shape)[:, :]
        ddl_ddextra[:, :] = np.reshape(np.tile(np.linspace(0.0, cell_d*n, n+1), 4).flatten(order="F"), ddl_ddextra.shape)[:, :]

        ddxb_ddextra[:, :] = np.reshape(np.repeat(np.linspace(0.5*cell_d, 2.5*cell_d, m), 3), ddxb_ddextra.shape)[:, :]
        ddyb_ddextra[:, :] = np.reshape(np.tile(np.linspace(0.5*cell_d, 2.5*cell_d, n), 3).flatten(order="F"), ddyb_ddextra.shape)[:, :]

        ddx_holes_ddextra[:, :] = np.reshape(np.repeat(np.linspace(0.0, cell_d*m, m+1), 4), ddx_holes_ddextra.shape)[:, :]
        ddy_holes_ddextra[:, :] = np.reshape(np.tile(np.linspace(0.0, cell_d*n, n+1), 4).flatten(order="F"), ddy_holes_ddextra.shape)[:, :]

        ddr_ddratio[:, :] = 0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0)

        # Finite difference this value myself
        pert = 1e-10
        dr1 = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
        dr2 = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra+pert)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
        ddr_ddextra_fd = (dr2 - dr1)/pert

        ddr_ddextra[:, :] = ddr_ddextra_fd[:]

        ddx_ddratio[:, :] = -2.0*ddr_ddratio
        ddx_ddextra[:, :] = cell_d - 2.0*ddr_ddextra
        ddy_ddratio[:, :] = -2.0*ddr_ddratio
        ddy_ddextra[:, :] = cell_d - 2.0*ddr_ddextra

        dx_ranges_ddextra[:, :] = [ddr_ddextra, ddr_ddextra + ddx_ddextra, 
                                   3*ddr_ddextra + ddx_ddextra, 3*ddr_ddextra + 2*ddx_ddextra, 
                                   5*ddr_ddextra + 2*ddx_ddextra, 5*ddr_ddextra + 3*ddx_ddextra]
        dx_ranges_ddratio[:, :] = [ddr_ddratio, ddr_ddratio + ddx_ddratio, 
                                   3*ddr_ddratio + ddx_ddratio, 3*ddr_ddratio + 2*ddx_ddratio, 
                                   5*ddr_ddratio + 2*ddx_ddratio, 5*ddr_ddratio + 3*ddx_ddratio]
        dy_ranges_ddextra[:, :] = [ddr_ddextra, ddr_ddextra + ddy_ddextra, 
                                   3*ddr_ddextra + ddx_ddextra, 3*ddr_ddextra + 2*ddy_ddextra, 
                                   5*ddr_ddextra + 2*ddy_ddextra, 5*ddr_ddextra + 3*ddy_ddextra]
        dy_ranges_ddratio[:, :] = [ddr_ddratio, ddr_ddratio + ddy_ddratio, 
                                   3*ddr_ddratio + ddy_ddratio, 3*ddr_ddratio + 2*ddy_ddratio, 
                                   5*ddr_ddratio + 2*ddy_ddratio, 5*ddr_ddratio + 3*ddy_ddratio]


class MeshDeformComp(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare("m", types=int, default=3, desc="number of battery columns in the pack")
        self.options.declare("n", types=int, default=3, desc="number of battery rows in the pack")
        self.options.declare("cell_d", types=float, default=0.018, desc="battery cell diameter (m)")
        self.options.declare("extra", types=float, default=1.5, desc="parametrized spacing between cells")
        self.options.declare("ratio", types=float, default=0.4, desc="parametrized hole size: ratio of hole size to max available space")
        self.options.declare("nnodes", types=int, desc="number of nodes in the mesh")
        self.options.declare("Xpts0", desc="Copy of initial mesh points which are static")
    
    def setup(self):
        Xpts0 = self.options["Xpts0"]
        m = self.options["m"]
        n = self.options["n"]
        cell_d = self.options["cell_d"]
        extra = self.options["extra"]
        ratio = self.options["ratio"]
        nnodes = self.options["nnodes"]

        # Initialize the mesh deformation object
        self.md = MeshDeformation(Xpts0, m=m, n=n, cell_d=cell_d, extra=extra, ratio=ratio)

        self.add_input("dratio", val=0.0, units=None, desc="change in ratio parameter from the initial mesh")
        self.add_input("dextra", val=0.0, units=None, desc="change in extra parameter from the initial mesh")

        self.add_output("Xpts", shape=(nnodes, 2), units="m", desc="Initial mesh coordinates")
        
        self.declare_partials(of="Xpts", wrt=["dratio", "dextra"])

    def compute(self, inputs, outputs):

        dratio = inputs["dratio"]
        dextra = inputs["dextra"]
        Xpts = outputs["Xpts"]

        # Update the parameterization
        self.md.dratio = dratio
        self.md.dextra = dextra

        # Deform the geometry
        Xnew = self.md.deform_geometry()
        Xpts[:, :] = Xnew[:, :]

        # # Finite difference the mesh deformation and output a csv
        # pert = 1e-10
        # Xpert = np.zeros(np.shape(Xpts))
        # dextra += pert
        # battery_deltas = get_battery_deltas(Xpts0, battery_edge_idx, dextra)
        # delta = np.zeros((len(Xpts0_cp_idx), 2))
        # delta[0:len(battery_edge_idx), :] = battery_deltas[:, :]
        # Xnew = update_points(Xpts0, dep_idx, Xpts0_cp, np.zeros((len(Xpts0_cp_idx), 2)))
        # Xnew[Xpts0_cp_idx[:], :] += delta[:, :]
        # Xpert[:, :] = Xnew[:, :]
        # dXpts_ddextra_fd = (Xpert - Xpts)/pert
        # np.savetxt("dXpts_ddextra_fd.csv", dXpts_ddextra_fd, delimiter=",")

    def compute_partials(self, inputs, partials):

        nnodes = self.options["nnodes"]

        dXpts_ddratio = partials["Xpts", "dratio"]
        dXpts_ddextra = partials["Xpts", "dextra"]

        # Reshape the derivative arrays
        dXpts_ddratio = dXpts_ddratio.reshape((nnodes, 2))
        dXpts_ddextra = dXpts_ddextra.reshape((nnodes, 2))

        ddelta_ddratio, ddelta_ddextra = self.md.compute_partials()
        dXpts_ddratio[:] = ddelta_ddratio[:]
        dXpts_ddextra[:] = ddelta_ddextra[:]

        # Undo the reshaping of the derivative arrays
        dXpts_ddratio = dXpts_ddratio.flatten()
        dXpts_ddextra = dXpts_ddextra.flatten()

comm = MPI.COMM_WORLD

# Instantiate FEASolver
structOptions = {
    # Specify what type of elements we want in the f5
    "outputElement": TACS.PLANE_STRESS_ELEMENT,
}

# Instantiate FEASolver
bdfFile = os.path.join(os.path.dirname(__file__), "battery_ratio_0.4_extra_1.5.bdf")
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
    if compDescript == "block":
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
        if elemDescript in ["CQUAD4", "CQUADR"]:
            basis = elements.LinearQuadBasis()
        elif elemDescript in ["CTRIA3", "CTRIAR"]:
            basis = elements.LinearTriangleBasis()
        elem = elements.Element2D(model, basis)
        elemList.append(elem)

    return elemList

# Set up constitutive objects and elements
FEAAssembler.initialize(elemCallBack)
problem = FEAAssembler.createStaticProblem("problem")

# Get the mesh points
Xpts0 = FEAAssembler.getOrigNodes()
Xnew_tacs = np.zeros(np.shape(Xpts0))

# Drop the z-values and reshape the vector
Xpts0 = np.delete(Xpts0, np.arange(2, Xpts0.size, 3))
nnodes = int(Xpts0.size/2)
Xpts0 = Xpts0.reshape((nnodes, 2))

# Parametrize the geometry
m = 3  # number of rows of battery cells
n = 3  # number of columns of battery cells
cell_d = 0.018  # diameter of the cell
extra = 1.5  # extra space along the diagonal, in [1, infty)
ratio = 0.4  # cell diameter/cutout diamter, in [0, 1]

# Get the control point node indexes
battery_edge_idx = get_battery_edge_nodes(Xpts0)
battery_idx = get_battery_nodes(Xpts0)
hole_idx = get_hole_nodes(Xpts0)
border_idx = get_border_nodes(Xpts0)

# Define the change in geometric parameters to be applied
dratio = 0.3
dextra = 0.3

w = cell_d*m*extra
l = cell_d*n*extra
dw = cell_d*m*dextra
dl = cell_d*n*dextra
hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
dhole_r = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
xb = np.repeat(np.linspace(0.5*w/m, 2.5*w/m, m), 3)
yb = np.tile(np.linspace(0.5*l/n, 2.5*l/n, n), 3).flatten()
dxb = np.repeat(np.linspace(0.5*dw/m, 2.5*dw/m, m), 3)
dyb = np.tile(np.linspace(0.5*dl/n, 2.5*dl/m, n), 3).flatten()

dr_pct = 100.0*(dhole_r/hole_r)
dxb_pct = 100.0*(dyb[1]-dyb[0])/(yb[1]-yb[0])

print(f"Hole radius change is {dr_pct}%")
print(f"Battery spacing change is {dxb_pct}%")

### Test the MeshDeformation class
# md = MeshDeformation(Xpts0, dratio=dratio, dextra=dextra)
# Xnew = md.deform_geometry()

# s = 2.0
# fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)
# ax0.scatter(Xpts0[:, 0], Xpts0[:, 1], s=2, color="tab:blue")
# ax1.scatter(Xnew[:, 0], Xnew[:, 1], s=2, color="tab:blue")
# plt.show()

# md.dratio = 0.1
# md.dextra = 0.1
# Xnew = md.deform_geometry()
# ddelta_ddratio, ddelta_ddextra = md.compute_partials()

# fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)
# ax0.scatter(Xpts0[:, 0], Xpts0[:, 1], s=2, color="tab:blue")
# ax1.scatter(Xnew[:, 0], Xnew[:, 1], s=2, color="tab:blue")
# plt.show()

#####

### test the derivative
# dextra_range = np.linspace(-0.4, 0.4, 20)
# dfdx = np.zeros(len(dextra_range))
# dfdx_fd = np.zeros(len(dextra_range))
# for i in range(len(dextra_range)):
#     pert = 1e-10
#     dr1 = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra_range[i])) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
#     dr2 = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra_range[i]+pert)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
#     dfdx_fd[i] = (dr2 - dr1)/pert
#     dfdx[i] = (np.log(2.0)/4.0)*(ratio+dratio)*cell_d*(2.0**0.5*(extra+dextra_range[i]))

# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot()
# ax.plot(dextra_range, dfdx, label="Analytic")
# ax.plot(dextra_range, dfdx_fd, label="FD")
# ax.legend()
# plt.show()

###


# ###### Try out the openmdao component

p = om.Problem()
ivc = om.IndepVarComp()
ivc.add_output("dratio", dratio)
ivc.add_output("dextra", dextra)
p.model.add_subsystem("ivc", ivc, promotes=["*"])

# Check the intermediate derivatives
#intermediates_comp = Intermediates()
#p.model.add_subsystem("intermediates", intermediates_comp, promotes=["*"])

mesh_deform_comp = MeshDeformComp(Xpts0=Xpts0, nnodes=nnodes)
p.model.add_subsystem("mesh_deform_comp", mesh_deform_comp, promotes=["*"])

p.setup(check=True)
p.run_model()
p.check_partials(compact_print=False)
asd
# ######

# Get dependent node indices
dep_idx = []
for i in range(len(Xpts0)):
    if (i not in hole_idx) and (i not in border_idx) and (i not in battery_edge_idx):
        dep_idx.append(i)

non_battery_idx = []
for i in range(len(Xpts0)):
    if i not in battery_idx:
        non_battery_idx.append(i)

# Get the node locations of the control points
Xpts0_cp_idx = hole_idx + border_idx + battery_edge_idx
Xpts0_cp = Xpts0[Xpts0_cp_idx[:]]

# Compute the delta of the hole points
hole_deltas = get_hole_deltas(Xpts0, hole_idx, dratio, dextra)
new_hole_pts = Xpts0[hole_idx[:], :] + hole_deltas[:, :]

# Get the delta of the battery nodes
battery_deltas = get_battery_deltas(Xpts0, battery_edge_idx, dextra)
new_battery_pts = Xpts0[battery_edge_idx[:], :] + battery_deltas[:, :]

# Get the delta of the border edges
border_deltas = get_border_deltas(Xpts0, border_idx, dratio, dextra)
new_border_pts = Xpts0[border_idx[:], :] + border_deltas[:, :]

# Set the total delta array
delta = np.zeros((len(Xpts0_cp_idx), 2))
delta[0:len(hole_idx), :] = hole_deltas[:, :]
delta[len(hole_idx):len(hole_idx)+len(border_idx), :] = border_deltas[:, :]
delta[len(hole_idx)+len(border_idx):len(hole_idx)+len(border_idx)+len(battery_edge_idx), :] = battery_deltas[:, :]

# Get the updated locations of the dependent nodes
Xnew = update_points(Xpts0, dep_idx, Xpts0_cp, delta)

# Update the independent node locations
Xnew[Xpts0_cp_idx[:], :] += delta[:, :]

# Plot the initial and deformed meshes nodes
s = 2.0
fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)
#ax0.scatter(Xpts0[:, 0], Xpts0[:, 1], s=2, color="tab:blue")
ax0.scatter(Xpts0[non_battery_idx[:], 0], Xpts0[non_battery_idx[:], 1], s=s, color="tab:blue")
ax0.scatter(Xpts0[battery_idx[:], 0], Xpts0[battery_idx[:], 1], s=s, color="tab:blue", alpha=0.5)
ax0.scatter(Xpts0[Xpts0_cp_idx[:], 0], Xpts0[Xpts0_cp_idx[:], 1], s=s, color="tab:red")
ax1.scatter(Xnew[non_battery_idx[:], 0], Xnew[non_battery_idx[:], 1], s=s, color="tab:blue")
ax1.scatter(Xnew[battery_idx[:], 0], Xnew[battery_idx[:], 1], s=s, color="tab:blue", alpha=0.5)
ax1.scatter(Xnew[Xpts0_cp_idx[:], 0], Xnew[Xpts0_cp_idx[:], 1], s=s, color="tab:red")
ax0.scatter(Xpts0[battery_edge_idx[:], 0], Xpts0[battery_edge_idx[:], 1], s=s, color="tab:green")

ax0.set_aspect("equal")
ax1.set_aspect("equal")
ax0.axis("off")
ax1.axis("off")
ax0.set_title("Original mesh")
ax1.set_title("Deformed mesh")

plt.show()
#plt.savefig(f"deformed_points_dratio_{dratio}_dextra_{dextra}.png", dpi=300)

# Update the nodes in the TACS Assembler object and write out the mesh
Xnew_tacs[0::3] = Xnew[:, 0]
Xnew_tacs[1::3] = Xnew[:, 1]
problem.setNodes(Xnew_tacs)
problem.writeSolution()
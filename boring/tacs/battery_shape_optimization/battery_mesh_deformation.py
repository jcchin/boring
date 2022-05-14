import os
import numpy as np
from mpi4py import MPI

import openmdao.api as om
from tacs import functions, constitutive, elements, TACS, pyTACS


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
        self.eps = 1e-4

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
            if (i not in self.hole_idx) and (i not in self.battery_edge_idx) and (i not in self.border_idx):
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
        dratio = self.dratio

        self.w = cell_d*m*extra
        self.l = cell_d*n*extra
        self.dw = cell_d*m*dextra
        self.dl = cell_d*n*dextra

        self.xb = np.repeat(np.linspace(0.5*self.w/m, (m-0.5)*self.w/m, m), n)
        self.yb = np.tile(np.linspace(0.5*self.l/n, (n-0.5)*self.l/n, n), m).flatten(order="F")

        self.x_holes = np.repeat(np.linspace(0.0, self.w, m+1), n+1)
        self.y_holes = np.tile(np.linspace(0.0, self.l, n+1), m+1).flatten(order="F")

        self.dxb = np.repeat(np.linspace(0.5*self.dw/m, (m-0.5)*self.dw/m, m), n)
        self.dyb = np.tile(np.linspace(0.5*self.dl/n, (n-0.5)*self.dl/n, n), m).flatten(order="F")

        self.dx_holes = np.repeat(np.linspace(0.0, self.dw, m+1), n+1)
        self.dy_holes = np.tile(np.linspace(0.0, self.dl, n+1), m+1).flatten(order="F")

        self.hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
        self.dhole_r = (ratio+dratio)*0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0) - ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)

        x_border_len = (self.w-2*m*self.hole_r)/m
        y_border_len = (self.l-2*n*self.hole_r)/n
        dx_border_len = cell_d*dextra-2*self.dhole_r
        dy_border_len = cell_d*dextra-2*self.dhole_r

        x_ranges = []
        for i in range(m):
            x_ranges.append((2*i+1)*(self.hole_r+self.dhole_r)+i*(x_border_len+dx_border_len))
            x_ranges.append((2*i+1)*(self.hole_r+self.dhole_r)+(i+1)*(x_border_len+dx_border_len))
        self.x_ranges = x_ranges[:]

        y_ranges = []
        for i in range(n):
            y_ranges.append((2*i+1)*(self.hole_r+self.dhole_r)+i*(y_border_len+dy_border_len))
            y_ranges.append((2*i+1)*(self.hole_r+self.dhole_r)+(i+1)*(y_border_len+dy_border_len))
        self.y_ranges = y_ranges[:]

        return

    def eval_geometry_partials(self):

        m = self.m
        n = self.n
        cell_d = self.cell_d
        extra = self.extra
        ratio = self.ratio
        dextra = self.dextra
        dratio = self.dratio

        self.ddw_ddextra = cell_d*m
        self.ddl_ddextra = cell_d*n

        self.ddxb_ddextra = np.repeat(np.linspace(0.5*self.ddw_ddextra/m, (m-0.5)*self.ddw_ddextra/m, m), n)
        self.ddyb_ddextra = np.tile(np.linspace(0.5*self.ddl_ddextra/n, (n-0.5)*self.ddl_ddextra/n, n), m).flatten(order="F")

        self.ddx_holes_ddextra = np.repeat(np.linspace(0.0, self.ddw_ddextra, m+1), n+1)
        self.ddy_holes_ddextra = np.tile(np.linspace(0.0, self.ddl_ddextra, n+1), m+1).flatten(order="F")

        self.ddr_ddextra = (ratio+dratio)*0.5*cell_d*(2.0**0.5)
        self.ddr_ddratio = 0.5*cell_d*((2.0**0.5*(extra+dextra)) - 1.0)

        ddx_ddextra = cell_d - 2.0*self.ddr_ddextra
        ddy_ddextra = cell_d - 2.0*self.ddr_ddextra
        ddx_ddratio = -2.0*self.ddr_ddratio
        ddy_ddratio = -2.0*self.ddr_ddratio

        dx_ranges_ddextra = []
        dx_ranges_ddratio = []
        for i in range(m):
            dx_ranges_ddextra.append((2*i+1)*self.ddr_ddextra + i*ddx_ddextra)
            dx_ranges_ddextra.append((2*i+1)*self.ddr_ddextra + (i+1)*ddx_ddextra)
            dx_ranges_ddratio.append((2*i+1)*self.ddr_ddratio + i*ddx_ddratio)
            dx_ranges_ddratio.append((2*i+1)*self.ddr_ddratio + (i+1)*ddx_ddratio)

        dy_ranges_ddextra = []
        dy_ranges_ddratio = []
        for i in range(n):
            dy_ranges_ddextra.append((2*i+1)*self.ddr_ddextra + i*ddy_ddextra)
            dy_ranges_ddextra.append((2*i+1)*self.ddr_ddextra + (i+1)*ddy_ddextra)
            dy_ranges_ddratio.append((2*i+1)*self.ddr_ddratio + i*ddy_ddratio)
            dy_ranges_ddratio.append((2*i+1)*self.ddr_ddratio + (i+1)*ddy_ddratio)

        self.dx_ranges_ddextra = dx_ranges_ddextra[:]
        self.dx_ranges_ddratio = dx_ranges_ddratio[:]
        self.dy_ranges_ddextra = dy_ranges_ddextra[:]
        self.dy_ranges_ddratio = dy_ranges_ddratio[:]

        # self.dx_ranges_ddextra = [self.ddr_ddextra, self.ddr_ddextra + ddx_ddextra,
        #                           3.0*self.ddr_ddextra + ddx_ddextra, 3.0*self.ddr_ddextra + 2.0*ddx_ddextra,
        #                           5.0*self.ddr_ddextra + 2.0*ddx_ddextra, 5.0*self.ddr_ddextra + 3.0*ddx_ddextra][:]
        # self.dy_ranges_ddextra = [self.ddr_ddextra, self.ddr_ddextra + ddy_ddextra,
        #                           3.0*self.ddr_ddextra + ddy_ddextra, 3.0*self.ddr_ddextra + 2.0*ddy_ddextra,
        #                           5.0*self.ddr_ddextra + 2.0*ddy_ddextra, 5.0*self.ddr_ddextra + 3.0*ddy_ddextra][:]
        # self.dx_ranges_ddratio = [self.ddr_ddratio, self.ddr_ddratio + ddx_ddratio,
        #                           3.0*self.ddr_ddratio + ddx_ddratio, 3.0*self.ddr_ddratio + 2.0*ddx_ddratio,
        #                           5.0*self.ddr_ddratio + 2.0*ddx_ddratio, 5.0*self.ddr_ddratio + 3.0*ddx_ddratio][:]
        # self.dy_ranges_ddratio = [self.ddr_ddratio, self.ddr_ddratio + ddy_ddratio,
        #                           3.0*self.ddr_ddratio + ddy_ddratio, 3.0*self.ddr_ddratio + 2.0*ddy_ddratio,
        #                           5.0*self.ddr_ddratio + 2.0*ddy_ddratio, 5.0*self.ddr_ddratio + 3.0*ddy_ddratio][:]

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
        m = self.m
        n = self.n
        w = self.w
        l = self.l
        hole_r = self.hole_r

        edge_cp_idx = []  # store a nested list of length 4: [[bottom edge cp nodes], [right edge ""], [top edge ""], [left edge ""]]

        edge_uv_0 = []
        for i in range(m+1):
            edge_uv_0.append([i, 0])

        edge_uv_1 = []
        for i in range(n+1):
            edge_uv_1.append([m, i])

        edge_uv_2 = []
        for i in range(m, -1, -1):
            edge_uv_2.append([i, n])

        edge_uv_3 = []
        for i in range(n, -1, -1):
            edge_uv_3.append([0, i])

        edge_uv = [edge_uv_0, edge_uv_1, edge_uv_2, edge_uv_3]

        xpt_offsets = np.array(m*[1, -1])
        ypt_offsets = np.array(n*[1, -1])
        for i in range(4):
            i_edge_cp_idx = []
            if i%2 == 0:  # bottom or top edge
                jlen = m+1
            else:
                jlen = n+1
            for j in range(jlen):
                [u, v] = edge_uv[i][j]
                if i%2 == 0:  # bottom or top edge
                    x = u*w/m + hole_r*xpt_offsets  # array of x-points to find on this edge
                    y = np.ones(2*m)*v*l/n   # array of y-points to find on this edge
                else:  # left or right edge
                    x = np.ones(2*n)*u*w/m  # array of x-points to find on this edge
                    y = v*l/n + hole_r*ypt_offsets  # array of y-points to find on this edge
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

        hole_deltas = np.zeros((len(self.hole_idx), 2))
        for i, idx in enumerate(self.hole_idx):
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

            elif np.absolute(pt[0] - self.w) < eps:  # right edge
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

            elif np.absolute(pt[1] - self.l) < eps:  # top edge
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
        battery_edge_idx = self.battery_edge_idx
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

        dx_ranges_ddextra = self.dx_ranges_ddextra
        dy_ranges_ddextra = self.dy_ranges_ddextra
        dx_ranges_ddratio = self.dx_ranges_ddratio
        dy_ranges_ddratio = self.dy_ranges_ddratio

        edge_cp_idx = self.edge_cp_idx
        x_cp = np.sort(Xpts0[edge_cp_idx[:], 0])
        y_cp = np.sort(Xpts0[edge_cp_idx[:], 1])

        ddelta_ddratio = np.zeros((len(self.border_idx), 2))
        ddelta_ddextra = np.zeros((len(self.border_idx), 2))
        for i, idx in enumerate(self.border_idx):
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

            elif np.absolute(pt[0] - self.w) < eps:  # right edge
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
                ddelta_ddextra[i, 0] = self.ddw_ddextra

            elif np.absolute(pt[1] - self.l) < eps:  # top edge
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
                ddelta_ddextra[i, 1] = self.ddl_ddextra

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

        return ddelta_ddratio, ddelta_ddextra

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

        # Reshape the node vector before passing it to the mesh deformation component
        Xpts0 = np.delete(Xpts0, np.arange(2, Xpts0.size, 3))
        nnodes = int(Xpts0.size/2)
        Xpts0 = Xpts0.reshape((nnodes, 2))

        # Initialize the mesh deformation object
        self.md = MeshDeformation(Xpts0, m=m, n=n, cell_d=cell_d, extra=extra, ratio=ratio)

        self.add_input("dratio", val=0.0, units=None, desc="change in ratio parameter from the initial mesh")
        self.add_input("dextra", val=0.0, units=None, desc="change in extra parameter from the initial mesh")

        self.add_output("Xpts", shape=(3*nnodes,), units="m", desc="Initial mesh coordinates")

        self.declare_partials(of="Xpts", wrt=["dratio", "dextra"])

    def compute(self, inputs, outputs):

        dratio = inputs["dratio"]
        dextra = inputs["dextra"]
        Xpts = outputs["Xpts"]

        # Update the parameterization
        self.md.dratio = dratio[0]
        self.md.dextra = dextra[0]

        # Deform the geometry
        Xnew = self.md.deform_geometry()
        Xpts[0::3] = Xnew[:, 0]
        Xpts[1::3] = Xnew[:, 1]

    def compute_partials(self, inputs, partials):

        dXpts_ddratio = partials["Xpts", "dratio"]
        dXpts_ddextra = partials["Xpts", "dextra"]

        ddelta_ddratio, ddelta_ddextra = self.md.compute_partials()
        dXpts_ddratio[0::3, 0] = ddelta_ddratio[:, 0]
        dXpts_ddratio[1::3, 0] = ddelta_ddratio[:, 1]
        dXpts_ddextra[0::3, 0] = ddelta_ddextra[:, 0]
        dXpts_ddextra[1::3, 0] = ddelta_ddextra[:, 1]

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create the TACS object to get the initial nodes
    comm = MPI.COMM_WORLD

    # Instantiate FEASolver
    #bdfFile = os.path.join(os.path.dirname(__file__), "battery_ratio_0.4_extra_1.5.bdf")
    bdfFile = os.path.join(os.path.dirname(__file__), "battery_4x4.bdf")
    FEAAssembler = pyTACS(bdfFile, comm)

    # Plate geometry
    tplate = 0.065

    # Material properties
    battery_rho = 1460.0  # density kg/m^3
    battery_kappa = 1.3  # Thermal conductivity W/(m⋅K)
    battery_cp = 880.0  # Specific heat J/(kg⋅K)

    alum_rho = 2700.0  # density kg/m^3
    alum_kappa = 204.0  # Thermal conductivity W/(m⋅K)
    alum_cp = 883.0  # Specific heat J/(kg⋅K)

    def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):

        # Setup property and constitutive objects
        if compDescript == "Block":
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

    # Drop the z-values and reshape the vector
    Xpts0 = np.delete(Xpts0, np.arange(2, Xpts0.size, 3))
    nnodes = int(Xpts0.size/2)
    Xpts0 = Xpts0.reshape((nnodes, 2))

    # Parametrize the initial geometry
    m = 4  # number of rows of battery cells
    n = 4  # number of columns of battery cells
    cell_d = 0.018  # diameter of the cell
    extra = 1.5  # extra space along the diagonal, in [1, infty)
    ratio = 0.4  # cell diameter/cutout diamter, in [0, 1]

    # Define the change in geometric parameters to be applied
    dratio = 0.3
    dextra = 0.3

    # Deform the mesh
    md = MeshDeformation(Xpts0, m=m, n=n, cell_d=cell_d, extra=extra, ratio=ratio, dextra=dextra, dratio=dratio)
    Xnew = md.deform_geometry()

    # Plot the initial and deformed meshes nodes
    s = 2.0
    fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)
    ax0.scatter(Xpts0[:, 0], Xpts0[:, 1], s=s, color="tab:blue")
    ax0.scatter(Xpts0[md.Xpts0_cp_idx[:], 0], Xpts0[md.Xpts0_cp_idx[:], 1], s=s, color="tab:red")
    ax1.scatter(Xnew[:, 0], Xnew[:, 1], s=s, color="tab:blue")

    ax0.set_aspect("equal")
    ax1.set_aspect("equal")
    ax0.axis("off")
    ax1.axis("off")
    ax0.set_title("Original mesh")
    ax1.set_title("Deformed mesh")

    plt.show()

    # Check the OpenMDAO components and the derivatives
    Xpts0 = FEAAssembler.getOrigNodes()
    prob = om.Problem()

    ivc = om.IndepVarComp()
    ivc.add_output("dratio", val=0.0, units=None)
    ivc.add_output("dextra", val=0.0, units=None)
    prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])

    md_comp = MeshDeformComp(m=m, n=n, cell_d=cell_d, extra=extra, ratio=ratio, Xpts0=Xpts0, nnodes=nnodes)
    prob.model.add_subsystem("mesh_deformation", md_comp, promotes_inputs=["dratio", "dextra"], promotes_outputs=["Xpts"])

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
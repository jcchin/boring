import numpy as np

from egads4py import egads

def make_grid_geometry(m=3, n=3, cell_d=0.018, extra=1.1516, ratio=0.7381):
    '''
    Make grid-style battery geometry

    Inputs:
    m:      number of rows of battery cells
    n:      number of columns of battery cells
    cell_d: diameter of the cell
    extra:  extra space along the diagonal
    ratio:  cell diameter/cutout diamter
    '''

    w = cell_d*m*extra # total width
    l = cell_d*n*extra # total length
    t = 0.5*(w+l) # thickness of the 3D block
    hole_r = ratio*0.5*cell_d*((2.0**0.5*extra) - 1.0)
    offset = cell_d*extra

    # Create the outer block
    battery_body = ctx.makeSolidBody(egads.BOX, rdata=[[0.0, 0.0, 0.0], [w, l, t]])

    # Subtract out the holes
    x_holes = np.linspace(0.0, w, m+1)
    y_holes = np.linspace(0.0, l, n+1)
    for i in range(m+1):
        for j in range(n+1):
            x1 = [x_holes[i], y_holes[j], 0.0]
            x2 = [x_holes[i], y_holes[j], t]
            hole = ctx.makeSolidBody(egads.CYLINDER, rdata=[x1, x2, hole_r])
            battery_model = battery_body.solidBoolean(hole, egads.SUBTRACTION)
            battery_body = battery_model.getChildren()[0]

    # Subtract out the batteries
    dx = 1.0/(2.0*m)
    dy = 1.0/(2.0*n)
    for i in range(m):
        for j in range(n):
            x = (float(i)/float(m) + dx)*w
            y = (float(j)/float(n) + dy)*l
            battery = ctx.makeSolidBody(egads.CYLINDER,
                                        rdata=[[x, y, 0.0], [x, y, t], 0.5*cell_d])
            battery_model = battery_body.solidBoolean(battery, egads.SUBTRACTION)
            battery_body = battery_model.getChildren()[0]

    # Add the batteries back in
    for i in range(m):
        for j in range(n):
            x = (float(i)/float(m) + dx)*w
            y = (float(j)/float(n) + dy)*l
            battery = ctx.makeSolidBody(egads.CYLINDER,
                                        rdata=[[x, y, 0.0], [x, y, t], 0.5*cell_d])
            battery_model = battery_body.solidBoolean(battery, egads.FUSION)
            battery_body = battery_model.getChildren()[0]

    return battery_model

# Create the egads context
ctx = egads.context()

# Create the egads battery model and save it as a step file
battery_model = make_grid_geometry(m=3, n=3, ratio=0.4, extra=1.5)
battery_model.saveModel('battery_3d.step', overwrite=True)
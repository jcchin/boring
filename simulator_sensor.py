cell_w = 0.059*2. #m , (2.25")
cell_l = 0.0571 #m , (2.0")
cell_h = 0.00635 #m , (0.25")

# Render in OpenSCAD using the OpenPySCAD library
import openpyscad as ops

cell = ops.Cube([cell_w, cell_l, cell_h]).color("yellow") # Amprius Cell
heater = ops.Cylinder(h=0.11, r=0.003,  _fn=100).color("orange")
heater2 = ops.Cube([cell_w-0.002, cell_l-0.002, cell_h/5]).color("orange") # Amprius Cell

sim = ops.Union()
# module = ops.Union()
d = ops.Difference()

sim.append(heater.rotate([0, 90, 0]).translate([0.005,0.014,0.003]))
sim.append(heater.rotate([0, 90, 0]).translate([0.005,0.028,0.003]))
sim.append(heater.rotate([0, 90, 0]).translate([0.005,0.042,0.003]))
sim.append(cell.translate([0, 0, 0])) # simulator cell
sim.append(heater2.translate([0.001,cell_l*1.1+0.001,0.003]))
sim.append(cell.translate([0, cell_l*1.1, 0])) # sensor cell
    


(sim).write("sim_sens.scad")


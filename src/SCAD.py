# Render in OpenSCAD using the OpenPySCAD library
import openpyscad as ops



t_PCM_m = t_PCM/1000

cell = ops.Cube([cell_w, cell_l, cell_h]) # Amprius Cell
pcm = ops.Cube([cell_w, cell_l, t_PCM_m]).color("Orange") # Phase Change Material
ohp = ops.Cube([cell_w, cell_l*n_cpm*s_h, cell_h]).color("Gray") # Oscillating Heat Pipe Straight-away
ohp_turn = ops.Cylinder(h=cell_h, r=cell_w*s_h2, _fn=100).color("Gray") # OHP End Turn
ohp_turn_d = ops.Cube([cell_w*2*s_h,cell_w*s_h,cell_h+0.02]) # Make it a semi-circle by subtracting a rectangle, plus a little height to avoid green
pack = ops.Union()
module = ops.Union()
d = ops.Difference()
d2 = ops.Difference()
insulation = ops.Cube([cell_w*2,cell_l*s_h*n_cpm*1.1,cell_h*s_v]).color("Blue")

stack_h = cell_h*2 + t_PCM_m*2

for b in range(n_cpm):
    # Cell Array
    module.append(cell.translate([0, cell_l*s_h*b, 0])) # first row cell
    module.append(cell.translate([0, cell_l*s_h*b, stack_h])) # second column, first row
    module.append(cell.translate([cell_w*s_h, cell_l*s_h*b, 0])) # second row cell
    module.append(cell.translate([cell_w*s_h, cell_l*s_h*b, stack_h])) # second column, second row cell
    # PCM Array
    module.append(pcm.translate([0, cell_l*s_h*b, cell_h])) # first row PCM
    module.append(pcm.translate([0, cell_l*s_h*b, stack_h-t_PCM_m])) # second column, first row
    module.append(pcm.translate([cell_w*s_h, cell_l*s_h*b, cell_h])) # second row cell
    module.append(pcm.translate([cell_w*s_h, cell_l*s_h*b, stack_h-t_PCM_m])) # second column, second row cell

# OHP
module.append(ohp.translate([0,0,cell_h+t_PCM_m]))
module.append(ohp.translate([cell_w*s_h,0,cell_h+t_PCM_m]))
d.append(ohp_turn.translate([cell_w*s_h2,0,stack_h/2]))
d.append(ohp_turn_d.translate([0,0,stack_h/2-0.01]))
module.append(d)

# Insulation
d2.append(insulation.translate([0,-0.03,0.017]))
for b in range(n_cpm): #subtract cells
    d2.append(cell.translate([-0.005, cell_l*s_h*b, stack_h*1.3])) # first row cell
    d2.append(cell.translate([-0.005+cell_w*s_h, cell_l*s_h*b, stack_h*1.3])) # second column, first row

for s in range(n_stacks_show):
    # Stack Array
    for m in range(n_modules):
        # Module Array
        pack.append(module.translate([cell_w*s_h*2*m, 0, stack_h*s_v*s]))
pack.append(d2.translate([-0.4,0,0]))
pack.write("PCM.scad")
d2.write("Insulation.scad")
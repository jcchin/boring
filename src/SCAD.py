# Render in OpenSCAD using the OpenPySCAD library
import openpyscad as ops
import openmdao.api as om
import numpy as np
import os.path as path

case_name = 'geometry.sql'

cr = om.CaseReader(case_name)
model_vars = cr.list_source_vars('problem', out_stream=None)
case = cr.get_cases()[0]
print(('inputs:', sorted(model_vars['inputs']), 'outputs:', sorted(model_vars['outputs'])))

t_PCM = float(case.get_val('t_PCM', units='mm'))
cell_w = float(case.get_val('cell_w'))
cell_l = float(case.get_val('cell_l'))
cell_h = float(case.get_val('cell_h'))
s_w = float(case.get_val('cell_s_w'))
s_h = float(case.get_val('cell_s_h'))
n_cpbl = int(np.ceil(case.get_val('n_cpb'))/2.)
n_bps = int(np.ceil(case.get_val('n_bps')))


s_h = 1.1 # horizontal spacing
s_v = 1.3 # vertical spacing
s_h2 = ((s_h-1)/2 +1)
n_stacks_show = 1

cell = ops.Cube([cell_w, cell_l, cell_h]) # Amprius Cell
pcm = ops.Cube([cell_w, cell_l, t_PCM]).color("Orange") # Phase Change Material
ohp = ops.Cube([cell_w, cell_l*n_cpb*s_h, cell_h]).color("Gray") # Oscillating Heat Pipe Straight-away
# ohp_turn = ops.Cylinder(h=cell_h, r=cell_w*s_h2, _fn=100).color("Gray") # OHP End Turn
# ohp_turn_d = ops.Cube([cell_w*2*s_h,cell_w*s_h,cell_h+0.02]) # Make it a semi-circl
#pack = ops.Union()
bar = ops.Union()
# d = ops.Difference()
# d2 = ops.Difference()
# insulation = ops.Cube([cell_w*2,cell_l*s_h*n_cpb*1.1,cell_h*s_v]).color("Blue")

stack_h = cell_h*2 + t_PCM*2

for b in range(n_cpbl):
    # Cell Array
    bar.append(cell.translate([0, cell_l*s_h*b, 0])) # first row cell
    bar.append(cell.translate([0, cell_l*s_h*b, stack_h])) # second column, first row
    bar.append(cell.translate([cell_w*s_h, cell_l*s_h*b, 0])) # second row cell
    bar.append(cell.translate([cell_w*s_h, cell_l*s_h*b, stack_h])) # second column, second row cell
    # PCM Array
    bar.append(pcm.translate([0, cell_l*s_h*b, cell_h])) # first row PCM
    bar.append(pcm.translate([0, cell_l*s_h*b, stack_h-t_PCM])) # second column, first row
    bar.append(pcm.translate([cell_w*s_h, cell_l*s_h*b, cell_h])) # second row cell
    bar.append(pcm.translate([cell_w*s_h, cell_l*s_h*b, stack_h-t_PCM])) # second column, second row cell

# OHP
bar.append(ohp.translate([0,0,cell_h+t_PCM]))
#bar.append(ohp.translate([cell_w*s_h,0,cell_h+t_PCM]))
# bar.append(d)

# # Insulation
# d2.append(insulation.translate([0,-0.03,0.017]))
# for b in range(n_cpb): #subtract cells
#     d2.append(cell.translate([-0.005, cell_l*s_h*b, stack_h*1.3])) # first row cell
#     d2.append(cell.translate([-0.005+cell_w*s_h, cell_l*s_h*b, stack_h*1.3])) # second column, first row

# for s in range(n_stacks_show):
#     # Stack Array
#     for m in range(n_bps):
#         # Module Array
#         pack.append(bar.translate([cell_w*s_h*2*m, 0, stack_h*s_v*s]))
# pack.append(d2.translate([-0.4,0,0]))
bar.write('../scad/PCM.scad')
# d2.write('../scad/Insulation.scad')
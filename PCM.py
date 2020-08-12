
#
#                    Vent 
#  --------- --------- --------- --------- 
# | Battery | Battery | Battery | Battery |
#  --------- --------- --------- ---------
# |   PCM   |   PCM   |   PCM   |   PCM   |
#  --------- --------- --------- ---------
# <        Oscillating Heat Pipe          > ...
# ---------------------------------------
# |   PCM   |   PCM   |   PCM   |   PCM   |
# --------- --------- --------- ---------
# | Battery | Battery | Battery | Battery |
#  --------- --------- --------- ---------
#                   Vent


# Starting with copper foam and Eicosane
n_modules = 1 # arches
n_cpm = 8 # cells per module
n_stacks = 40 # stacks
n_stacks_show = 3
s_h = 1.1 # horizontal spacing
s_v = 1.3 # vertical spacing
s_h2 = ((s_h-1)/2 +1)

n_cells = n_modules*n_cpm*n_stacks*4 # number of prismatic cells
frame_mass = 10 # grams, frame mass per cell
k_pcm = 3.06 # W/m*K, Conductivity of Eicosane with copper foam
LH = 190 # J/g, Latent Heat of Eicosane
rho_pcm = 900 #kg/m^3
melt = 60 # degC, Metling Point of Eicosane
missionJ = 1200 #J, Energy rejected in a normal mission
runawayJ = 48000 # J, Runaway heat of a 18650 battery
frac_absorb = 1.0 # fraction of energy absorbed during runaway
dur = 45 # seconds, of runaway duration
v_n_c = 3.4 #  nominal voltage
q_max = 3.5*2. # A-h cells
cell_mass = 31.6*2. #g, cell mass Dimensions: 57mm x 50mm x 6.35mm
cell_w = 0.059*2. #m , (2.25")
cell_l = 0.0571 #m , (2.0")
cell_h = 0.00635 #m , (0.25")
cell_area = cell_w*cell_l # Dimensions: 2.25" x 2" x 0.25"
ext_cooling = 0 # W, external cooling
ext_cool_mass = 0 # g, mass of external cooling
#mass_OHP = 26 # g,  #2.5mm x 50mm x 250mm (< 270W) 0.2C/W
#https://amecthermasol.co.uk/datasheets/MHP-2550A-250A.pdf

mass_PCM = n_modules*n_stacks*runawayJ*frac_absorb/LH + missionJ/LH - ext_cooling
t_PCM = mass_PCM/(rho_pcm*n_cells*cell_area)
flux = runawayJ/dur #heat flux during runaway
Areal_weight = 0.3866*flux+1.7442 # NH3   kg/m^2
# Areal_weight = 0.4922*flux+1.5599 # Acetone
# Areal_weight = 0.7805*flux+1.9131 # C4H10
mass_OHP = Areal_weight*cell_area*n_cells/2

p_mass = mass_PCM + mass_OHP + frame_mass*n_cells + ext_cool_mass
tot_mass = p_mass + cell_mass*n_cells 

mass_frac = p_mass/tot_mass
energy = n_cells * q_max * v_n_c


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

print("num of cells: ", n_cells)
print("flux: ", flux)
print("PCM mass: ", mass_PCM)
print("PCM thickness (mm): ", t_PCM)
print("OHP mass: ", mass_OHP)
print("packaging mass: ", p_mass)
print("total mass: ", tot_mass)
print("package mass fraction: ", mass_frac)
print("pack energy density: ", energy/(tot_mass/1000.))
print("cell energy density: ", (q_max * v_n_c) / (cell_mass/1000.))
print("pack energy (kWh): ", energy/1000.)
print("pack cost ($K): ", n_cells*0.4)
print("overall pack dimensions: %.3f  ft x %.3f ft x %.3f ft" % (cell_l*s_h*n_cpm*3.28, stack_h*s_v*n_stacks*3.28, cell_w*s_h*2*3.28))


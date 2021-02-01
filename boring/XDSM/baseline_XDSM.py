from pyxdsm.XDSM import XDSM

# styling names for the boxes
opt = 'Optimization' # blue pill
subopt = 'SubOptimization' 
solver = 'MDA' # orange pill
doe = 'DOE'
ifunc = 'ImplicitFunction'
func = 'Function' # green box
group = 'Group'
igroup = 'ImplicitGroup'
metamodel = 'Metamodel'

x = XDSM()

# Create subsystem components
# spec_name = False (don't write a spec)
# spec_name = 'xyz' (changes the spec name from defalt 'e_comp' to 'xyz')
x.add_system('Sweep', solver, ['Sweep'], spec_name=False)
x.add_system('Optimizer', opt, ['Optimizer'], spec_name=False)
x.add_system('Geometry', func, ['Geometry'])
x.add_system('Thermal', func, ['Thermal'])
x.add_system('Structural', func, ['Structural'])


# Optimizer
x.add_input('Optimizer','constraints_{temp,stress}')
x.connect('Optimizer','Geometry', ['extra','ratio','cell_d'])


# Geometry
# x.add_input('pack_design', ['L_{pack}', 'W_{pack}','L_{cell}', 'W_{cell}', 'H_{cell}',
#             'mass_{cell}','voltage_{low,cell}','voltage_{nom,cell}','dischargeRate_{cell}','Q_{max}','V_{batt}'])
x.connect('Geometry', 'Thermal', ['dimensions'])
x.connect('Geometry', 'Structural','dimensions')
x.connect('Geometry', 'Optimizer','mass')
# x.add_output('pack_design', ['n_{series}','n_{parallel}'], side='right')

# Thermal
x.connect('Thermal', 'Optimizer','temp')

# Geometry
# x.add_input('heat_pipe', ['d_{init}','rho_{HP}', 'L_{pack}'])
x.connect('Geometry', 'Structural', 'mesh')
x.connect('Geometry', 'Thermal', 'mesh')
x.connect('Geometry', 'Sweep', r'\frac{Wh}{kg}')

# Structural
x.add_input('Structural','Loads')
x.connect('Structural', 'Optimizer', 'stress')

# Sweep
x.connect('Sweep','Geometry','properties')
x.connect('Sweep','Thermal','energy')
# x.add_output('Struct', ['mass_{battery}'], side='right')


x.write('baseline_XDSM')

# x.write_sys_specs('Design_specs')
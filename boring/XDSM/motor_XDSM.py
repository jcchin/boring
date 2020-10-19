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
#x.add_system('Optimizer', opt, ['Optimizer'], spec_name=False)
x.add_system('ElectroMag', func, ['ElectroMag'])
x.add_system('Thermal', func, ['Thermal'])
x.add_system('Geometry', func, ['Geometry'])
x.add_system('Balance', solver, ['Balance'], spec_name=False)


# Optimizer
# x.connect('Optimizer','pack_design', ['energy_{required}','eta_{batt}','I_{batt}'])


# ElectroMag
# x.add_input('pack_design', ['L_{pack}', 'W_{pack}','L_{cell}', 'W_{cell}', 'H_{cell}',
#             'mass_{cell}','voltage_{low,cell}','voltage_{nom,cell}','dischargeRate_{cell}','Q_{max}','V_{batt}'])
x.connect('ElectroMag', 'Thermal', ['torque'])
x.connect('ElectroMag', 'Geometry','gap_{fields}')
# x.add_output('pack_design', ['n_{series}','n_{parallel}'], side='right')

# Thermal
x.connect('Thermal', 'Geometry','copper_{loss}')

# Geometry
# x.add_input('heat_pipe', ['d_{init}','rho_{HP}', 'L_{pack}'])
x.connect('Geometry', 'ElectroMag', 'size')
x.connect('Geometry', 'Thermal', 'mass')
x.connect('Geometry', 'Balance', 'size')

# Balance
x.connect('Balance','Geometry','radius_{motor}')
# x.add_output('Struct', ['mass_{battery}'], side='right')


x.write('Motor_XDSM')

# x.write_sys_specs('Design_specs')
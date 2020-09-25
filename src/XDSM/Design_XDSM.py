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
x.add_system('Optimizer', opt, ['Optimizer'], spec_name=False)
x.add_system('e_solver', solver, ['Solver'], spec_name=False)
x.add_system('cell', func, ['cell'])
x.add_system('PCM', func, ['PCM'])
x.add_system('HP', func, ['HP'])
x.add_system('Struct', func, ['Struct'])
x.add_system('TMS', func, ['TMS'])

x.connect('Optimizer','cell', [r'energy_{required}',r'eta_{batt}','I_{batt}'])


# Cell
x.add_input('cell', ['mass_{cell}','voltage_{low,cell}','voltage_{nom,cell}','dischargeRate_{cell}','Q_{max}','V_{batt}'])
x.connect('cell', 'PCM', ['n_{series}','n_{parallel}'])
x.connect('cell', 'Struct','mass_{cell}')
x.add_output('cell', ['n_{series}','n_{parallel}'], side='right')

# PCM
x.connect('PCM', 'Struct','mass_{PCM}')

# HP
x.connect('HP', 'Struct','mass_{HP}')



# Connect Battery outputs to subsystems
x.add_output('Struct', ['mass_{battery}'], side='right')

x.write('Design_XDSM')

x.write_sys_specs('Design_specs')
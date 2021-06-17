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
x.add_system('Thermal', func, ['Thermal (Transient)'])
x.add_system('Structural', func, ['Structural'])


# Optimizer
x.add_input('Optimizer','constraints_{temp,stress}')
x.connect('Optimizer','Geometry', ['spacing','ratio'])


# Geometry
x.add_input('Geometry', ['D_{cell}', 'Core_{props}'])
x.connect('Geometry', 'Thermal', ['dimensions'])
x.connect('Geometry', 'Structural','dimensions')
x.connect('Geometry', 'Optimizer','mass')
# x.add_output('pack_design', ['n_{series}','n_{parallel}'], side='right')

# Thermal
x.add_input('Thermal', ['h_{boundary}','R_{contact}'])
x.connect('Thermal', 'Optimizer','temp')

# Geometry
# x.add_input('heat_pipe', ['d_{init}','rho_{HP}', 'L_{pack}'])
x.connect('Geometry', 'Structural', 'mesh')
x.connect('Geometry', 'Thermal', 'mesh')
x.connect('Geometry', 'Sweep', r'pack \frac{Wh}{kg}')

# Structural
x.add_input('Structural','loads_{mech}')
x.connect('Structural', 'Optimizer', 'stress')

# Sweep
# x.connect('Sweep','Geometry','properties')
x.connect('Sweep','Thermal',r'load_{heat}=f(cell \frac{Wh}{kg})')
# x.add_output('Struct', ['mass_{battery}'], side='right')


x.write('baseline_XDSM')

# x.write_sys_specs('Design_specs')
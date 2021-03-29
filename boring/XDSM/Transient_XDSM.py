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
x.add_system('dymos', solver, ['Dymos'], spec_name=False)
x.add_system('battery', func, ['Battery'], spec_name=False, stack=True)
x.add_system('PCM', func, ['PCM'], spec_name=False, stack=True)
x.add_system('heat_pipe', func, ['Heat pipe'], spec_name=False)

# Dymos
#x.connect('dymos','Optimizer','T_{neighbor}')
x.connect('dymos','heat_pipe','T_{wall}', stack=True)
x.connect('dymos','PCM','T_{pcm}', stack=True)
x.connect('dymos','battery','T_{cell}', stack=True)

# Battery
x.add_input('battery', ['L_{cell}', 'W_{cell}', 'H_{cell}','rho/cp_{cell}','q_{in,cell}'])
x.connect('battery', 'PCM', ['q_{in,pcm}'], stack=True)
x.connect('battery','dymos','dT_{cells}/dt', stack=True)
# x.add_output('battery', ['n_{series}','n_{parallel}'], side='right')

# PCM
x.add_input('PCM',['t_{pad}, A_{pad}', 'porosity', 'rho/LH/K/cp_{foam/pcm}','T_{hi/lo}'])
x.connect('PCM', 'dymos', 'dT_{pcm}/dt', stack=True)
x.connect('PCM','heat_pipe', 'q_{in,hp}', stack=True)
x.connect('PCM','battery', 'q_{out,pcm}', stack=True)
x.add_output('PCM', ['PS'], side='right', stack=True)


# HP
x.add_input('heat_pipe', ['geometry (round/flat)','num_{cells}'])
x.connect('heat_pipe', 'PCM','q_{out,hp}', stack=True)
x.connect('heat_pipe', 'dymos','dT_{wall}/dt', stack=True)


x.write('Transient_XDSM')

x.write_sys_specs('Transient_specs')
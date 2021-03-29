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
x.add_system('e_solver', solver, ['Dymos'], spec_name=False)
x.add_system('pack_design', func, ['Pack'])
x.add_system('PCM', func, ['PCM'])
x.add_system('heat_pipe', func, ['Heat pipe'])
x.add_system('Struct', func, ['Struct'])


# Optimizer
x.connect('Optimizer','pack_design', ['energy_{required}','eta_{batt}','I_{batt}'])
x.connect('Optimizer','heat_pipe', ['D_{hp}, (W_{hp})','t_{wall}','t_{wick}','epsilon_{hp}','L_{hp}'])
x.connect('Optimizer','PCM', ['porosity_{pcm}'])

# Pack Size
x.add_input('pack_design', ['L_{pack}', 'W_{pack}','L_{cell}', 'W_{cell}', 'H_{cell}',
            'mass_{cell}','voltage_{low,cell}','voltage_{nom,cell}','dischargeRate_{cell}','Q_{max}','V_{batt}'])
x.connect('pack_design', 'PCM', ['q_{cell}'])
x.connect('pack_design', 'Struct','mass_{cell}')
x.connect('pack_design','e_solver','dT_{cells}/dt')
x.add_output('pack_design', ['n_{series}','n_{parallel}'], side='right')

# PCM
x.add_input('PCM',['rho_{cond}','rho_{pcm}','n_{cpk}','n_{kps}'])
x.connect('PCM', 'pack_design', 't_{pcm}')
x.connect('PCM', 'Struct','mass_{pcm}')
x.connect('PCM', 'e_solver', 'dT_{pcm}/dt')
x.connect('PCM','heat_pipe', 'q_{in,hp}')

# Dymos
x.connect('e_solver','Optimizer','T_{neighbor}')
x.connect('e_solver','heat_pipe','T_{wall}')
x.connect('e_solver','PCM','T_{pcm}')
x.connect('e_solver','pack_design','T_{cell}')

# HP
x.add_input('heat_pipe', ['rho_{HP}', 'L_{pack}','req_flux','ref_len'])
x.connect('heat_pipe', 'Struct','mass_{HP}')
x.connect('heat_pipe', 'pack_design','t_{HP}')
x.connect('heat_pipe', 'e_solver','dT_{wall}/dt')

# Struct
x.connect('Struct','pack_design','t_{wall}')
x.connect('Struct','Optimizer','mass_{battery}')
#x.add_output('Struct', ['mass_{battery}'], side='right')


x.write('Design_XDSM')

x.write_sys_specs('Design_specs')
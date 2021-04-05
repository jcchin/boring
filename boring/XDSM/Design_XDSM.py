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
x.add_system('case', func, ['Case'], spec_name=False)
# x.add_system('battery', func, ['Battery'], spec_name=False)
x.add_system('PCM', func, ['PCM'], spec_name=False)
x.add_system('heat_pipe', func, ['Heat pipe'], spec_name=False)
x.add_system('pack', func, ['Pack'], spec_name=False)

x.add_system('transient', func, ['Temp Transient'], spec_name=False)



# Optimizer
#x.connect('Optimizer','pack', ['energy_{required}','eta_{batt}','I_{batt}'])
x.add_input('Optimizer',['T_{neighbor,limit}','Q_{runaway}'])
x.connect('Optimizer','case',['t_{case}'])
x.connect('Optimizer','heat_pipe', ['D_{hp}, (W_{hp})','t_{wall}','t_{wick}','t_{case}','epsilon_{hp}'])
x.connect('Optimizer','transient', ['D_{hp}, (W_{hp})','t_{wall}','t_{wick}','epsilon_{hp}'])
x.connect('Optimizer','PCM', ['t_{pcm}','porosity_{pcm}'])
x.connect('Optimizer','pack',['t_{case}'])

# Case
x.add_input('case',['L_{cell}', 'W_{cell}', 'H_{cell}'])
x.connect('case','pack',['mass_{case}'])
#x.add_output('case', ['mass_{battery}'], side='right')

# Battery
# x.connect('battery', 'pack',['mass_{cell}','A_{cell}','t_{cell}'])
# x.connect('battery', 'transient',['mass_{cell}','A_{cell}'])


# PCM
x.add_input('PCM',['rho_{foam}','rho_{pcm}','n_{cpk}','n_{kps}'])
x.connect('PCM', 'pack',['mass_{pcm}'])
x.connect('PCM','transient',['mass_{pcm}'])

# HP
x.add_input('heat_pipe', ['geometry (round/flat)'])
x.connect('heat_pipe', 'pack','mass_{HP}')
x.connect('heat_pipe','transient',['L_{hp}'])


# Pack Size
# x.add_input('pack', ['L_{pack}', 'W_{pack}','L_{cell}', 'W_{cell}', 'H_{cell}',
#             'mass_{cell}','voltage_{low,cell}','voltage_{nom,cell}','dischargeRate_{cell}','Q_{max}','V_{batt}'])
x.add_input('pack', ['num_{cells}','mass_{cell}'])
x.connect('pack','Optimizer',['mass_{pack}','vol_{pack}'])
x.connect('pack','transient',['num_{cells}'])
x.add_output('pack', ['n_{series}','n_{parallel}'], side='right')


# Dymos
x.add_input('transient',['geometry (round/flat)'])
x.connect('transient','Optimizer','T_{neighbor}')

x.write('Design_XDSM')

x.write_sys_specs('Design_specs')
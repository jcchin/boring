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
# spec_name = 'happy' (changes the spec name from defalt 'e_comp' to 'happy')
x.add_system('DYMOS', opt, ['DYMOS'], spec_name=False)
x.add_system('e_solver', solver, ['Solver'], spec_name=False)
x.add_system('e_comp', func, ['Zappy'])
x.add_system('battery', func, ['Battery'])
x.add_system('TMS', func, ['TMS'])


# Connect Dymos outputs to subsystems
x.connect('DYMOS', 'battery', ['SOC', 'V_{thev}','Q_{fire}','PCM_{sat}'])

# Connect Solver states to subsystems
x.connect('e_solver', 'e_comp', '')
x.connect('e_solver', 'battery', '')

# Connect Zappy outputs to subsystems
x.connect('e_comp', 'battery', 'I_{batt}')
x.add_output('e_comp', 'Q_{lines}', side='right') # output to thermal phase
# x.add_output('e_comp', r'C_{max}', side='right')

# Connect Battery outputs to subsystems
x.add_input('battery', ['n_{series}','n_{parallel}','Q_{max}'])
x.connect('battery', 'DYMOS', ['dXdt:SOC', 'dXdt:V_{thev}','dXdt:PCM_{sat}'])
x.connect('battery', 'e_comp', 'V_{batt,actual}')
x.connect('battery', 'e_solver', '')
x.connect('battery', 'TMS', ['T_{cold}'])
x.add_output('battery', ['T_{batt}'], side='right')

x.add_input('TMS', ['A_{cold}'])
x.connect('TMS','battery',['Q_{rej}'])


# # Connect Thermal outputs to subsystems
# x.add_input('tms_comp', [r'W_{coolant}', 'mass_{res,coolant}',r'mass_{motor}',r'mass_{battery}',r'width_{ACC}',r'height_{ACCc}',r'height_{ACCa}',r'Area_{throat}'])
# x.connect('tms_comp', 'DYMOS', ['dXdt:T_{coolant}','dXdt:T_{batt}','dXdt:T_{motor}'])
# x.add_output('tms_comp', 'Power_{TMS}', side='right')
# x.add_output('tms_comp', r'T_{coolant}', side='right')
# #x.connect('tms_comp', 'drag_comp', 'D_{cool}')

x.write('ODE_XDSM')

x.write_sys_specs('ODE_specs')


# General form for promotions
# phase.add_parameter('YOUT_VAR_NAME', targets='YOUR_VAR_NAME', units='YOUR_VAR_UNITS', dynamic=False)
    # Note: dynamic = False takes a single static value and fanns it out to all nodes in the phase
# problem.model.promotes(traj.name, inputs=[(f'{phase.name}.parameters:'+'YOUR_VAR_NAME','YOUR_VAR_NAME')])

def battery_setup(traj, phase, problem):

    phase.add_parameter('n_{series}', targets='n_{series}', units=None, dynamic=True) 
    phase.add_parameter('n_{parallel}', targets='n_{parallel}', units=None, dynamic=True) 
    phase.add_parameter('Q_{max}', targets='Q_{max}', units='A*h', dynamic=False)

    problem.model.promotes(traj.name, inputs=[(f'{phase.name}.parameters:'+'n_{series}','n_{series}')])
    problem.model.promotes(traj.name, inputs=[(f'{phase.name}.parameters:'+'n_{parallel}','n_{parallel}')])
    problem.model.promotes(traj.name, inputs=[(f'{phase.name}.parameters:'+'Q_{max}','Q_{max}')])

    # note: T_{batt} is created in the thermal phase based on weather that phase is on or off

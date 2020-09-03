from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om
import dymos as dm
from dymos.examples.plotting import plot_results


class tempODE(om.ExplicitComponent):

    states = {'T': {'rate_source': 'Tdot', 'units': 'K'}}
    parameters = {'d': {'units': 'm'}}

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('K', val=0.03*np.ones(nn), desc='insulation conductivity', units='W/m*K') #static
        self.add_input('A', val=.102*.0003*np.ones(nn), desc='area', units='m**2') #static
        self.add_input('d', val=0.03*np.ones(nn), desc='insulation thickness', units='m') #static
        self.add_input('m', val=0.06*np.ones(nn), desc='cell mass', units='kg') #static
        self.add_input('Cp', val=3.56*np.ones(nn), desc='specific heat capacity', units='kJ/kg*K') #static
        self.add_input('Th', val=773.*np.ones(nn), desc='hot side temp', units='K') #static
        self.add_input('T', val=293.*np.ones(nn), desc='cold side temp', units='K')

        # Outputs
        self.add_output('Tdot', val=np.zeros(nn), desc='temp rate of change', units='K/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])
        c = np.zeros(self.options['num_nodes'])
        self.declare_partials(of='Tdot', wrt='K', rows=arange, cols=arange) #static
        self.declare_partials(of='Tdot', wrt='A', rows=arange, cols=arange) #static
        self.declare_partials(of='Tdot', wrt='m', rows=arange, cols=arange) #static
        self.declare_partials(of='Tdot', wrt='Cp', rows=arange, cols=arange) #static
        self.declare_partials(of='Tdot', wrt='Th', rows=arange, cols=arange) #static
        self.declare_partials(of='Tdot', wrt='d', rows=arange, cols=arange)
        self.declare_partials(of='Tdot', wrt='T', rows=arange, cols=arange)

    def compute(self, i, o):

        dT_num = i['K']*i['A']*(i['Th']-i['T'])/i['d']
        dT_denom = i['m']*i['Cp']
        o['Tdot'] = dT_num/dT_denom

    def compute_partials(self, i, partials):

        partials['Tdot','T'] = -i['K']*i['A']/(i['d']*i['m']*i['Cp'])
        partials['Tdot','d']  = i['K']*i['A']*(i['Th']-i['T'])/(i['m']*i['Cp']*i['d']**2)
        partials['Tdot','K']  = i['A']*(i['Th']-i['T'])/(i['d']*i['m']*i['Cp'])
        partials['Tdot','A']  = i['K']*(i['Th']-i['T'])/(i['d']*i['m']*i['Cp'])
        partials['Tdot','m']  = i['K']*i['A']*(i['Th']-i['T'])/(i['d']*i['Cp']*i['m']**2)
        partials['Tdot','Cp'] = i['K']*i['A']*(i['Th']-i['T'])/(i['d']*i['m']*i['Cp']**2)



p = om.Problem(model=om.Group())
p.driver = om.ScipyOptimizeDriver()
p.driver.declare_coloring()

traj = p.model.add_subsystem('traj', dm.Trajectory())

phase = traj.add_phase('phase0',
                       dm.Phase(ode_class=tempODE,
                                transcription=dm.GaussLobatto(num_segments=10)))

phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(45, 45))

phase.add_state('T', rate_source=tempODE.states['T']['rate_source'],
                units=tempODE.states['T']['units'],
                fix_initial=True, fix_final=True, solve_segments=False)

phase.add_boundary_constraint('T', loc='final', units='K', upper=333, lower=293, shape=(1,))
phase.add_parameter('d', val=0.001, opt=True, lower=0.001, upper=0.5, units='m')
phase.add_objective('T', loc='final',scaler=1)
p.model.linear_solver = om.DirectSolver()
p.setup()
p['traj.phase0.t_initial'] = 0.0
p['traj.phase0.states:T'] = phase.interpolate(ys=[293, 333], nodes='state_input')
dm.run_problem(p)
exp_out = traj.simulate()
plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:T','time (s)','temp (K)')],
             title='Temps', p_sol=p, p_sim=exp_out)
plt.show()





# t = np.linspace(0, 45, 101)


# def dT_calc(Ts,t):

#     Tf = Ts[0]
#     T = Ts[1]
#     K = 0.03 # W/mk
#     A = .102*.0003 # m^2
#     d = 0.003 #m
#     m = 0.06 #kg
#     Cp = 3.58 #kJ/kgK

#     dT_num = K*A*(Tf-T)/d
#     dT_denom = m*Cp

#     return [0, (dT_num/dT_denom)]


# y0 = [900, 20]
# sol = odeint(dT_calc, y0, t)

# print(sol[100,1])


# #plt.plot(t, sol[:, 0], 'b', label='hot')
# plt.plot(t, sol[:, 1], 'g', label='cold')
# plt.legend(loc='best')
# plt.xlabel('t')
# plt.grid()
# plt.show()
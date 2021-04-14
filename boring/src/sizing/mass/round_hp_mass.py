"""Author: Dustin Hall """
import numpy as np
import openmdao.api as om

class roundHPmass(om.ExplicitComponent):
    ''' Class to calculate only the mass of the heat pipe '''

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('D_od',val=0.03*np.ones(nn), desc='heatpipe outer diameter', units='m')
        self.add_input('D_v',val=0.03*np.ones(nn), desc='vapor outer diameter', units='m')
        self.add_input('L_heatpipe',val=0.3*np.ones(nn), desc='Total length of heatpipe', units='m')
        self.add_input('t_w', val=.0005*np.ones(nn), desc='wall thickness of heatpipe', units='m')
        self.add_input('t_wk', val=0.0005*np.ones(nn), desc='wick thickness', units='m')
        self.add_input('cu_density',val=8960*np.ones(nn), desc='density of aluminum', units='kg/m**3')
        self.add_input('fill_wk', val=0.10*np.ones(nn), desc='fill factor of the wick')
        self.add_input('liq_density', val=1000*np.ones(nn), desc='density of the heat pipe liquid', units='kg/m**3')
        self.add_input('fill_liq', val=0.85*np.ones(nn), desc='fill factor for liquid inside heat pipe')

        self.add_output('mass_heatpipe', val=1.0 * np.ones(nn), desc='mass of the heat pipe', units='kg')
        self.add_output('mass_wick', val=1.0 * np.ones(nn), desc='mass of the heat pipe', units='kg')
        self.add_output('mass_liquid', val=1.0 * np.ones(nn), desc='mass of the heat pipe', units='kg')

    def compute(self, i, o):
        L_heatpipe = i['L_heatpipe']
        D_od = i['D_od']
        t_w = i['t_w']
        D_v  = i['D_v']
        cu_density = i['cu_density']
        t_wk = i['t_wk']
        fill_wk = i['fill_wk']
        liq_density = i['liq_density']
        fill_liq = i['fill_liq']

        o['mass_heatpipe'] = L_heatpipe*cu_density* np.pi*( (D_od/2)**2 - (D_od/2 - t_w)**2 )
        o['mass_liquid'] = L_heatpipe*liq_density*fill_liq* np.pi/4*( D_v**2 )
        o['mass_wick'] = L_heatpipe*cu_density*fill_wk* np.pi*( (D_v/2+t_wk)**2 - (D_v/2)**2 ) # this can also be defined as fun(D_od, t_w)

    def setup_partials(self):
        nn=self.options['num_nodes']
        ar = np.arange(nn) 

        self.declare_partials('mass_heatpipe', ['L_heatpipe', 'D_od', 't_w', 'cu_density'], rows=ar, cols=ar)
        self.declare_partials('mass_liquid',['L_heatpipe', 'liq_density', 'fill_liq', 'D_v'], rows=ar, cols=ar)
        self.declare_partials('mass_wick', ['L_heatpipe', 't_wk', 'cu_density', 'fill_wk', 'D_v'], rows=ar, cols=ar)

    def compute_partials(self,i,J):
        L_heatpipe = i['L_heatpipe']
        D_od = i['D_od']
        t_w = i['t_w']
        D_v  = i['D_v']
        cu_density = i['cu_density']
        t_wk = i['t_wk']
        fill_wk = i['fill_wk']
        liq_density = i['liq_density']
        fill_liq = i['fill_liq']

        J['mass_heatpipe', 'L_heatpipe'] = cu_density* np.pi*( (D_od/2)**2 - (D_od/2 - t_w)**2 )
        J['mass_heatpipe', 'D_od'] =L_heatpipe*cu_density* np.pi*( (D_od/2) - (D_od/2 - t_w) )
        J['mass_heatpipe', 't_w'] =L_heatpipe*cu_density* (2)*np.pi* ( D_od/2 - t_w )
        J['mass_heatpipe', 'cu_density'] =L_heatpipe* np.pi*( (D_od/2)**2 - (D_od/2 - t_w)**2 )

        J['mass_liquid', 'L_heatpipe'] =liq_density*fill_liq* np.pi/4*( D_v**2 )
        J['mass_liquid', 'D_v'] =L_heatpipe*liq_density*fill_liq* np.pi/2*( D_v )
        J['mass_liquid', 'liq_density'] =L_heatpipe*fill_liq* np.pi/4*( D_v**2 )
        J['mass_liquid', 'fill_liq'] =L_heatpipe*liq_density* np.pi/4*( D_v**2 )

        J['mass_wick', 'L_heatpipe'] =cu_density*fill_wk*       np.pi*( (D_v/2+t_wk)**2 - (D_v/2)**2 )
        J['mass_wick', 'cu_density'] =L_heatpipe*fill_wk*       np.pi*( (D_v/2+t_wk)**2 - (D_v/2)**2 )
        J['mass_wick', 'fill_wk'] =L_heatpipe*cu_density*       np.pi*( (D_v/2+t_wk)**2 - (D_v/2)**2 )
        J['mass_wick', 't_wk'] =L_heatpipe*cu_density*fill_wk*  np.pi*2*( D_v/2 + t_wk )
        J['mass_wick', 'D_v'] =L_heatpipe*cu_density*fill_wk*   np.pi*( (D_v/2+t_wk) - (D_v/2) )




if __name__ == "__main__":
    from openmdao.api import Problem

    nn = 1
    prob = Problem()

    prob.model.add_subsystem('hp_mass', roundHPmass(num_nodes=nn), promotes=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)


    print('mass_heatpipe = ', prob.get_val('hp_mass.mass_heatpipe'))
    print('mass_liquid = ', prob.get_val('hp_mass.mass_liquid'))
    print('mass_wick = ', prob.get_val('hp_mass.mass_wick'))


import numpy as np

from openmdao.api import ExplicitComponent, Group, IndepVarComp


class BatteryStatics(ExplicitComponent):
    """
    Calculates battery pack mass
    """

    def setup(self):

        # Inputs
        #control from optimizer        
        self.add_input('energy_{required}', val=65., units='kW*h', desc='required battery energy')
        self.add_input('eta_{batt}', val=0.85, desc='battery efficiency')
        # output from zappy
        self.add_input('I_{batt}', val=260., units='A', desc='Max Amperage Through the Pack')
        #params (fixed)
        self.add_input('cp_cell', val=1020., units='J/(kg*K)',  desc='specific heat of a battery cell')
        self.add_input('cp_case', val =921., units='J/(kg*K)', desc='specific heat of the case')
        self.add_input('mass_{cell}', val=0.0316, units='kg', desc='single cell mass')
        self.add_input('voltage_{low,cell}', val=2.9, units='V', desc='Cell Voltage at low SOC, before drop-off')
        self.add_input('voltage_{nom,cell}', val=3.4, units='V', desc='Cell Voltage at nominal SOC')
        self.add_input('dischargeRate_{cell}', val=10.5, units='A', desc='Cell Discharge Rate')
        self.add_input('Q_{max}', val=3.5, units='A*h', desc='Max Energy Capacity of a battery cell')
        self.add_input('weightFrac_{case}', val=1.3, desc='Scalar battery cell mass needed additionally for case')
        self.add_input('V_{batt}', val=500., units='V', desc='Nominal Bus Voltage')
        self.add_input('volume_{cell}', val=1.125, units='inch**3', desc='cell volume')

        self.add_output('n_{series}', val=128., desc='number of cells in series')
        self.add_output('n_{parallel}', val=20., desc='number of cells in parallel, based on power constraint')
        self.add_output('n_{parallel}2', val=20., desc='number of cells in parallel, based on energy constraint') #how do we handle this max?
        self.add_output('mass_{battery}', val=300., units='kg', desc='battery mass')
        self.add_output('C_{p,batt}', val=1000., units='J/(kg*K)', desc='mass averaged specific heat')
        #unconnected output used for test checking
        self.add_output('nominal_energy', val=75., units='kW*h', desc='nominal battery energy')
        self.add_output('volume_{pack}', val=5000, units='inch**3', desc='pack volume' )

        # #Finite difference all partials.
        # self.declare_partials('*', '*', method='cs')

        self.declare_partials('n_{series}',['V_{batt}','voltage_{low,cell}'])
        self.declare_partials('n_{parallel}',['I_{batt}','dischargeRate_{cell}'])
        self.declare_partials('mass_{battery}',['V_{batt}','voltage_{low,cell}','I_{batt}','dischargeRate_{cell}','mass_{cell}','weightFrac_{case}'])
        self.declare_partials('nominal_energy',['V_{batt}','voltage_{low,cell}','I_{batt}','dischargeRate_{cell}','Q_{max}','voltage_{nom,cell}','eta_{batt}'])
        self.declare_partials('n_{parallel}2',['V_{batt}','voltage_{low,cell}','Q_{max}','voltage_{nom,cell}','eta_{batt}','energy_{required}'])
        self.declare_partials('C_{p,batt}',['cp_case','cp_cell','weightFrac_{case}'])    
        self.declare_partials('volume_{pack}',['V_{batt}','voltage_{low,cell}','I_{batt}','dischargeRate_{cell}','volume_{cell}'])


    def compute(self, inputs, outputs):

        bus_v = inputs['V_{batt}']
        c_l_v = inputs['voltage_{low,cell}']
        max_amps = inputs['I_{batt}']

        outputs['n_{series}'] = bus_v / c_l_v
        outputs['n_{parallel}'] = max_amps / inputs['dischargeRate_{cell}']
        outputs['mass_{battery}'] = outputs['n_{series}']*outputs['n_{parallel}']*inputs['mass_{cell}']*inputs['weightFrac_{case}']
        outputs['nominal_energy'] = outputs['n_{series}']*outputs['n_{parallel}']*inputs['Q_{max}']*inputs['voltage_{nom,cell}']*inputs['eta_{batt}']/1000.
        outputs['n_{parallel}2'] = outputs['n_{parallel}']*inputs['energy_{required}']/outputs['nominal_energy']
        outputs['C_{p,batt}'] = (inputs['cp_cell']+inputs['cp_case']*(1.-inputs['weightFrac_{case}']))/inputs['weightFrac_{case}']
        outputs['volume_{pack}'] = outputs['n_{series}']*outputs['n_{parallel}']*inputs['volume_{cell}']

    def compute_partials(self, inputs, partials):

        bus_v = inputs['V_{batt}']
        c_l_v = inputs['voltage_{low,cell}']
        max_amps = inputs['I_{batt}']
        I_rate = inputs['dischargeRate_{cell}']
        m_cell = inputs['mass_{cell}']
        wf_case = inputs['weightFrac_{case}']
        n_s = bus_v / c_l_v
        n_p = max_amps / inputs['dischargeRate_{cell}']
        cp_case = inputs['cp_case']
        cp_cell = inputs['cp_cell']
        q_max = inputs['Q_{max}']
        v_n_c = inputs['voltage_{nom,cell}']
        eta = inputs['eta_{batt}']
        e = inputs['energy_{required}']
        nom_e = n_s * n_p * q_max * v_n_c * eta / 1000.
        vc = inputs['volume_{cell}']


        # n_{series}
        partials['n_{series}','V_{batt}'] = 1. / c_l_v
        partials['n_{series}','voltage_{low,cell}'] = -bus_v/(c_l_v**2)
        # n_{parallel}
        partials['n_{parallel}','I_{batt}'] = 1. / I_rate
        partials['n_{parallel}','dischargeRate_{cell}'] = -max_amps/(I_rate**2)
        # mass_{battery}
        partials['mass_{battery}','V_{batt}'] = partials['n_{series}','V_{batt}'] * n_p * m_cell * wf_case
        partials['mass_{battery}','voltage_{low,cell}'] = partials['n_{series}','voltage_{low,cell}'] * n_p  * m_cell * wf_case
        partials['mass_{battery}','I_{batt}'] = partials['n_{parallel}','I_{batt}'] * n_s  * m_cell * wf_case
        partials['mass_{battery}','dischargeRate_{cell}'] = partials['n_{parallel}','dischargeRate_{cell}'] * n_s  * m_cell * wf_case
        partials['mass_{battery}','mass_{cell}'] = n_s * n_p * wf_case
        partials['mass_{battery}','weightFrac_{case}'] = n_s * n_p * m_cell
        # C_{p,batt}
        partials['C_{p,batt}','cp_cell'] = 1./wf_case
        partials['C_{p,batt}','cp_case'] = (1.-wf_case)/wf_case
        partials['C_{p,batt}','weightFrac_{case}'] = -(cp_cell+cp_case)/wf_case**2
        # nominal_energy
        partials['nominal_energy','V_{batt}'] = partials['n_{series}','V_{batt}'] * n_p * q_max * v_n_c * eta / 1000.
        partials['nominal_energy','voltage_{low,cell}'] = partials['n_{series}','voltage_{low,cell}'] * n_p  * q_max * v_n_c * eta / 1000.
        partials['nominal_energy','I_{batt}'] = partials['n_{parallel}','I_{batt}'] * n_s  * q_max * v_n_c * eta / 1000.
        partials['nominal_energy','dischargeRate_{cell}'] = partials['n_{parallel}','dischargeRate_{cell}'] * n_s  * q_max * v_n_c * eta / 1000.
        partials['nominal_energy','voltage_{nom,cell}'] = n_s * n_p * q_max * eta / 1000.
        partials['nominal_energy','Q_{max}'] = n_s * n_p * v_n_c * eta / 1000.
        partials['nominal_energy','eta_{batt}'] = n_s * n_p * v_n_c * q_max / 1000.
        # n_{parallel}2 = outputs['n_{parallel}']*inputs['energy_{required}']/outputs['nominal_energy']
        partials['n_{parallel}2','V_{batt}'] = -1000.*c_l_v*e / (eta*bus_v**2*q_max*v_n_c)
        partials['n_{parallel}2','voltage_{low,cell}'] = 1000.* e / (eta*bus_v*q_max*v_n_c)
        #partials['n_{parallel}2','I_{batt}'] = 0.
        #partials['n_{parallel}2','dischargeRate_{cell}'] = 0. 
        partials['n_{parallel}2','voltage_{nom,cell}'] = -1000.*c_l_v*e / (eta*bus_v*q_max*v_n_c**2) 
        partials['n_{parallel}2','Q_{max}'] = -1000.*c_l_v*e / (eta*bus_v*q_max**2*v_n_c) 
        partials['n_{parallel}2','eta_{batt}'] = -1000.*c_l_v*e / (eta**2*bus_v*q_max*v_n_c)
        partials['n_{parallel}2','energy_{required}'] = n_p / nom_e
        # volume_{pack}
        partials['volume_{pack}','V_{batt}'] = partials['n_{series}','V_{batt}'] * n_p * vc
        partials['volume_{pack}','voltage_{low,cell}'] = partials['n_{series}','voltage_{low,cell}'] * n_p * vc
        partials['volume_{pack}','I_{batt}'] = partials['n_{parallel}','I_{batt}'] * n_s * vc
        partials['volume_{pack}','dischargeRate_{cell}'] = partials['n_{parallel}','dischargeRate_{cell}'] * n_s * vc
        partials['volume_{pack}', 'volume_{cell}'] = n_s * n_p

class BatteryStaticsGroup(Group):

    def setup(self):
        
        des_vars = self.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        des_vars.add_output('cp_cell', val=1020., units='J/(kg*K)')
        des_vars.add_output('cp_case', val=921., units='J/(kg*K)')
        des_vars.add_output('volume_{cell}', val=1.125, units='inch**3')

        self.add_subsystem(name='BatteryStatics', subsys=BatteryStatics(), promotes=['*'])


if __name__ == '__main__':
    from openmdao.api import Problem, Group, IndepVarComp

    p = Problem()
    p.model = Group()
    des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

    des_vars.add_output('V_{batt}', 500., units='V')
    des_vars.add_output('I_{batt}', 300., units='A')
    des_vars.add_output('mass_{cell}', 0.045, units='kg')

    p.model.add_subsystem('mass', BatteryStatics(), promotes=['*'])

    p.setup(check=False, force_alloc_complex=True)
    p.check_partials(compact_print=True, method='cs')
    
    p.setup()
    p.run_model()

    print('Num series:  ', p.get_val('n_{series}'))
    print('Num parallel:  ', p.get_val('n_{parallel}'))
    print('Num parallel:  ', p.get_val('n_{parallel}2'))
    print('Battery Mass:  ', p.get_val('mass_{battery}'))
    print('Battery Nominal Energy:  ', p.get_val('nominal_energy'))
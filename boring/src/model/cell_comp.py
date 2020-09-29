import numpy as np

from openmdao.api import ExplicitComponent


class CellComp(ExplicitComponent):
    """
    Compute behavior of a single battery cell then expand to the full pack.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        # Inputs
        self.add_input('I_pack', val=3.25*np.ones(n), units='A', desc='Pack_current')
        # Static Constants
        self.add_input('Q_max', val=3.*np.ones(1), units='A*h', desc='Max Energy Capacity of a battery cell')
        self.add_input('n_parallel', val=40.*np.ones(n), units=None, desc='battery strings in parallel')
        self.add_input('n_series', val=128.0*np.ones(n), units=None, desc='cells in a series string')
        # Integrated State Variables
        self.add_input('SOC', val=0.98*np.ones(n), units=None, desc='State of charge fraction')
        self.add_input('U_Th', val=np.ones(n), units='V', desc='Thevenin (Polarization) voltage')
        # Map Inputs From The Interpolation Component
        self.add_input('U_oc', val=4.16*np.ones(n), units='V', desc='Open-circuit voltage')
        self.add_input('C_Th', val=2000.*np.ones(n), units='F', desc='Thevenin RC parallel capacitance (polarization)')
        self.add_input('R_Th', val=0.01*np.ones(n), units='ohm', desc='Thevenin RC parallel resistance (polarization)')
        self.add_input('R_0', val=0.01*np.ones(n), units='ohm', desc='Internal resistance of the battery')

        # Outputs
        self.add_output('dXdt:SOC', val=np.ones(n), units='1/s', desc='Time derivative of state of charge')
        self.add_output('dXdt:U_Th', val=np.ones(n), units='V/s', desc='Time derivative of Thevenin voltage')
        self.add_output('U_pack', val=416*np.ones(n), units='V', desc='Total Pack Voltage')
        self.add_output('Q_pack', val=np.ones(n), units='W', desc='Total Pack Heat Load')
        self.add_output('pack_eta', val=np.ones(n), units=None, desc='Efficiency of the pack')

        # Partials
        self.declare_partials(of='*', wrt='*', dependent=False)
        ar = np.arange(n)

        self.declare_partials(of='dXdt:SOC', wrt='I_pack', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:SOC', wrt='n_parallel', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:SOC', wrt='Q_max')

        self.declare_partials(of='dXdt:U_Th', wrt='U_Th', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:U_Th', wrt='R_Th', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:U_Th', wrt='C_Th', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:U_Th', wrt='I_pack', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:U_Th', wrt='n_parallel', rows=ar, cols=ar)

        self.declare_partials(of='U_pack', wrt='U_oc', rows=ar, cols=ar)
        self.declare_partials(of='U_pack', wrt='U_Th', rows=ar, cols=ar)
        self.declare_partials(of='U_pack', wrt='I_pack', rows=ar, cols=ar)
        self.declare_partials(of='U_pack', wrt='n_parallel', rows=ar, cols=ar)
        self.declare_partials(of='U_pack', wrt='R_0', rows=ar, cols=ar)
        self.declare_partials(of='U_pack', wrt='n_series', rows=ar, cols=ar)

        self.declare_partials(of='Q_pack', wrt='I_pack', rows=ar, cols=ar)
        self.declare_partials(of='Q_pack', wrt='n_parallel', rows=ar, cols=ar)
        self.declare_partials(of='Q_pack', wrt='n_series', rows=ar, cols=ar)
        self.declare_partials(of='Q_pack', wrt='R_0', rows=ar, cols=ar)
        self.declare_partials(of='Q_pack', wrt='U_Th', rows=ar, cols=ar)

        self.declare_partials(of='pack_eta', wrt='I_pack', rows=ar, cols=ar)
        self.declare_partials(of='pack_eta', wrt='n_parallel', rows=ar, cols=ar)
        # self.declare_partials(of='pack_eta', wrt='n_series', rows=ar, cols=ar) # known 0
        self.declare_partials(of='pack_eta', wrt='R_0', rows=ar, cols=ar)
        self.declare_partials(of='pack_eta', wrt='U_Th', rows=ar, cols=ar)
        self.declare_partials(of='pack_eta', wrt='U_oc', rows=ar, cols=ar)


    def compute(self, inputs, outputs):
        C_Th = inputs['C_Th']
        R_Th = inputs['R_Th']
        U_Th = inputs['U_Th']
        R_0 = inputs['R_0']
        Q_max = inputs['Q_max']
        U_oc = inputs['U_oc']
        n_p = inputs['n_parallel']
        n_s = inputs['n_series']
        I_pack = inputs['I_pack']

        #cell
        I_Li = I_pack / n_p
        outputs['dXdt:SOC'] = -I_Li / (3600.0 * Q_max) # conversion from hours to seconds
        outputs['dXdt:U_Th'] = -U_Th / (R_Th * C_Th) + I_Li / C_Th
        U_L = U_oc - U_Th - (I_Li * R_0)

        #thermal
        Q_cell = I_Li**2 * (R_0 + U_Th/I_Li)

        #pack
        outputs['U_pack'] = U_L * n_s
        outputs['Q_pack'] = Q_cell * n_s * n_p
        P_tot = I_pack * outputs['U_pack']
        outputs['pack_eta'] = 1. - outputs['Q_pack']/((P_tot + outputs['Q_pack']))

    def compute_partials(self, inputs, partials):

        C_Th = inputs['C_Th']
        R_Th = inputs['R_Th']
        U_Th = inputs['U_Th']
        R_0 = inputs['R_0']
        Q_max = inputs['Q_max']
        U_oc = inputs['U_oc']
        n_p = inputs['n_parallel']
        n_s = inputs['n_series']
        I_pack = inputs['I_pack']

        I_Li = I_pack / n_p
        U_L = U_oc - U_Th - (I_Li * R_0)
        Q_cell = I_Li**2 * (R_0 + U_Th/I_Li)
        U_pack = U_L * n_s
        Q_pack = Q_cell * n_s * n_p
        P_tot = I_pack * U_pack
        pack_eta = 1. - Q_pack/((P_tot + Q_pack))

        dI_li__dnp = -I_pack/n_p**2
        dU_L__dnp = -R_0*dI_li__dnp
        dU_pack__dnp = dU_L__dnp * n_s
        dQ_cell__dnp = 2*I_Li*dI_li__dnp*(R_0+U_Th/I_Li) - U_Th*dI_li__dnp 
        dQ_pack__dnp = n_s*(dQ_cell__dnp*n_p + Q_cell)
        dPtot__dnp = I_pack * dU_pack__dnp
        dpack_eta__dnp = -dQ_pack__dnp/(P_tot+Q_pack) + Q_pack/(P_tot + Q_pack)**2 * (dPtot__dnp+dQ_pack__dnp)

        dI_li__dI_pack = 1/n_p
        dQ_cell__dI_pack = 2*I_Li*dI_li__dI_pack*(R_0+U_Th/I_Li) - U_Th*dI_li__dI_pack
        dQ_pack__dI_pack = dQ_cell__dI_pack * n_s * n_p
        dU_pack__dI_pack = -R_0*n_s/n_p
        dPtot__dI_pack = U_pack + dU_pack__dI_pack*I_pack
        dpack_eta__dI_pack = -dQ_pack__dI_pack/(P_tot+Q_pack) + Q_pack/(P_tot + Q_pack)**2 * (dPtot__dI_pack+dQ_pack__dI_pack)

        dPtot__dU_th = -I_pack*n_s
        dQ_pack_dU_th = I_pack * n_s
        dpack_eta__dU_t = -dQ_pack_dU_th/(P_tot+Q_pack) + Q_pack/(P_tot + Q_pack)**2 * (dPtot__dU_th+dQ_pack_dU_th)

        dU_pack__dR_0 = -I_pack * n_s / n_p
        dPtot__dR_0 = I_pack * dU_pack__dR_0
        dQ_pack__dR_0 = (I_pack**2 * n_s) / n_p
        dpack_eta__dR_0 = -dQ_pack__dR_0/(P_tot+Q_pack) + Q_pack/(P_tot + Q_pack)**2 * (dPtot__dR_0+dQ_pack__dR_0)

        dPtot__dU_oc = I_pack*n_s
        dpack_eta__dU_oc = Q_pack/(P_tot + Q_pack)**2 * dPtot__dU_oc


        partials['dXdt:SOC','I_pack'] = -1./(3600.0*Q_max*n_p)
        partials['dXdt:SOC','n_parallel'] = I_pack/(3600.0*Q_max*n_p**2)
        partials['dXdt:SOC','Q_max'] = I_pack/(3600.0*n_p*Q_max**2)

        partials['dXdt:U_Th','U_Th'] = -1./(R_Th*C_Th)
        partials['dXdt:U_Th','R_Th'] = U_Th/(C_Th * R_Th**2)
        partials['dXdt:U_Th','C_Th'] = (n_p*U_Th - I_pack*R_Th) / (C_Th**2 * n_p* R_Th)
        partials['dXdt:U_Th','I_pack'] = 1./(C_Th*n_p)
        partials['dXdt:U_Th','n_parallel'] = -I_pack/(C_Th*n_p**2)

        partials['U_pack','U_oc'] = n_s
        partials['U_pack','U_Th'] = -n_s
        partials['U_pack','R_0'] = dU_pack__dR_0
        partials['U_pack','n_parallel'] = dU_pack__dnp
        partials['U_pack','I_pack'] = dU_pack__dI_pack
        partials['U_pack','n_series'] = U_oc - U_Th - (I_pack/n_p * R_0)

        partials['Q_pack','I_pack'] = n_s * ((2 * I_pack * R_0)/n_p + U_Th)
        partials['Q_pack','n_parallel'] = dQ_pack__dnp
        partials['Q_pack','R_0'] = (I_pack**2 * n_s) / n_p
        partials['Q_pack','U_Th'] = dQ_pack_dU_th
        partials['Q_pack','n_series'] = (I_pack**2 * ((n_p * U_Th)/I_pack + R_0)) / n_p

        partials['pack_eta','U_oc'] = dpack_eta__dU_oc
        partials['pack_eta','n_parallel'] = dpack_eta__dnp
        partials['pack_eta','I_pack'] = dpack_eta__dI_pack
        partials['pack_eta','R_0'] = dpack_eta__dR_0
        partials['pack_eta','U_Th'] = dpack_eta__dU_t
        # partials['pack_eta','n_series'] = 0. 

if __name__ == "__main__":

    from openmdao.api import Problem, Group, IndepVarComp
    
    p = Problem()
    p.model = Group()

    p.model.add_subsystem('CellComp', CellComp(num_nodes=1), promotes=['*'])
    
    p.setup(mode='auto', check=True, force_alloc_complex=True)

    p.check_partials(compact_print=True, method='cs',step=1e-50)


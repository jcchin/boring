"""
Author: Jeff Chin

"""
import openmdao.api as om

class pcmMass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('rho_foam', 8960, units='kg/m**3', desc='density of the conductive foam')
        self.add_input('rho_pcm', 1450, units='kg/m**3', desc='density of the phase change material')
        self.add_input('t_pad', 0.020, units='m', desc='pcm pad thickness')
        self.add_input('batt_l', .10599, units='m', desc='cell length (105.99mm for Large Amprius)')
        self.add_input('batt_w', 0.04902, units='m', desc='cell width, excluding lip material (49.02mm for Large Amprius')
        self.add_input('porosity', 0.97, desc='porosity of the foam, 1 = completely void, 0 = solid')
        self.add_input('batt_l_pcm_scaler', 0.5, desc='sizes the pcm pad as a fraction of the cell length')

        self.add_output('A_pad', 0.000020, units='m**2', desc='pcm pad area')
        self.add_output('mass_pcm', .010, units='kg', desc='PCM bulk mass')

        self.declare_partials('*', '*', method='cs')

    # def setup_partials(self):
    #     self.declare_partials('mass_pcm', ['rho_pcm', 'rho_foam', 't_pad', 'A_pad','porosity'])

    def compute(self, inputs, outputs):
        rho_f    =  inputs['rho_foam']
        rho_p    =  inputs['rho_pcm']
        t        =  inputs['t_pad']
        batt_l   =  inputs['batt_l']
        batt_w   =  inputs['batt_w']
        porosity =  inputs['porosity']
        l_scaler = inputs['batt_l_pcm_scaler']
        
        rho_bulk = 1. / (porosity / rho_p + (1 - porosity) / rho_f)
        print('bulk density: ', rho_bulk)

        outputs['A_pad'] = (batt_l*l_scaler) * batt_w
        outputs['mass_pcm'] = rho_bulk*t*outputs['A_pad']


    # def compute_partials(self, inputs, J):
    #     rho_f    =  inputs['rho_foam']
    #     rho_p    =  inputs['rho_pcm']
    #     t        =  inputs['t_pad']
    #     A        =  inputs['A_pad']
    #     porosity =  inputs['porosity']

    #     d_porosity = -rho_f * rho_p * (rho_f - rho_p) / (
    #                 rho_p * (porosity - 1.) - rho_f * porosity) ** 2
    #     d_pcm = rho_f ** 2 * porosity / (rho_p * (porosity - 1.) - rho_f * porosity) ** 2
    #     d_foam = -rho_p ** 2 * (porosity - 1.) / (rho_p * (porosity - 1.) - rho_f * porosity) ** 2

    #     J['mass_pcm', 'porosity'] = d_porosity * A * t
    #     J['mass_pcm', 'rho_foam'] = d_foam * A * t
    #     J['mass_pcm', 'rho_pcm']  = d_pcm * A * t
    #     J['mass_pcm', 't_pad']    = A / (porosity / rho_p + (1 - porosity) / rho_f)
    #     J['mass_pcm', 'A_pad']    = t / (porosity / rho_p + (1 - porosity) / rho_f)



if __name__ == "__main__":
    prob = om.Problem(model=om.Group())  
    nn=1  

    prob.model.add_subsystem('comp1', pcmMass(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    # prob.check_partials(method='cs', compact_print=True)

    print('mass pcm: ', prob.get_val('mass_pcm'))
    

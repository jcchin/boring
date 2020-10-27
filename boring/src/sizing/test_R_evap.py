#Author: K. Look  Created 10/27/20

import os 
cwd = os.path.dirname(os.path.realpath(__file__))
chdir(cwd)

import openmdao.api as om
from R_evap import R_evapComp


if __name__ == "__main__":
    p = om.Problem()
    model = p.model
    nn = 1

    p.model.add_subsystem(name = 'r_evap',
                          subsys = R_evapComp(num_nodes=nn),
                          
                          promotes_inputs=['alpha', 'h_fg', 'T_hp', 'v_fg', 'R_g', 'P_v', 'D_od', 'r_i', 'k_w', 'L_evap', 'r_i', 'D_v', 'k_wk', 'L_evap', 'A_intere'],
                          promotes_outputs = ['h_intere','R_we','R_wke','R_intere'])
    

p.setup(force_alloc_complex=True)

p.set_val('alpha', 0.069157409) #this is a randomly assigned value for testing
p.set_val('h_fg', 0.123091285) #this is a randomly assigned value for testing
p.set_val('T_hp', 0.660239167) #this is a randomly assigned value for testing
p.set_val('v_fg', 0.464872246) #this is a randomly assigned value for testing
p.set_val('R_g', 0.476206797) #this is a randomly assigned value for testing
p.set_val('P_v', 0.643959038) #this is a randomly assigned value for testing
p.set_val('D_od', 0.115820412) #this is a randomly assigned value for testing
p.set_val('r_i', 0.62955964) #this is a randomly assigned value for testing
p.set_val('k_w', 0.27957197) #this is a randomly assigned value for testing
p.set_val('L_evap', 0.634016837) #this is a randomly assigned value for testing
p.set_val('r_i', 0.802373207) #this is a randomly assigned value for testing
p.set_val('D_v', 0.589928558) #this is a randomly assigned value for testing
p.set_val('k_wk', 0.78728651) #this is a randomly assigned value for testing
p.set_val('L_evap', 0.592621117) #this is a randomly assigned value for testing
p.set_val('h_intere', 0.376772618) #this is a randomly assigned value for testing
p.set_val('A_intere', 0.24770787) #this is a randomly assigned value for testing

p.run_model()
p.check_partials(includes='r_evap',method = 'cs',compact_print=True,show_only_incorrect=False)
    
print('Finished Successfully')

print('\n', '\n')
print('--------------Outputs---------------')
print('\n', '\n')
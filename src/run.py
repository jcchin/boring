import openmdao.api as om
from src.mass import packMass
from src.heat_pipe import OHP
from src.pack_design import packSize, pcmSize, SizingGroup


def test_run():
    p = om.Problem()
    model = p.model
    nn = 1

    record_file = 'geometry.sql'
    p.add_recorder(om.SqliteRecorder(record_file))
    p.recording_options['includes'] = ['*']
    p.recording_options['record_objectives'] = True
    p.recording_options['record_constraints'] = True
    p.recording_options['record_desvars'] = True
    p.recording_options['record_inputs'] = True


    model.add_subsystem('sizing', SizingGroup(num_nodes=nn), promotes=['*'])
    #model.add_design_var('sizing.L', lower=-1, upper=1)
    #model.add_objective('OD1.Eff')

    p.setup(force_alloc_complex=True)

    #p.set_val('DESIGN.rot_ir' , 60)

    p.run_model()
    p.record('final') #trigger problem record (or call run_driver if it's attached to the driver)

    # p.run_driver()
    # p.cleanup()

    model.list_inputs(prom_name=True)
    model.list_outputs(prom_name=True)
    # p.check_partials(compact_print=True, method='cs')

    print("num of cells: ", p['n_cells'])
    print("flux: ", p['flux'])
    print("PCM mass: ", p['PCM_tot_mass'])
    print("PCM thickness (mm): ", p['t_PCM'])
    print("OHP mass: ", p['mass_OHP'])
    print("packaging mass: ", p['p_mass'])
    print("total mass: ", p['tot_mass'])
    print("package mass fraction: ", p['mass_frac'])
    print("pack energy density: ", p['energy']/(p['tot_mass']))
    print("cell energy density: ", (p['q_max'] * p['v_n_c']) / (p['cell_mass']))
    print("pack energy (kWh): ", p['energy']/1000.)
    print("pack cost ($K): ", p['n_cells']*0.4)


if __name__ == "__main__":

    test_run()
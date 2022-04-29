import numpy as np
import pandas as pd

import openmdao.api as om

from battery_mesh_deformation import MeshDeformComp
from thermal_analysis_component import ThermalAnalysisComp


def make_curve():

    num_opts = 5
    energy_densities = np.linspace(200.0, 600.0, num_opts)
    opt_dratio_vals = np.zeros(num_opts)
    opt_dextra_vals = np.zeros(num_opts)
    opt_mass_vals = np.zeros(num_opts)

    for i in range(num_opts):
        p, fea_assembler = make_problem(cell_energy_density=energy_densities[i])
        recorder = om.SqliteRecorder(f'case{i}.sql')
        p.driver.add_recorder(recorder)
        p.run_driver()
        opt_dratio_vals[i] = p.get_val("dratio")[0]
        opt_dextra_vals[i] = p.get_val("dextra")[0]
        opt_mass_vals[i] = p.get_val("mass")[0]

    print(opt_dratio_vals)
    print(opt_dextra_vals)
    print(opt_mass_vals)
    df = pd.DataFrame({"energy_density":energy_densities,
                       "dratio":opt_dratio_vals,
                       "dextra":opt_dextra_vals,
                       "mass":opt_mass_vals})
    df.to_csv("shape_optimization_values_ratio_0.4_extra_1.5_200_350_tfinal_5.csv", index=False)

    return

def make_problem(cell_energy_density=225.0):

    ratio0 = 0.4
    extra0 = 1.5
    ratio_lb = 0.25
    ratio_ub = 0.75
    extra_lb = 1.1
    extra_ub = 3.5

    prob = om.Problem()

    # Set up the component for the shape parameter design variables
    ivc = om.IndepVarComp()
    ivc.add_output("dratio", val=0.0, units=None)
    ivc.add_output("dextra", val=0.0, units=None)
    prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])

    # Create the finite-element analysis component but don't set it as a subsystem yet
    analysis_comp = ThermalAnalysisComp(cell_energy_density=cell_energy_density)
    fea_assembler = analysis_comp.fea_assembler

    # Get the original finite-element node locations and set up the mesh-deformation component
    Xpts0 = fea_assembler.getOrigNodes()
    nnodes = int(len(Xpts0)/3)
    mesh_deformation_comp = MeshDeformComp(Xpts0=Xpts0, nnodes=nnodes, extra=extra0, ratio=ratio0)
    prob.model.add_subsystem("mesh_deformation", mesh_deformation_comp, promotes_inputs=["dratio", "dextra"], promotes_outputs=["Xpts"])

    # Add the analysis subsystem after the mesh_deformation component so that Xpts flows the right direction
    prob.model.add_subsystem("analysis", analysis_comp, promotes_inputs=["Xpts"], promotes_outputs=["mass", "ks_temp_adjacent", "ks_temp_diagonal", "temp_ratio"])

    # Set up the driver
    opt_settings = {"Print file":f"SNOPT_print_ced_{cell_energy_density:.0f}.out",
                    "Summary file":f"SNOPT_summary_ced_{cell_energy_density:.0f}.out"}
    prob.driver = om.pyOptSparseDriver(optimizer="SNOPT")
    prob.driver.opt_settings = opt_settings

    # Define the optimization problem
    prob.model.add_design_var("dratio", lower=(ratio_lb-ratio0), upper=(ratio_ub-ratio0))
    prob.model.add_design_var("dextra", lower=(extra_lb-extra0), upper=(extra_ub-extra0))

    init_temp = 298.0 # initial battery temperature, degK
    max_temp = 340.0 # max allowable battery temperature, degK

    prob.model.add_objective("mass", index=0, scaler=1.0)
    prob.model.add_constraint("ks_temp_adjacent", upper=(max_temp-init_temp), scaler=1.0/(max_temp-init_temp))
    prob.model.add_constraint("ks_temp_diagonal", upper=(max_temp-init_temp), scaler=1.0/(max_temp-init_temp))
    #prob.model.add_constraint("temp_ratio", upper=1.15)

    prob.setup()
    om.n2(prob, show_browser=False, outfile="battery_shape_optimization.html")

    return prob, fea_assembler

# p, fea_assembler = make_problem(cell_energy_density=300.0)
# p.run_driver()
# dratio = p.get_val("dratio")
# dextra = p.get_val("dextra")
# print(f"dratio = {dratio}, dextra = {dextra}")
make_curve()
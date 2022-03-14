import openmdao.api as om

from battery_mesh_deformation import MeshDeformComp
from thermal_analysis_component import ThermalAnalysisComp

# TODO:
# - Find way to get T4/T1 ratio to constriain
# - Make optimization curve

def make_problem():

    prob = om.Problem()

    # Set up the component for the shape parameter design variables
    ivc = om.IndepVarComp()
    ivc.add_output("dratio", val=0.0, units=None)
    ivc.add_output("dextra", val=0.0, units=None)
    prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])

    # Create the finite-element analysis component but don't set it as a subsystem yet
    analysis_comp = ThermalAnalysisComp()
    fea_assembler = analysis_comp.fea_assembler

    # Get the original finite-element node locations and set up the mesh-deformation component
    Xpts0 = fea_assembler.getOrigNodes()
    nnodes = int(len(Xpts0)/3)
    mesh_deformation_comp = MeshDeformComp(Xpts0=Xpts0, nnodes=nnodes, extra=1.1516, ratio=0.7381) #extra=1.5, ratio=0.4)
    prob.model.add_subsystem("mesh_deformation", mesh_deformation_comp, promotes_inputs=["dratio", "dextra"], promotes_outputs=["Xpts"])

    # Add the analysis subsystem after the mesh_deformation component so that Xpts flows the right direction
    prob.model.add_subsystem("analysis", analysis_comp, promotes_inputs=["Xpts"], promotes_outputs=["mass", "ks_temp_adjacent", "ks_temp_diagonal"])

    # Set up the driver
    prob.driver = om.ScipyOptimizeDriver(debug_print=["objs", "nl_cons"], maxiter=200)
    prob.driver.options["optimizer"] = "SLSQP"

    # Define the optimization problem
    prob.model.add_design_var("dratio", lower=-0.4881, upper=0.0119)  # ratio in [0.25, 0.75]
    prob.model.add_design_var("dextra", lower=-0.01516, upper=0.8484)  # extra in [1.1, 2.00]

    prob.model.add_objective("mass", index=0, scaler=26.0)
    prob.model.add_constraint("ks_temp_adjacent", upper=100.0, scaler=50.0)  # Max temp = 135 degC; assume starting at 35 degC
    prob.model.add_constraint("ks_temp_diagonal", upper=100.0, scaler=50.0)

    prob.setup()
    om.n2(prob, show_browser=False, outfile="battery_shape_optimization.html")

    return prob, fea_assembler

p, fea_assembler = make_problem()
p.run_driver()
dratio = p.get_val("dratio")
dextra = p.get_val("dextra")
print(f"dratio = {dratio}, dextra = {dextra}")
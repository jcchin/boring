import os

from mpi4py import MPI
import numpy as np

import openmdao.api as om
from tacs import functions, constitutive, elements, TACS, pyTACS


def setup_assembler(cell_energy_density=225.0):

    comm = MPI.COMM_WORLD

    # Name of the bdf file to get the mesh
    bdfFile = os.path.join(os.path.dirname(__file__), 'battery_ratio_0.4_extra_1.5.bdf')
    # Instantiate the pyTACS object
    FEAAssembler = pyTACS(bdfFile, comm)

    # Specify the plate thickness
    tplate = 0.065

    # Define material properties for two materials used in this problem
    # Properties of the battery cells
    battery_rho = 1460.0  # density kg/m^3
    battery_kappa = 1.3 # Thermal conductivity W/(m⋅K)
    battery_cp = 880.0 # Specific heat J/(kg⋅K)

    # Compute the total cell energy
    thermal_energy = 1000.0*cell_energy_density/12.5  # Total energy, J

    # Properties of the battery pack (aluminum)
    alum_rho = 2700.0  # density kg/m^3
    alum_kappa = 204.0 # Thermal conductivity W/(m⋅K)
    alum_cp = 883.0 # Specific heat J/(kg⋅K)

    # The callback function to define the element properties
    def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):

        # Setup property and constitutive objects
        if compDescript == 'Block': #'block':  # If the bdf file labels this component as "block", then it is aluminum
            prop = constitutive.MaterialProperties(rho=alum_rho, kappa=alum_kappa, specific_heat=alum_cp)
        else:  # otherwise it is a battery
            prop = constitutive.MaterialProperties(rho=battery_rho, kappa=battery_kappa, specific_heat=battery_cp)

        # Set one thickness value for every component
        con = constitutive.PlaneStressConstitutive(prop, t=tplate, tNum=-1)

        # For each element type in this component,
        # pass back the appropriate tacs element object
        elemList = []
        model = elements.HeatConduction2D(con)
        for elemDescript in elemDescripts:
            if elemDescript in ['CQUAD4', 'CQUADR']:
                basis = elements.LinearQuadBasis()
            elif elemDescript in ['CTRIA3', 'CTRIAR']:
                basis = elements.LinearTriangleBasis()
            else:
                print("Element '%s' not recognized" % (elemDescript))
            elem = elements.Element2D(model, basis)
            elemList.append(elem)

        return elemList

    # Set up constitutive objects and elements
    FEAAssembler.initialize(elemCallBack)

    # Create a transient problem that will represent time varying convection
    transientProblem = FEAAssembler.createTransientProblem(f"Transient_{int(cell_energy_density)}", tInit=0.0, tFinal=30.0, numSteps=60)

    # Get the time steps and declare the loads
    timeSteps = transientProblem.getTimeSteps()
    for i, t in enumerate(timeSteps):
        if t <= 2.0:  # only apply the load for the first 2 seconds
            # select the component of the battery undergoing thermal runaway
            compIDs = FEAAssembler.selectCompIDs(include=["Battery.00"])
            transientProblem.addLoadToComponents(i, compIDs, [thermal_energy/2.0])

    # Define the functions of interest as maximum temperature within 2 different batteries
    compIDs_01 = FEAAssembler.selectCompIDs(["Battery.01"])#battery6"])  # adjecent battery
    compIDs_04 = FEAAssembler.selectCompIDs(["Battery.04"])#battery5"])  # diagonal battery

    transientProblem.addFunction('mass', functions.StructuralMass)
    transientProblem.addFunction('ks_temp_adjacent', functions.KSTemperature,
                                 ksWeight=100.0, compIDs=compIDs_01)
    transientProblem.addFunction('ks_temp_diagonal', functions.KSTemperature,
                                 ksWeight=100.0, compIDs=compIDs_04)

    return FEAAssembler, transientProblem


class ThermalAnalysisComp(om.ExplicitComponent):

    def __init__(self, cell_energy_density=225.0):
        super().__init__()
        self.cell_energy_density = cell_energy_density  # (W*hr/kg)
        fea_assembler, transient_problem = setup_assembler(cell_energy_density=self.cell_energy_density)
        self.fea_assembler = fea_assembler
        self.transient_problem = transient_problem

    def setup(self):

        Xpts0 = self.fea_assembler.getOrigNodes()

        self.add_input("Xpts", val=Xpts0, units="m")

        self.add_output("mass", val=0.0, units="kg")
        self.add_output("ks_temp_adjacent", val=0.0, units="degC")
        self.add_output("ks_temp_diagonal", val=0.0, units="degC")
        self.add_output("temp_ratio", val=0.0, units=None, desc="ks_temp_adjacent/ks_temp_diagonal")

        self.declare_partials(of=["mass", "ks_temp_adjacent", "ks_temp_diagonal", "temp_ratio"], wrt="Xpts")

    def compute(self, inputs, outputs):

        self.transient_problem.setNodes(inputs["Xpts"])
        self.transient_problem._updateAssemblerVars()

        self.transient_problem.solve()

        funcs = {}
        self.transient_problem.evalFunctions(funcs)
        self.transient_problem.writeSolution()

        outputs["mass"] = funcs[f"Transient_{int(self.cell_energy_density)}_mass"]/30.0
        outputs["ks_temp_adjacent"] = funcs[f"Transient_{int(self.cell_energy_density)}_ks_temp_adjacent"]
        outputs["ks_temp_diagonal"] = funcs[f"Transient_{int(self.cell_energy_density)}_ks_temp_diagonal"]
        outputs["temp_ratio"] = funcs[f"Transient_{int(self.cell_energy_density)}_ks_temp_adjacent"]/funcs[f"Transient_{int(self.cell_energy_density)}_ks_temp_diagonal"]

    def compute_partials(self, inputs, partials):

        self.transient_problem.setNodes(inputs["Xpts"])
        self.transient_problem._updateAssemblerVars()

        funcs = {}
        self.transient_problem.evalFunctions(funcs)
        self.transient_problem.writeSolution()
        f1 = funcs[f"Transient_{int(self.cell_energy_density)}_ks_temp_adjacent"]
        f2 = funcs[f"Transient_{int(self.cell_energy_density)}_ks_temp_diagonal"]

        funcs_sens = {}
        self.transient_problem.evalFunctionsSens(funcs_sens)
        df1_dXpts = funcs_sens[f"Transient_{int(self.cell_energy_density)}_ks_temp_adjacent"]["Xpts"]
        df2_dXpts = funcs_sens[f"Transient_{int(self.cell_energy_density)}_ks_temp_diagonal"]["Xpts"]

        partials["mass", "Xpts"] = funcs_sens[f"Transient_{int(self.cell_energy_density)}_mass"]["Xpts"]/30.0
        partials["ks_temp_adjacent", "Xpts"] = df1_dXpts
        partials["ks_temp_diagonal", "Xpts"] = df2_dXpts
        partials["temp_ratio", "Xpts"] = df1_dXpts*(1.0/f2) + (df2_dXpts)*(-f1/f2**2)

if __name__ == "__main__":

    # Test the assembler setup
    fea_assembler, transient_problem = setup_assembler()
    transient_problem.solve()
    funcs = {}
    transient_problem.evalFunctions(funcs)
    print(funcs)

    # prob = om.Problem()
    # prob.model.add_subsystem("thermal_comp", ThermalAnalysisComp())
    # prob.setup()
    # prob.run_model()
    # prob.check_partials()
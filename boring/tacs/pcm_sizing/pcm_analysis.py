from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from tmr import TMR, TopOptUtils
from tacs import TACS, elements, constitutive, functions

# Constants and assumptions
# -------------------------

foam_material = 'cu'
# pcm_material = 'croda'  -> same conductivity for both

# Material properties
if foam_material == 'al':
    porosity = 0.90
    k_foam = 58.0
    rho_foam = 8960.0
else:  # 'cu'
    porosity = 0.94
    k_foam = 401.0
    rho_foam = 8960.0

k_pcm = 0.25
rho_pcm = 1450.0

bulk_rho = 1. / (porosity / rho_pcm + (1 - porosity) / rho_foam)
bulk_k = 1. / (porosity / k_pcm + (1 - porosity) / k_foam)

battery_rho = (44e-3)/((61e-3)*(54e-3)*(5.6e-3))  # spec 44g weight/ volume
battery_k = 200.0  # ???? https://tfaws.nasa.gov/wp-content/uploads/TFAWS18-PT-11.pdf

# Heat flux rates
#efficiency = 0.95  # efficiency of battery (rest lost to heat)
discharge_rate = 10.4  # 2C
voltage = 3.28
resistance = 15e-3
Tmax = 19.0

'''
Compute required heat-pipe interface temperature through energy balance:
Battery is only heat-source, and heat pipe is only heat sink.
Assuming heat-pipe interface behavior similar to convection Q = h*A(T-Tref)
Choose h, then by balance of energy, we can set the interface temperature of T
'''

Q = resistance*discharge_rate**2
Tref = 0.0  # reference temperature of heat pipe (above ambient)
h = 300.0  # convective heat transfer coefficient to from PCM to heat pipe
A = (5e-3)*(61e-3)  # area of heat pipe face, m^2
dT = Q/(h*A) + Tref  # fixed temperature boundary condition at heat pipe interface

def get_elems_and_surfs(forest, names):
    # Get the element numbers and faces for a list of names
    elems = []
    surfs = []
    for n in names:
        octants = forest.getOctsWithName(n)
        for oc in octants:
            elems.append(oc.tag)
            surfs.append(oc.info)

    return elems, surfs

def create_forest(comm, depth, htarget):
    '''
    Load the 3D step file and convert it to a 2D model
    '''

    geo1 = TMR.LoadModel('model0.step')
    geo2 = TMR.LoadModel('model1.step')
    geo3 = TMR.LoadModel('model2.step')
    geo4 = TMR.LoadModel('model3.step')
    geo5 = TMR.LoadModel('model4.step')
    geo6 = TMR.LoadModel('model5.step')
    all_geos = [geo1, geo2, geo3, geo4, geo5, geo6]

    # Create the full list of vertices, edges, faces and volumes
    verts = []
    edges = []
    faces = []
    vols = []
    for geo in all_geos:
        verts.extend(geo.getVertices())
        edges.extend(geo.getEdges())
        faces.extend(geo.getFaces())
        vols.extend(geo.getVolumes())

    # Combine the geometries and mesh the assembly
    TMR.setMatchingFaces(all_geos)

    # Create the geometry
    geo = TMR.Model(verts, edges, faces, vols)

    # Name the objects that we need to use
    faces[29].setName('heat_pipe')
    faces[35].setName('ambient')
    faces[23].setName('ambient')
    aerogel_faces = [16, 10, 4, 33, 15, 27, 9, 21, 3,
                     31, 13, 18, 0, 32, 14, 26, 8, 20, 2]
    for i in aerogel_faces:
        faces[i].setName('insulated')

    # Create the mesh object
    mesh = TMR.Mesh(comm, geo)

    # Mesh the part
    opts = TMR.MeshOptions()
    opts.write_mesh_quality_histogram = 1

    # Mesh the geometry with the given target size
    mesh.mesh(htarget, opts=opts)

    # Create a model from the mesh
    model = mesh.createModelFromMesh()

    # Create the corresponding mesh topology from the mesh-model
    topo = TMR.Topology(comm, model)

    # Create the oct forest and set the topology of the forest
    forest = TMR.OctForest(comm)
    forest.setTopology(topo)
    forest.createTrees(depth)

    # Loop over all of the forest volumes to add labels based on z-coord
    vols = model.getVolumes()
    for v in vols:
        min_z = 1e6
        max_z = -1e6
        faces = v.getFaces()
        for f in faces:
            for i in range(f.getNumEdgeLoops()):
                eloop = f.getEdgeLoop(i)
                edges, dirs = eloop.getEdgeLoop()
                for e in edges:
                    v1, v2 = e.getVertices()
                    x1 = v1.evalPoint()
                    x2 = v2.evalPoint()
                    min_z = min(min_z, x1[2], x2[2])
                    max_z = max(max_z, x1[2], x2[2])
        if min_z < 5.6e-3:
            v.setName("battery")
        elif max_z > 5.6e-3:
            v.setName("pcm")
        else:
            print("Volume did not meet either criteria")
            print(min_z, max_z)

    forest.setMeshOrder(2, TMR.GAUSS_LOBATTO_POINTS)

    return forest

class CreateMe(TMR.OctCreator):
    def __init__(self, bcs):
        TMR.OctCreator.__init__(bcs)

    def createElement(self, order, octant):
        """
        Create an element for the entire mesh
        """

        if octant.tag in pcm_tags:
            stiff = constitutive.SolidConstitutive(pcm_props)
        elif octant.tag in battery_tags:
            stiff = constitutive.SolidConstitutive(battery_props)
        else:
            print("oct not defined")

        # Create the model
        model = elements.HeatConduction3D(stiff)

        # Set the basis functions and create the element
        basis = elements.LinearHexaBasis()

        # Create the element type
        element = elements.Element3D(model, basis)

        return element

    def createMg(self, forest, nlevels=2, order=2):
        # Create the forest
        forest.balance(1)
        forest.repartition()
        forest.setMeshOrder(order)

        # Create the forests
        forests = [ forest ]
        assemblers = [ self.createTACS(forest) ]

        for i in range(nlevels-1):
            forests.append(forests[-1].coarsen())
            forests[-1].balance(1)
            forests[-1].repartition()
            forests[-1].setMeshOrder(2)
            assemblers.append(self.createTACS(forests[-1]))

        # Create the multigrid object
        mg = TMR.createMg(assemblers, forests)

        return assemblers[0], mg


def create_force(assembler, forest):

    # Get the basis object from one of the elements
    elems = assembler.getElements()
    basis = elems[0].getElementBasis()
    vpn = elems[0].getVarsPerNode()

    # Add the heat flux traction on the clamped end
    tractions = []
    trac = [Q]
    for findex in range(6):
        tractions.append(elements.Traction3D(vpn, findex,
                                             basis, trac))

    force = TopOptUtils.computeTractionLoad('battery',
                                            forest,
                                            assembler,
                                            tractions)

    f = force.getArray()

    return force

def solve(assembler, mg, forest):

    # Set the function of interest as the max pcm temperature
    kstemp = functions.KSTemperature(assembler, 100.0)
    elems, faces = get_elems_and_surfs(forest, ['pcm'])
    kstemp.setDomain(elems)

    # Set the design variables
    x_vec = assembler.createDesignVec()
    assembler.getDesignVars(x_vec)
    x = x_vec.getArray()
    x[:] = 1.0
    assembler.setDesignVars(x_vec)

    force = create_force(assembler, forest)

    u = assembler.createVec()
    assembler.zeroVariables()
    mat = mg.getMat()
    ksm = TACS.KSM(mat, mg, 100, 5, 0)
    mg.assembleJacobian(1.0, 0.0, 0.0, None)
    mg.factor()
    assembler.setBCs(force)
    ksm.solve(force, u)
    assembler.setBCs(u)
    assembler.setVariables(u)

    pcm_temp = assembler.evalFunctions([kstemp])[0]
    print("Max PCM temperature is {0}".format(pcm_temp))
    print("Max allowable temperate is {0}".format(Tmax))

    flag = (TACS.OUTPUT_CONNECTIVITY |
            TACS.OUTPUT_NODES |
            TACS.OUTPUT_DISPLACEMENTS |
            TACS.OUTPUT_EXTRAS)

    f5 = TACS.ToFH5(assembler,
                    TACS.SCALAR_3D_ELEMENT,
                    flag)
    f5.writeToFile('pcm.f5')

    return


# Create the communicator
comm = MPI.COMM_WORLD

# Create the forest
forest = create_forest(comm, 1, 2e-3)

# Set the boudnary conditions (None)
bcs = TMR.BoundaryConditions()
bcs.addBoundaryCondition('heat_pipe', bc_nums=[0], bc_vals=[dT])

pcm_tags = []
battery_tags = []
octants = forest.getOctants()
for oc in octants:
    if oc.z > 4.4e-3:
        pcm_tags.append(oc.tag)
    else:
        battery_tags.append(oc.tag)

pcm_tags = []
for oc in forest.getOctsWithName('pcm'):
    pcm_tags.append(oc.tag)

battery_tags = []
for oc in forest.getOctsWithName('battery'):
    battery_tags.append(oc.tag)

battery_octs = forest.getOctsWithName('battery')

pcm_props = constitutive.MaterialProperties(rho=bulk_rho, kappa=bulk_k)
battery_props = constitutive.MaterialProperties(rho=battery_rho, kappa=battery_k)

# Allocate the creator class
creator = CreateMe(bcs)

# Create the initial forest
nlevels = 2
forest.createTrees(nlevels-1)

# Create the TACS assembler
assembler, mg = creator.createMg(forest, nlevels=nlevels, order=2)

# Solve the steady-state conduction problem
solve(assembler, mg, forest)
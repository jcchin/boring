from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from tmr import TMR, TopOptUtils
from tacs import TACS, elements, constitutive, functions

# Constants and assumptions
# -------------------------

# Material properties
porosity = 0.90
k_foam = 401.0
rho_foam = 8960.0
k_pcm = 2.0
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

qdot_in = resistance*discharge_rate**2
h_pipe = 100.0
Tref = 0.0

def get_elems_and_surfs(names):
    # Get the element numbers and faces for a list of names
    elems = []
    surfs = []
    for n in names:
        octants = forest.getOctsWithName(n)
        for oc in octants:
            elems.append(oc.tag)
            surfs.append(oc.info)

    return elems, surfs

def addTractionAuxElements(names, forest,
                           aux, trac):

    octants = forest.getOctants()
    if isinstance(names, str):
        face_octs = forest.getOctsWithName(names)
    else:
        face_octs = []
        for name in names:
            face_octs.extend(forest.getOctsWithName(name))

    # Add the auxiliary elements
    for i in range(len(face_octs)):
        index = face_octs[i].tag
        if index is not None:
            aux.addElement(index, trac[face_octs[i].info])

    return

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

    # vols[0].setName('battery')
    # vols[1].setName('battery')
    # vols[2].setName('battery')
    # vols[3].setName('pcm')
    # vols[4].setName('pcm')
    # vols[5].setName('pcm')

    # Combine the geometries and mesh the assembly
    TMR.setMatchingFaces(all_geos)

    # Create the geometry
    geo = TMR.Model(verts, edges, faces, vols)

    #geo.writeModelToTecplot('model.dat', vlabels=False, elabels=False)

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
    # mesh.writeToVTK('battery_mesh.vtk')

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


def create_force(forest, assembler):

    force = assembler.createVec()
    assembler.zeroVariables()
    aux = TACS.AuxElements()

    # Get the basis object from one of the elements
    elems = assembler.getElements()
    basis = elems[0].getElementBasis()
    vpn = elems[0].getVarsPerNode()

    # Add the heat flux from the battery discharge
    tractions = []
    trac = [-qdot_in]
    for findex in range(6):
        tractions.append(elements.Traction3D(vpn, findex,
                                             basis, trac))

    addTractionAuxElements('battery', forest,
                           aux, tractions)

#     # Insulate the sides
#     tractions = []
#     trac = [0.0]
#     for findex in range(6):
#         tractions.append(elements.Traction3D(vpn, findex,
#                                              basis, trac))
#
#     addTractionAuxElements(['insulated', 'ambient'],
#                            forest, aux, tractions)

    # Remove the heat from the heat pipe
    tractions = []
    for findex in range(6):
        tractions.append(elements.ConvectiveTraction3D(vpn, findex,
                                                       vpn-1, h_pipe,
                                                       Tref, basis))

    addTractionAuxElements('heat_pipe', forest, aux, tractions)

    # Set the auxilliary elements and get the force vector
    assembler.setAuxElements(aux)
    assembler.assembleRes(force)
    force.scale(-1.0)

    return force

def integrate(assembler, forest, tfinal=300.0,
              nsteps=30, output=False):

    # Create the BDF integrator
    tinit = 0.0
    order = 2
    bdf = TACS.BDFIntegrator(assembler,
                             tinit, tfinal,
                             nsteps, order)
    bdf.setPrintLevel(0)
    bdf.setAbsTol(1e-6)
    bdf.setRelTol(1e-15)
    if output:
        # Set the output file name
        flag = (TACS.OUTPUT_CONNECTIVITY |
                TACS.OUTPUT_NODES |
                TACS.OUTPUT_DISPLACEMENTS |
                TACS.OUTPUT_EXTRAS)
        f5 = TACS.ToFH5(assembler,
                        TACS.PLANE_STRESS_ELEMENT,
                        flag)
        bdf.setFH5(f5)
        bdf.setOutputFrequency(1)
        bdf.setOutputPrefix('time_history/')

    # Create a vector that will store the instantaneous traction
    # load at any point in time
    forces = create_force(forest, assembler)

    # Iterate in time to march the equations of motion forward
    # in time
    t_array = np.linspace(tinit, tfinal,
                          nsteps+1)
    for i, t in enumerate(t_array):
        # Iterate forward in time for one time step
        print("Starting time step {0}".format(i))
        bdf.iterate(i, forces=forces)

    # Extract the time history
    qvals = np.zeros(nsteps+1)
    tvals = np.zeros(nsteps+1)
    for time_step in range(nsteps+1):
        # Extract vectors
        time, q , _, _ = bdf.getStates(time_step)
        # Extract Arrays
        qarray = q.getArray()
        qvals[time_step] = np.amax(qarray)
        tvals[time_step] = time

    return tvals, qvals, q


# Create the communicator
comm = MPI.COMM_WORLD

# Create the forest
forest = create_forest(comm, 1, 4e-3)

# Set the boudnary conditions (None)
bcs = TMR.BoundaryConditions()

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

force = create_force(forest, assembler)

# Set the design vars to 1
dv_vec = assembler.createDesignVec()
x = dv_vec.getArray()
x[:] = 1.0
assembler.setDesignVars(dv_vec)

tvals, qvals, qfinal = integrate(assembler, forest)

plt.style.use('mark')
fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
ax.plot(tvals, qvals)
plt.show()

assembler.setVariables(qfinal)

# Output for visualization
flag = (TACS.OUTPUT_CONNECTIVITY |
        TACS.OUTPUT_NODES |
        TACS.OUTPUT_DISPLACEMENTS)

f5 = TACS.ToFH5(assembler,
                TACS.SCALAR_3D_ELEMENT,
                flag)
f5.writeToFile('pouch_cell.f5')

# # Solve the steady-state conduction problem
# u = assembler.createVec()
# assembler.zeroVariables()
# mat = mg.getMat()
# ksm = TACS.KSM(mat, mg, 100, 5, 0)
# mg.assembleJacobian(1.0, 0.0, 0.0, None)
# mg.factor()
# assembler.setBCs(force)
# ksm.solve(force, u)
# assembler.setBCs(u)
# assembler.setVariables(u)
#
# # Get the maximum temperature of the PCM
# print("Max temp = {0}".format(np.amax(u.getArray())))
#
# # Output for visualization
# flag = (TACS.OUTPUT_CONNECTIVITY |
#         TACS.OUTPUT_NODES |
#         TACS.OUTPUT_DISPLACEMENTS)
#
# f5 = TACS.ToFH5(assembler,
#                 TACS.SCALAR_3D_ELEMENT,
#                 flag)
# f5.writeToFile('pouch_cell.f5')


from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from tmr import TMR, TopOptUtils
from tacs import TACS, elements, constitutive, functions

# TODO:
# - add functions to evaluate max temperature in different domains
# - make batteries a different material
# - add air gap resistance
# - initial conditions?

def create_forest(comm, depth, htarget):
    '''
    Load the 3D step file and convert it to a 2D model
    '''

    geo = TMR.LoadModel('battery_3d.step')
    verts = []
    edges = []
    faces = []
    all_faces = geo.getFaces()

    # Select only the facs on one face
    faces.append(all_faces[3]) # The structure's face
    faces.extend(all_faces[(len(all_faces)-9):len(all_faces)]) # The face of the 9 batteries

    # Add only the verts and edges associated with the faces that we selected
    vert_ids = []
    edge_ids = []
    for f in faces:
        num_loops = f.getNumEdgeLoops()
        for i in range(num_loops):
            el = f.getEdgeLoop(i)
            edge_list, _ = el.getEdgeLoop()
            for e in edge_list:
                if e.getEntityId() not in edge_ids:
                    edges.append(e)
                    edge_ids.append(e.getEntityId())
                v1, v2 = e.getVertices()
                if v1.getEntityId() not in vert_ids:
                    verts.append(v1)
                    vert_ids.append(v1.getEntityId())
                if v2.getEntityId() not in vert_ids:
                    verts.append(v2)
                    vert_ids.append(v2.getEntityId())

    # Create the new 2D geometry model
    geo = TMR.Model(verts, edges, faces)

    # Name the faces that we need to use
    faces[7].setName('battery_0') # Corner battery face
    faces[8].setName('battery_1') # Adjacent battery
    faces[1].setName('battery_2') # Diagonal battery
    faces[2].setName('battery') # All the other batteries
    faces[3].setName('battery')
    faces[4].setName('battery')
    faces[5].setName('battery')
    faces[6].setName('battery')
    faces[9].setName('battery')

    # Create the mesh object
    mesh = TMR.Mesh(comm, geo)

    # Mesh the part
    opts = TMR.MeshOptions()
    opts.write_mesh_quality_histogram = 1

    # Mesh the geometry with the given target size
    mesh.mesh(htarget, opts=opts)
    mesh.writeToVTK('battery_mesh.vtk')

    # Create a model from the mesh
    model = mesh.createModelFromMesh()

    # Create the corresponding mesh topology from the mesh-model
    topo = TMR.Topology(comm, model)

    # Create the quad forest and set the topology of the forest
    forest = TMR.QuadForest(comm)
    forest.setTopology(topo)
    forest.createTrees(depth)

    return forest

class CreateMe(TMR.QuadCreator):
    def __init__(self, bcs):
        TMR.QuadCreator.__init__(bcs)

    def createElement(self, order, quad):
        """
        Create an element for the entire mesh
        """

        thickness = 0.065
        rho = 2700.0*thickness
        E = 72.4e9
        nu = 0.33
        ys = 345e6
        aT = 24e-6
        c = 883.0
        kcond = 204.0
        props = constitutive.MaterialProperties(rho=rho, E=E, nu=nu,
                                                ys=ys, alpha=aT, kappa=kcond,
                                                specific_heat=c)
        stiff = constitutive.PlaneStressConstitutive(props)

        # Create the model
        model = elements.HeatConduction2D(stiff)

        # Set the basis functions and create the element
        if order == 2:
            basis = elements.LinearQuadBasis()
        elif order == 3:
            basis = elements.QuadraticQuadBasis()
        elif order == 4:
            basis = elements.CubicQuadBasis()
        elif order == 5:
            basis = elements.QuarticQuadBasis()
        elif order == 6:
            basis = elements.QuinticQuadBasis()

        # Create the element type
        element = elements.Element2D(model, basis)

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


def qdot_in_func(t):
    '''
    Overshoot heat flux function with steady state
    '''
    qmax = 6000.0

    qdot_in = 0.0
    if t <= 2.0:
        qdot_in = qmax

    return qdot_in

def update_force(forest, assembler, qdot_in=0.0):

    # Get the basis object from one of the elements
    elems = assembler.getElements()
    basis = elems[0].getElementBasis()
    vpn = elems[0].getVarsPerNode()

    # Add the heat flux traction on the clamped end
    tractions = []
    for findex in range(4):
        tractions.append(elements.Traction2D(vpn, findex,
                                             basis, [-qdot_in]))

    force = TopOptUtils.computeTractionLoad('battery_0',
                                            forest,
                                            assembler,
                                            tractions)

    return force

def integrate(assembler, forest, tfinal=30.0,
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
                        TACS.SCALAR_2D_ELEMENT,
                        flag)
        bdf.setFH5(f5)
        bdf.setOutputFrequency(1)
        bdf.setOutputPrefix('time_history/')

    # Create a vector that will store the instantaneous traction
    # load at any point in time
    forces = assembler.createVec()

    # Compute the tractions due to a unit input heat flux
    unit_forces = update_force(forest,
                               assembler,
                               qdot_in=1.0)

    # Iterate in time to march the equations of motion forward
    # in time
    t_array = np.linspace(tinit, tfinal,
                          nsteps+1)
    for i, t in enumerate(t_array):
        # Compute the magnitude of the input heat flux
        q_in = qdot_in_func(t)

        # Copy the unit force values and scale by the heat flux
        forces.copyValues(unit_forces)
        forces.scale(q_in)

        # Iterate forward in time for one time step
        bdf.iterate(i, forces=forces)

    qvals = np.zeros(nsteps+1)
    tvals = np.zeros(nsteps+1)
    for time_step in range(nsteps+1):
        # Extract vectors
        time, q , _, _ = bdf.getStates(time_step)
        # Extract Arrays
        qarray = q.getArray()
        qvals[time_step] = np.amax(qarray)
        tvals[time_step] = time

    return tvals, qvals

# Create the communicator
comm = MPI.COMM_WORLD

# Create the forest
forest = create_forest(comm, 1, 0.0005)
forest.setMeshOrder(2, TMR.GAUSS_LOBATTO_POINTS)

# Set the boudnary conditions (None)
bcs = TMR.BoundaryConditions()

# Allocate the creator class
creator = CreateMe(bcs)

# Create the initial forest
nlevels = 2
forest.createTrees(nlevels-1)

# Create the TACS assembler
assembler, mg = creator.createMg(forest, nlevels=nlevels, order=2)

# Set the design vars to 1
dv_vec = assembler.createDesignVec()
x = dv_vec.getArray()
x[:] = 1.0
assembler.setDesignVars(dv_vec)

# Perform the numerical integration
tfinal = 5.0
nsteps = 50
t, u = integrate(assembler, forest, nsteps=nsteps,
                 tfinal=tfinal, output=True)

# Plot the maximum temperature over time
plt.style.use('mark')
fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
ax.plot(t, u)
ax.set_xlabel(r"Time (s)")
ax.set_ylabel(r"$\theta_{max}(t)$")
plt.savefig('time_response.pdf', transparent=False)
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from tmr import TMR, TopOptUtils
from tacs import TACS, elements, constitutive, functions

# TODO:
# - add air gap resistance
# - initial conditions

def get_elems_and_surfs(names):
    # Get the element numbers and faces for a list of names
    elems = []
    surfs = []
    for n in names:
        quads = forest.getQuadsWithName(n)
        for q in quads:
            elems.append(q.tag)
            surfs.append(q.info)

    return elems, surfs

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
    faces[0].setName('aluminum')

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

        if quad.tag in alum_tags:
            stiff = constitutive.PlaneStressConstitutive(alum_props)
        elif quad.tag in battery_tags:
            stiff = constitutive.PlaneStressConstitutive(battery_props)
        else:
            print("quad not defined")

        # Create the model
        model = elements.LinearThermoelasticity2D(stiff)

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
    qmax = 6000.0*thickness

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
    trac = [0.0]*vpn
    trac[-1] = -qdot_in
    for findex in range(4):
        tractions.append(elements.Traction2D(vpn, findex,
                                             basis, trac))

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
                        TACS.PLANE_STRESS_ELEMENT,
                        flag)
        bdf.setFH5(f5)
        bdf.setOutputFrequency(1)
        bdf.setOutputPrefix('time_history/')

    # Define the functions of interest
    temp0 = functions.KSTemperature(assembler, 50.0)
    temp0.setKSTemperatureType('discrete')
    elems, _ = get_elems_and_surfs(['battery_0'])
    temp0.setDomain(elems)
    temp1 = functions.KSTemperature(assembler, 50.0)
    temp1.setKSTemperatureType('discrete')
    elems, _ = get_elems_and_surfs(['battery_1'])
    temp1.setDomain(elems)
    temp2 = functions.KSTemperature(assembler, 50.0)
    temp2.setKSTemperatureType('discrete')
    elems, _ = get_elems_and_surfs(['battery_2'])
    temp2.setDomain(elems)

    # Set the functions into the integrator class
    bdf.setFunctions([temp0, temp1, temp2])

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

    # Compute the nodal sensitivities
    fvals = bdf.evalFunctions([temp0, temp1, temp2])
    bdf.integrateAdjoint()
    df0dXpt = bdf.getXptGradient(0)
    df1dXpt = bdf.getXptGradient(1)
    df2dXpt = bdf.getXptGradient(2)
    dfdXpt = [df0dXpt, df1dXpt, df2dXpt]

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

    # Evaluate the functions at every time step
    temp0_vals = np.zeros(nsteps+1)
    temp1_vals = np.zeros(nsteps+1)
    temp2_vals = np.zeros(nsteps+1)
    for time_step in range(nsteps+1):
        # Extract vectors
        _, q , qdot, qddot = bdf.getStates(time_step)
        # Compute the function values
        assembler.setVariables(q)
        fvals = assembler.evalFunctions([temp0, temp1, temp2])
        temp0_vals[time_step] = fvals[0]
        temp1_vals[time_step] = fvals[1]
        temp2_vals[time_step] = fvals[2]

    fvals = [temp0_vals, temp1_vals, temp2_vals]

    return tvals, qvals, fvals, dfdXpt

# Create the communicator
comm = MPI.COMM_WORLD

# Create the forest
forest = create_forest(comm, 1, 0.0005)
forest.setMeshOrder(2, TMR.GAUSS_LOBATTO_POINTS)

# Set the boudnary conditions (None)
bcs = TMR.BoundaryConditions()

alum_tags = []
for quad in forest.getQuadsWithName('aluminum'):
    alum_tags.append(quad.tag)

battery_tags = []
for n in ['battery_0', 'battery_1', 'battery_2', 'battery']:
    for quad in forest.getQuadsWithName(n):
        battery_tags.append(quad.tag)

thickness = 0.065
alum_rho = 2700.0*thickness
battery_rho = 1460.0*thickness
alum_props = constitutive.MaterialProperties(rho=alum_rho, E=72.4e9, nu=0.33,
                                             ys=345e6, alpha=24e-6, kappa=204.0,
                                             specific_heat=883.0)
battery_props = constitutive.MaterialProperties(rho=battery_rho, kappa=1.3,
                                                specific_heat=880.0)

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
t, u, fvals, dfdXpt = integrate(assembler, forest, nsteps=nsteps,
                                tfinal=tfinal, output=True)

flag = (TACS.OUTPUT_CONNECTIVITY |
        TACS.OUTPUT_NODES |
        TACS.OUTPUT_DISPLACEMENTS |
        TACS.OUTPUT_EXTRAS)

assembler.setVariables(dfdXpt[0])
f5 = TACS.ToFH5(assembler,
                TACS.PLANE_STRESS_ELEMENT,
                flag)
f5.writeToFile('Xpt0.f5')

assembler.setVariables(dfdXpt[1])
f5 = TACS.ToFH5(assembler,
                TACS.PLANE_STRESS_ELEMENT,
                flag)
f5.writeToFile('Xpt1.f5')

assembler.setVariables(dfdXpt[2])
f5 = TACS.ToFH5(assembler,
                TACS.PLANE_STRESS_ELEMENT,
                flag)
f5.writeToFile('Xpt2.f5')

# Plot the maximum temperature over time
i0 = np.argmax(fvals[0])
i1 = np.argmax(fvals[1])
i2 = np.argmax(fvals[2])

plt.style.use('mark')
fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
ax.plot(t, fvals[0], label='Corner cell')
ax.plot(t, fvals[1], label='Adjacent cell')
ax.plot(t, fvals[2], label='Diagonal cell')
ax.scatter(t[i0], fvals[0][i0], color='C0', zorder=10, s=30, edgecolors='white', linewidths=1.5)
ax.scatter(t[i1], fvals[1][i1], color='C1', zorder=10, s=30, edgecolors='white', linewidths=1.5)
ax.scatter(t[i2], fvals[2][i2], color='C2', zorder=10, s=30, edgecolors='white', linewidths=1.5)

ax.set_xticks([0.0, t[i0], t[i2], 5.0])
ax.set_yticks([0.0, fvals[0][i0], fvals[1][i1], fvals[2][i2]])
ax.set_yticklabels(['0.0', '{0:.1f}'.format(fvals[0][i0]), '{0:.1f}'.format(fvals[1][i1]), '{0:.1f}'.format(fvals[2][i2])])

ax.set_xlabel(r"Time (s)")
ax.set_ylabel(r"$\Delta$T$_{max}(t)$ ($^\circ$C)")
ax.legend()
plt.savefig('time_response.pdf', transparent=False)

""""
Parallel DOE for FENICS runs

Requires all meshes to be generated ahead of time with
make_mesh.py

Author: Jeff Chin
"""

import openmdao.api as om
from mpi4py import MPI
from boring.fenics.fenics_baseline import FenicsBaseline

from openmdao.test_suite.components.paraboloid import Paraboloid


prob = om.Problem()
nn=1
prob.model.add_subsystem('comp', FenicsBaseline(num_nodes=nn, cluster=True),
                         promotes_inputs=['extra', 'ratio'],
                         promotes_outputs=['temp2_data'])
prob.model.add_design_var('extra', lower=1.05, upper=1.5)
prob.model.add_design_var('ratio', lower=0.05, upper=0.95)
prob.model.add_objective('temp2_data')
# prob.model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])
# prob.model.add_design_var('x', lower=0.0, upper=1.0)
# prob.model.add_design_var('y', lower=0.0, upper=1.0)
# prob.model.add_objective('f_xy')

prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=3))
prob.driver.options['run_parallel'] = True
prob.driver.options['procs_per_model'] = 1

prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

prob.setup(force_alloc_complex=True)
prob.run_model()
#prob.run_driver()

prob.cleanup()

# print(MPI.COMM_WORLD.size)

# check recorded cases from each case file
rank = MPI.COMM_WORLD.rank
filename = "cases.sql_%d" % rank
# print(filename)
cr = om.CaseReader(filename)
cases = cr.list_cases('driver',out_stream=None)
print(len(cases))
values = []
for case in cases:
    outputs = cr.get_case(case).outputs
    values.append((outputs['extra'], outputs['ratio'], outputs['temp2_data']))

print("\n"+"\n".join(["extra: %5.2f, ratio: %5.2f, temp2: %6.2f" % xyf for xyf in values]))
from boring.metamodel.metaOptGroup import MetaOptimize

import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import more_itertools as mit

if __name__ == "__main__":
    p = om.Problem()
    model = p.model
    nn = 1


    p.model.add_subsystem(name='meta_optimize',
                          subsys=MetaOptimize(num_nodes=nn),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
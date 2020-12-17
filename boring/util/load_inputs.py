import importlib
import openmdao.api as om
import numpy as np

"""
load inputs from a provided file,
the loaded file must contain a dictionary called "inputs"
the first value of each key/value is the variable value, the second is the optional variable units
this function can only be called after setup

example:

load_inputs('boring.input.assumptions',p)


Author: Jeff Chin
"""


def load_inputs(filename, prob, nn=1):
    x = importlib.import_module(filename)

    myDict = getattr(x, 'inputs')

    for key, val in myDict.items():
        if len(val) == 1:  # no units
            prob.set_input_defaults(name=key, val=val * np.ones(nn))
        if len(val) == 2:  # units provided
            prob.set_input_defaults(name=key, val=val[0] * np.ones(nn), units=val[1])

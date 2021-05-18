

import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from boring.src.sizing.static_sizing import StaticSizing

class TestSizing(unittest.TestCase):
    """ Check general sizing groups for insulation and heatpipe, similar to the first two
    subsystems of build_pack.py"""

    def setUp(self):
        p1 = self.prob = om.Problem(model=om.Group())
        p1.model.add_subsystem('size', subsys=StaticSizing(num_nodes=1))

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def _test_tot_mass(self):  # calculation regression test
    # Cells weigh 31.6g

        print(self.prob.get_val('size.mass_total'))



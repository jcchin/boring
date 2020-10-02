from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_rel_error

from boring.src.model.cell_comp import CellComp


class TestBatteryCellComp(unittest.TestCase):

    def setUp(self):
        n = 1

        prob = self.prob = Problem(model=Group())

        cell_comp = CellComp(num_nodes=n)

        ivc = IndepVarComp()
        ivc.add_output('I_pack', val=2.*np.ones(n), units='A', desc='Line current')
        ivc.add_output('R_0', val=0.095*np.ones(n), units='ohm', desc='resistance at reference')
        ivc.add_output('R_Th', val=0.095*np.ones(n), units='ohm', desc='Thevenin resistance')
        
        prob.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['*'])
        prob.model.add_subsystem(name='cell', subsys=cell_comp, promotes_inputs=['*'])

        prob.setup(force_alloc_complex=True)

        prob.run_model()

    def test_cell_vals(self):

        print(self.prob.get_val('cell.pack_eta'))
        assert_rel_error(self, self.prob.get_val('cell.pack_eta', units=None), [0.75847356], tolerance=1.0E-5)
        
    def test_cell_partials(self):

        self.prob['R_Th'] = 0.001*np.ones(1)

        data = self.prob.check_partials(compact_print=False, method='cs', step=1e-50) #out_stream=None,
        assert_check_partials(data, atol=1e-10, rtol=1e-10)




if __name__ == '__main__': # pragma: no cover
    unittest.main()
from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from boring.src.model.reg_thevenin_interp_group import RegTheveninInterpGroup
from boring.src.model.maps.s18650_battery import battery
from numpy.testing import assert_almost_equal
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials


class TestRegComp(unittest.TestCase):

    def setUp(self):
        n = 60

        prob = self.prob = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output('SOC', val=np.ones(n), units=None)
        ivc.add_output('T_batt', val=np.ones(n), units='degC')

        prob.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['*'])
        prob.model.add_subsystem(name='interp', subsys=RegTheveninInterpGroup(num_nodes=n))

        prob.model.connect('SOC', 'interp.SOC')
        prob.model.connect('T_batt', 'interp.T_batt')
        prob.setup(force_alloc_complex=True)

        prob['SOC'] = np.random.rand(n)
        prob['T_batt'] = np.random.rand(n)*60.0

        prob.run_model()

    def test_results(self):
         
        import matplotlib
        matplotlib.use('agg') # <--- comment out if you want to show this plot.
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xx, yy = np.meshgrid(battery.SOC_bp, battery.T_bp)
        ax.scatter(self.prob['SOC'], self.prob['T_batt'], self.prob['interp.R_0'], color='r')
        ax.plot_surface(xx, yy, battery.tR_0.reshape(xx.shape))
        plt.title('RegGrid')
        plt.show()

        #spot check shape and points
        assert_rel_error(self, battery.tR_0.shape, [5,33], tolerance=1.0E-6)
        assert_rel_error(self, battery.tR_0[-1][-1], 0.03, tolerance=1.0E-3)

    def test_derivs(self):
        
        data = self.prob.check_partials(out_stream=None)
        assert_check_partials(data, atol=1e-6, rtol=1e-6) # this is really just a metamodel comp


if __name__ == '__main__': # pragma: no cover
    unittest.main()

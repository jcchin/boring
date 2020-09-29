from __future__ import print_function, division, absolute_import

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import openmdao.api as om 

from openmdao.utils.general_utils import set_pyoptsparse_opt

from dymos import Phase, Radau

from aviary_quadrotor.subsystems.battery.battery_group import BatteryGroup


class TestBatteryODE(unittest.TestCase):

    def test_battery_power(self):
        """
            for battery explicit integration testings
        """
        _, local_opt = set_pyoptsparse_opt('SNOPT')
        if local_opt != 'SNOPT':
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT")

        p = om.Problem()

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SNOPT'
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['Major optimality tolerance'] = 5.0E-3
        p.driver.opt_settings['Major feasibility tolerance'] = 1e-6
        p.driver.opt_settings['iSumm'] = 6

        transcription = Radau(num_segments=15, order=3, compressed=True)

        phase0 = Phase(transcription=transcription, ode_class=BatteryGroup)

        phase0.set_time_options(fix_initial=True, duration_bounds=(30, 30))

        p.model.add_subsystem(name='phase0', subsys=phase0)

        phase0.add_state('SOC', fix_initial=True, rate_source='dXdt:SOC', lower=0.0, upper=1.)
        phase0.add_state('U_Th', units='V', fix_initial=False, rate_source='dXdt:V_{thev}', lower=0.0, upper=5.0)

        # phase0.add_parameter('P_out', units='W', opt=False)

        # phase0.add_boundary_constraint('U_pack', units='V', loc='initial', equals=5100)

        phase0.add_objective('time', loc='final', ref=1)

        p.model.linear_solver = om.DirectSolver(assemble_jac=True)

        phase0.add_timeseries_output('Q_{batt}', output_name='Q_{batt}', units='W')
        # phase0.add_timeseries_output('U_pack', output_name='V', units='V')

        p.setup()

        # p.check_partials()

        T0 = 10+273

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 30

        p['phase0.states:SOC'] = phase0.interpolate(ys=[1.0,0.0], nodes='state_input')
        p['phase0.states:U_Th'] = phase0.interpolate(ys=[0.1,0.1], nodes='state_input')
        # p['phase0.parameters:P_out'][:] = 72000.

        p.run_driver()

        fig, ax = plt.subplots(3,1, sharex=True)
        fig.suptitle('Temperature Plots')

        t_opt = p.get_val('phase0.timeseries.time')
        SOC_opt = p.get_val('phase0.timeseries.states:SOC', units=None)

        Q_batt_opt = p.get_val('phase0.timeseries.Q_{batt}', units='kW')


        ax[1].plot(t_opt, Q_batt_opt*128*40, 'r', label='$Q_{cell}$')

        ax[2].plot(t_opt, SOC_opt, 'r', label='$SOC$')

        #spot check final values
        # assert_rel_error(self, T_batt_opt[-1], 1.25934406, tolerance=1.0E-6)

        # ax[3].plot(t_opt, V_opt, 'r', label='$Voltage$')


        # axarr = fig.add_subplot(1, 2, 2)
        # axarr.plot(sim_out.get_values('time'),sim_out.get_values('electric.battery.I_Li'), 'b')
        # # # axarr.plot(p['phase0.state_interp.state_col:r'],
        # # #            p['phase0.controls:h'], 'bo', ms=4)
        # axarr.set_ylabel('I_Li, amps')
        # axarr.set_xlabel('time, s')
        # axarr.axes.get_xaxis().set_visible(True)

        import matplotlib
        matplotlib.use('agg') # <--- comment out if you want to show this plot.
        plt.show()


if __name__ =='__main__':
    unittest.main()
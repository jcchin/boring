from __future__ import print_function, division, absolute_import

import unittest
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from boring.util.spec_test import assert_match_spec
from boring.src.sizing.thermal_resistance.radial_thermal_resistance import RadialThermalResistance


class TestRoundRadialResistance(unittest.TestCase):

    def setUp(self):
        p1 = self.prob = Problem(model=Group())
        p1.model.add_subsystem('cond_thermal', subsys=RadialThermalResistance(num_nodes=40, geom='round'), promotes=['*'])

        p1.setup(force_alloc_complex=True)
        p1.run_model()

    def test_round_cond_outputs(self):
        use_poly = True

        if not use_poly:  # use thermo package
            from thermo.chemical import Chemical

        ######################################## Solid and Wick Properties ########################################################
        k_w = 11.4
        epsilon = 0.46
        ######################################## Overall Geometry ########################################################
        L_evap = 0.02
        L_cond = 0.02
        L_adiabatic = 0.03
        L_eff = (L_evap + L_cond) / 2 + L_adiabatic
        t_w = 0.0005
        t_wk = 0.00069
        D_od = 0.006
        D_v = 0.00362  # make this a calculation
        r_i = (D_od / 2 - t_w)
        A_cond = np.pi * D_od * L_cond 
        A_evap = np.pi * D_od * L_evap 
        ######################################## Heat Pipe Core Geometry ########################################################
        A_w = np.pi * ((D_od / 2) ** 2 - (D_od / 2 - t_w) ** 2)
        A_wk = np.pi * ((D_od / 2 - t_w) ** 2 - (D_v / 2) ** 2)
        A_interc = np.pi * D_v * L_cond
        A_intere = np.pi * D_v * L_evap
        ######################################## External Convection at Condenser ########################################################
        h_c = 1200
        T_coolant = 285

        def f(T, a_0, a_1, a_2, a_3, a_4, a_5):
            poly = np.exp(a_0 + a_1 * T + a_2 * T ** 2 + a_3 * T ** 3 + a_4 * T ** 4 + a_5 * T ** 5)
            return poly

        alpha_array = []
        h_fg_array = []
        T_hp_array = []
        v_fg_array = []
        R_g_array = []
        P_v_array = []
        D_od_array = []
        r_i_array = []
        k_w_array = []
        L_cond_array = []
        D_v_array = []
        k_wk_array = []
        A_interc_array = []
        k_l_array = []
        epsilon_array = []
        h_interc_array = []
        R_wc_array = []
        R_wkc_array = []
        R_interc_array = []

        i = 1

        for Q_hp in range(10, 50, 1):

            T_cond = Q_hp / (A_cond * h_c) + T_coolant
            T_hp = T_cond
            T_hpfp = T_hp - 273.15

            ######################################## Fluid Properties for Water. From Faghri's book, Table C.70, page 909 ########################################################
            if use_poly:
                P_v = f(T_hpfp, -5.0945, 7.2280e-2, -2.8625e-4, 9.2341e-7, -2.0295e-9, 2.1645e-12) * 1e5  # Ezra
                h_fg = f(T_hpfp, 7.8201, -5.8906e-4, -9.1355e-6, 8.4738e-8, -3.9635e-10, 5.9150e-13) * 1e3  # Ezra
                rho_l = f(T_hpfp, 6.9094, -2.0146e-5, -5.9868e-6, 2.5921e-8, -9.3244e-11, 1.2103e-13)  # Ezra
                rho_v = f(T_hpfp, -5.3225, 6.8366e-2, -2.7243e-4, 8.4522e-7, -1.6558e-9, 1.5514e-12)  # Ezra
                mu_l = f(T_hpfp, -6.3530, -3.1540e-2, 2.1670e-4, -1.1559e-6, 3.7470e-9, -5.2189e-12)  # Ezra
                mu_v = f(T_hpfp, -11.596, 2.6382e-3, 6.9205e-6, -6.1035e-8, 1.6844e-10, -1.5910e-13)  # Ezra
                k_l = f(T_hpfp, -5.8220e-1, 4.1177e-3, -2.7932e-5, 6.5617e-8, 4.1100e-11, -3.8220e-13)  # Ezra
                k_v = f(T_hpfp, -4.0722, 3.2364e-3, 6.3860e-6, 8.5114e-9, -1.0464e-10, 1.6481e-13)  # Ezra
                sigma_l = f(T_hpfp, 4.3438, -3.0664e-3, 2.0743e-5, -2.5499e-7, 1.0377e-9, -1.7156e-12) / 1e3  # Ezra
                cp_l = f(T_hpfp, 1.4350, -3.2231e-4, 6.1633e-6, -4.4099e-8, 2.0968e-10, -3.040e-13) * 1e3  # Ezra
                cp_v = f(T_hpfp, 6.3198e-1, 6.7903e-4, -2.5923e-6, 4.4936e-8, 2.2606e-10, -9.0694e-13) * 1e3  # Ezra
            else:
                hp_fluid = Chemical('water')
                hp_fluid.calculate(T_hp)
                P_v = hp_fluid.Psat
                hp_fluid.calculate(T_hp, P_v)
                rho_v = hp_fluid.rhog
                mu_v = hp_fluid.mug
                rho_l = hp_fluid.rhol
                mu_l = hp_fluid.mul
                k_l = hp_fluid.kl
                cp_l = hp_fluid.Cpl
                cp_v = hp_fluid.Cpg
                cv = hp_fluid.Cvg
                Pr_l = cp_l * mu_l / k_l
                h_fg = hp_fluid.Hvap
                sigma_l = hp_fluid.sigma

            v_fg = 1 / rho_v - 1 / rho_l
            R_g = P_v / (T_hp * rho_v)
            cv_v = cp_v - R_g
            gamma = cp_v / cv_v

            ######################################## Axial Thermal Resistances ########################################################
            k_wk = (1 - epsilon) * k_w + epsilon * k_l
            ######################################## Condenser Section Thermal Resistances ########################################################
            alpha = 1  # Look into this, need better way to determine this rather than referencing papers.
            h_interc = 2 * alpha / (2 - alpha) * (h_fg ** 2 / (T_hp * v_fg)) * np.sqrt(1 / (2 * np.pi * R_g * T_hp)) * (
                        1 - P_v * v_fg / (2 * h_fg))
            R_wc = np.log((D_od / 2) / (r_i)) / (2 * np.pi * k_w * L_cond)
            R_wkc = np.log((r_i) / (D_v / 2)) / (2 * np.pi * k_wk * L_cond)
            R_interc = 1 / (h_interc * A_interc)

            alpha_array.append(alpha)
            h_fg_array.append(h_fg)
            T_hp_array.append(T_hp)
            v_fg_array.append(v_fg)
            R_g_array.append(R_g)
            P_v_array.append(P_v)
            D_od_array.append(D_od)
            r_i_array.append(r_i)
            k_w_array.append(k_w)
            L_cond_array.append(L_cond)
            D_v_array.append(D_v)
            k_wk_array.append(k_wk)
            A_interc_array.append(A_interc)
            k_l_array.append(k_l)
            epsilon_array.append(epsilon)

            h_interc_array.append(h_interc)
            R_wc_array.append(R_wc)
            R_wkc_array.append(R_wkc)
            R_interc_array.append(R_interc)

        #self.prob['axial_thermal.epsilon'] = 0.46
        self.prob.set_val('alpha', alpha_array)
        self.prob.set_val('LW:L_flux', L_cond_array)
        self.prob.set_val('h_fg', h_fg_array)
        self.prob.set_val('T_hp', T_hp_array)
        self.prob.set_val('v_fg', v_fg_array)
        self.prob.set_val('R_g', R_g_array)
        self.prob.set_val('P_v', P_v_array)
        self.prob.set_val('XS:D_od', D_od_array)
        self.prob.set_val('XS:r_i', r_i_array)
        self.prob.set_val('k_w', k_w_array)
        self.prob.set_val('k_l', k_l_array)
        self.prob.set_val('epsilon', epsilon_array)
        self.prob.set_val('XS:D_v', D_v_array)
        self.prob.set_val('k_wk', k_wk_array)
        self.prob.set_val('LW:A_inter', A_interc_array)
        self.prob.run_model()

        # assert_near_equal(self.prob.get_val('axial_thermal.k_wk'), 6.442931876303132, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('h_inter'), h_interc_array, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('R_w'), R_wc_array, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('R_wk'), R_wkc_array, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('R_inter'), R_interc_array, tolerance=1.0E-5)

    def test_partials(self):  # derivative check

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)

    # def test_io_spec(self): 

    #     subsystem = condThermalResistance(num_nodes=1)
    #     assert_match_spec(subsystem, 'Design_specs/struct.json')


if __name__ == '__main__':
    unittest.main()

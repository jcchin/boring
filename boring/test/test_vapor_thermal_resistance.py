from __future__ import print_function, division, absolute_import

import unittest
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from boring.util.spec_test import assert_match_spec
from boring.src.sizing.thermal_resistance.vapor_thermal_resistance import VaporThermalResistance


class TestFlatVaporResistance(unittest.TestCase):

    def setUp(self):
        p2 = self.prob = Problem(model=Group())
        p2.model.add_subsystem('cond_thermal', subsys=VaporThermalResistance(num_nodes=40, geom='round'), promotes=['*'])

        p2.setup(force_alloc_complex=True)
        p2.run_model()

    def test_flat_vapor_outputs(self):
        ######################################## Solid and Wick Properties ########################################################
        k_w=11.4
        epsilon=0.46
        ######################################## Overall Geometry ########################################################
        L_flux = L_cond = L_evap= 0.02
        L_adiabatic=0.03
        L_eff=(L_evap+L_cond)/2+L_adiabatic
        t_w=0.0005
        t_wk=0.00069
        D_od=0.006
        D_v=0.00362
        r_i=(D_od/2-t_w)
        A_cond=np.pi*D_od*L_cond
        A_evap=np.pi*D_od*L_evap
        ######################################## Heat Pipe Core Geometry ########################################################
        A_w=np.pi*((D_od/2)**2-(D_od/2-t_w)**2)
        A_wk=np.pi*((D_od/2-t_w)**2-(D_v/2)**2)
        A_interc=np.pi*D_v*L_cond
        A_intere=np.pi*D_v*L_evap
        ######################################## External Convection at Condenser ########################################################
        h_c=1200
        T_coolant=285

        def f(T,a_0,a_1,a_2,a_3,a_4,a_5):
            poly=np.exp(a_0+a_1*T+a_2*T**2+a_3*T**3+a_4*T**4+a_5*T**5)
            return poly

        D_v_array = []
        h_fg_array = []
        T_hp_array = []
        R_g_array = []
        P_v_array = []
        rho_v_array = []
        L_flux_array = []
        L_adiabatic_array = []
        mu_v_array = []

        r_h_array = []
        R_v_array = []

        i = 1

        for Q_hp in range(10,50,1):
            
            T_cond=Q_hp/(A_cond*h_c)+T_coolant
            T_hp=T_cond
            T_hpfp=T_hp-273.15

            ######################################## Fluid Properties for Water. From Faghri's book, Table C.70, page 909 ########################################################
            P_v = f(T_hpfp,-5.0945,7.2280e-2,-2.8625e-4,9.2341e-7,-2.0295e-9,2.1645e-12)*1e5
            h_fg = f(T_hpfp,7.8201,-5.8906e-4,-9.1355e-6,8.4738e-8,-3.9635e-10,5.9150e-13)*1e3
            rho_l = f(T_hpfp,6.9094,-2.0146e-5,-5.9868e-6,2.5921e-8,-9.3244e-11,1.2103e-13)     
            rho_v = f(T_hpfp,-5.3225,6.8366e-2,-2.7243e-4,8.4522e-7,-1.6558e-9,1.5514e-12)
            mu_l = f(T_hpfp,-6.3530,-3.1540e-2,2.1670e-4,-1.1559e-6,3.7470e-9,-5.2189e-12)
            mu_v = f(T_hpfp,-11.596,2.6382e-3,6.9205e-6,-6.1035e-8,1.6844e-10,-1.5910e-13)
            k_l = f(T_hpfp,-5.8220e-1,4.1177e-3,-2.7932e-5,6.5617e-8,4.1100e-11,-3.8220e-13)
            k_v = f(T_hpfp,-4.0722,3.2364e-3,6.3860e-6,8.5114e-9,-1.0464e-10,1.6481e-13)
            sigma_l = f(T_hpfp,4.3438,-3.0664e-3,2.0743e-5,-2.5499e-7,1.0377e-9,-1.7156e-12)/1e3
            cp_l = f(T_hpfp,1.4350,-3.2231e-4,6.1633e-6,-4.4099e-8,2.0968e-10,-3.040e-13)*1e3
            cp_v = f(T_hpfp,6.3198e-1,6.7903e-4,-2.5923e-6,4.4936e-8,2.2606e-10,-9.0694e-13)*1e3
            v_fg = 1/rho_v-1/rho_l
            R_g=P_v/(T_hp*rho_v)
            cv_v=cp_v-R_g
            gamma=cp_v/cv_v

            # print("T= ",T_hpfp)
            # print(P_v,h_fg,rho_l,rho_v,mu_l*1e7,mu_v*1e7,k_l,k_v,sigma_l*1e3,cp_l/1e3,cp_v/1e3)

            ######################################## Axial Thermal Resistances ########################################################
            k_wk=(1-epsilon)*k_w+epsilon*k_l
            R_aw=L_adiabatic/(A_w*k_w)
            R_awk=L_adiabatic/(A_wk*k_wk)

            ######################################## Condenser Section Thermal Resistances ########################################################
            alpha=1           # Look into this, need better way to determine this rather than referencing papers.
            h_interc=2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg))
            R_wc=np.log((D_od/2)/(r_i))/(2*np.pi*k_w*L_cond)
            R_wkc=np.log((r_i)/(D_v/2))/(2*np.pi*k_wk*L_cond)
            R_interc=1/(h_interc*A_interc)
            ######################################## Evaporator Section Thermal Resistances ########################################################
            h_intere=2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg))
            R_we=np.log((D_od/2)/(r_i))/(2*np.pi*k_w*L_evap)
            R_wke=np.log((r_i)/(D_v/2))/(2*np.pi*k_wk*L_evap)
            R_intere=1/(h_intere*A_intere)
            ######################################## Vapor Region Thermal Resistance ########################################################
            r_h=D_v/2
            R_v=8*R_g*mu_v*T_hp**2/(np.pi*h_fg**2*P_v*rho_v)*(L_eff/(r_h**4))

            D_v_array.append(D_v)
            h_fg_array.append(h_fg)
            T_hp_array.append(T_hp)
            R_g_array.append(R_g)
            P_v_array.append(P_v)
            rho_v_array.append(rho_v)
            L_flux_array.append(L_flux)
            L_adiabatic_array.append(L_adiabatic)
            mu_v_array.append(mu_v)

            r_h_array.append(r_h)
            R_v_array.append(R_v)

        self.prob.set_val('XS:D_v', D_v_array)
        self.prob.set_val('h_fg', h_fg_array)
        self.prob.set_val('T_hp', T_hp_array)
        self.prob.set_val('R_g', R_g_array)
        self.prob.set_val('P_v', P_v_array)
        self.prob.set_val('rho_v', rho_v_array)
        self.prob.set_val('LW:L_flux', L_flux_array)
        self.prob.set_val('LW:L_adiabatic', L_adiabatic_array)
        self.prob.set_val('mu_v', mu_v_array)
        self.prob.run_model()

        assert_near_equal(self.prob.get_val('r_h'), r_h_array, tolerance=1.0E-5)
        assert_near_equal(self.prob.get_val('R_v'), R_v_array, tolerance=1.0E-5)

    def test_partials(self):  # derivative check

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)

if __name__ == '__main__':
    unittest.main()

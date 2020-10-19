"""
This code estimates the thermal resistance of a heat pipe, given various dimensions
pertaining to the size, wick, wall thicnkess, wick properties, and working fluid. These 
equations assume a constant heat flux applied at the evaporator, and convective heat 
loss at the external surface of the condenser.

Author: Ezra McNichols
        NASA Glenn Research Center
        Turbomachinery and Turboelectric Systems
"""

import numpy as np
import matplotlib.pyplot as plt
from thermo.chemical import Chemical

plt.rc('font', family='serif')
######################################## Solid and Wick Properties ########################################################
k_w=11.4
epsilon=0.46
roomtemp_fluid=Chemical('water')
roomtemp_fluid.calculate(293.15)
k_rtl=roomtemp_fluid.kl
k_wk=(1-epsilon)*k_w+epsilon*k_rtl
######################################## Overall Geometry ########################################################
L_evap=0.01
L_cond=0.02
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
######################################## Axial Thermal Resistances ########################################################
R_aw=L_adiabatic/(A_w*k_w)
R_awk=L_adiabatic/(A_wk*k_wk)
######################################## External Convection at Condenser ########################################################
h_c=1200
T_coolant=293

for Q_hp in range(1,75,1):
    
    T_cond=Q_hp/(A_cond*h_c)+T_coolant
    T_hp=T_cond
    ######################################## Fluid Properties ########################################################
    hp_fluid=Chemical('water')
    hp_fluid.calculate(T_hp)
    P_v=hp_fluid.Psat
    hp_fluid.calculate(T_hp,P_v)
    rho_v=hp_fluid.rhog
    mu_v=hp_fluid.mug
    rho_l=hp_fluid.rhol
    mu_l=hp_fluid.mul
    k_l=hp_fluid.kl
    cp_l=hp_fluid.Cpl
    cp_v=hp_fluid.Cpg
    cv=hp_fluid.Cvg
    Pr_l=cp_l*mu_l/k_l
    h_fg=hp_fluid.Hvap
    sigma_l=hp_fluid.sigma
    v_fg = 1/rho_v-1/rho_l
    R_g=P_v/(T_hp*rho_v)
    cv_v=cp_v-R_g
    gamma=cp_v/cv_v 
    ######################################## Condenser Section Thermal Resistances ########################################################
    alpha=0.1
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
    ######################################## Total Thermal Resistance ########################################################
    R_1=(R_wke+R_intere+R_v+R_interc+R_wkc)*R_awk/(R_wke+R_intere+R_v+R_interc+R_wkc+R_awk)
    R_hp=(R_we+R_wc+R_1)*R_aw/(R_we+R_wc+R_1+R_aw)
    plt.plot(Q_hp,R_hp,marker='o',color='k')
    # plt.plot(T_hp,R_wc,marker='o',color='k')
    # plt.plot(T_hp,R_wkc,marker='d',color='b')
    # plt.plot(T_hp,R_interc,marker='v',color='r')
    # plt.plot(T_hp,R_we,marker='s',color='g')
    # plt.plot(T_hp,R_wke,marker='+',color='purple')
    # plt.plot(T_hp,R_intere,marker='*',color='orange')
    # plt.plot(T_hp,R_v,marker='^',color='cyan')
    
    plt.ylabel('$R_{th}$ [K/W]')
    plt.xlabel('Heat Load [W]')

plt.show()
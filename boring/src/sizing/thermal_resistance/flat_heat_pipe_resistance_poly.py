"""
This code estimates the thermal resistance of a heat pipe, given various dimensions
pertaining to the size, wick, wall thicnkess, wick properties, and working fluid. These 
equations assume a constant heat flux applied at the evaporator, and convective heat 
loss at the external surface of the condenser.

Author: Ezra McNichols
        NASA Glenn Research Center
        Turbomachinery and Turboelectric Systems
"""


# Make the change to set t_wk, calculate D_v based on that.
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
######################################## Solid and Wick Properties ########################################################
k_w=11.4
epsilon=0.46
######################################## Overall Geometry ########################################################
L_evap=0.01
L_cond=0.02
L_adiabatic=0.03
L_eff=(L_evap+L_cond)/2+L_adiabatic
t_w=0.0005
t_wk=0.00069
D_od=0.006

W=0.02  # Width of heat pipe into the page
H=0.02  # Total thickness of heat pipe

t_v=H-2*t_w-2*t_wk
A_cond=W*L_cond
A_evap=W*L_evap
r_h=(t_v*W)/(2*W+2*t_v)
######################################## Heat Pipe Core Geometry ########################################################
A_w=t_w*W
A_wk=t_wk*W
A_interc=A_cond
A_intere=A_evap
######################################## External Convection at Condenser ########################################################
h_c=1200
T_coolant=285

def f(T,a_0,a_1,a_2,a_3,a_4,a_5):
    poly=np.exp(a_0+a_1*T+a_2*T**2+a_3*T**3+a_4*T**4+a_5*T**5)
    return poly

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
    #cv_v=cp_v-R_g
    #gamma=cp_v/cv_v

    # print("T= ",T_hpfp)
    # print(P_v,h_fg,rho_l,rho_v,mu_l*1e7,mu_v*1e7,k_l,k_v,sigma_l*1e3,cp_l/1e3,cp_v/1e3)

    ######################################## Axial Thermal Resistances ########################################################
    k_wk=(1-epsilon)*k_w+epsilon*k_l
    R_aw=L_eff/(A_w*k_w)
    R_awk=L_eff/(A_wk*k_wk)
    ######################################## Condenser Section Thermal Resistances ########################################################
    alpha=1           # Look into this, need better way to determine this rather than referencing papers.
    h_interc=2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg))
    R_wc=t_w/(k_w*A_cond)
    R_wkc=t_wk/(k_wk*A_cond)
    R_interc=1/(h_interc*A_interc)
    ######################################## Evaporator Section Thermal Resistances ########################################################
    h_intere=2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg))
    R_we=t_w/(k_w*A_evap)
    R_wke=t_wk/(k_wk*A_evap)
    R_intere=1/(h_intere*A_intere)
    ######################################## Vapor Region Thermal Resistance ########################################################
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
    plt.legend(['Analytical'])

plt.show()
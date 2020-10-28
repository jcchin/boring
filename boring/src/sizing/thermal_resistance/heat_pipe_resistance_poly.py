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

use_poly = True

if not use_poly: #use thermo package
    from thermo.chemical import Chemical

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
D_v=0.00362 # make this a calculation
r_i=(D_od/2-t_w)
A_cond=np.pi*D_od*L_cond   # Karsten
A_evap=np.pi*D_od*L_evap   # Karsten
######################################## Heat Pipe Core Geometry ########################################################
A_w=np.pi*((D_od/2)**2-(D_od/2-t_w)**2) # Dustin
A_wk=np.pi*((D_od/2-t_w)**2-(D_v/2)**2) # Dustin
A_interc=np.pi*D_v*L_cond # Dustin
A_intere=np.pi*D_v*L_evap # Dustin
######################################## External Convection at Condenser ########################################################
h_c=1200
T_coolant=285

R_comsol=[1.54,1.52,1.494,1.476,1.46]
#R_comsol=[1.65,1.666,1.677,1.695,1.715]
Q_comsol=[10,20,30,40,50]
plt.plot(Q_comsol,R_comsol,marker='*',color='blue')

def f(T,a_0,a_1,a_2,a_3,a_4,a_5):
    poly=np.exp(a_0+a_1*T+a_2*T**2+a_3*T**3+a_4*T**4+a_5*T**5)
    return poly

alpha_array = np.arange(10,50,1)
h_fg_array = np.arange(10,50,1)
T_hp_array = np.arange(10,50,1)
v_fg_array = np.arange(10,50,1)
R_g_array = np.arange(10,50,1)
P_v_array = np.arange(10,50,1)
D_od_array = np.arange(10,50,1)
r_i_array = np.arange(10,50,1)
k_w_array = np.arange(10,50,1)
L_cond_array = np.arange(10,50,1)
D_v_array = np.arange(10,50,1)
k_wk_array = np.arange(10,50,1)
A_interc_array = np.arange(10,50,1)

h_interc_array = np.arange(10,50,1)
R_wc_array = np.arange(10,50,1)
R_wkc_array = np.arange(10,50,1)
R_interc_array = np.arange(10,50,1)

i = 0

for Q_hp in range(10,50,1):
    
    T_cond=Q_hp/(A_cond*h_c)+T_coolant
    T_hp=T_cond
    T_hpfp=T_hp-273.15

    ######################################## Fluid Properties for Water. From Faghri's book, Table C.70, page 909 ########################################################
    if use_poly:
        P_v = f(T_hpfp,-5.0945,7.2280e-2,-2.8625e-4,9.2341e-7,-2.0295e-9,2.1645e-12)*1e5  # Ezra
        h_fg = f(T_hpfp,7.8201,-5.8906e-4,-9.1355e-6,8.4738e-8,-3.9635e-10,5.9150e-13)*1e3 # Ezra
        rho_l = f(T_hpfp,6.9094,-2.0146e-5,-5.9868e-6,2.5921e-8,-9.3244e-11,1.2103e-13)      # Ezra
        rho_v = f(T_hpfp,-5.3225,6.8366e-2,-2.7243e-4,8.4522e-7,-1.6558e-9,1.5514e-12) # Ezra
        mu_l = f(T_hpfp,-6.3530,-3.1540e-2,2.1670e-4,-1.1559e-6,3.7470e-9,-5.2189e-12) # Ezra
        mu_v = f(T_hpfp,-11.596,2.6382e-3,6.9205e-6,-6.1035e-8,1.6844e-10,-1.5910e-13) # Ezra
        k_l = f(T_hpfp,-5.8220e-1,4.1177e-3,-2.7932e-5,6.5617e-8,4.1100e-11,-3.8220e-13) # Ezra
        k_v = f(T_hpfp,-4.0722,3.2364e-3,6.3860e-6,8.5114e-9,-1.0464e-10,1.6481e-13) # Ezra
        sigma_l = f(T_hpfp,4.3438,-3.0664e-3,2.0743e-5,-2.5499e-7,1.0377e-9,-1.7156e-12)/1e3 # Ezra
        cp_l = f(T_hpfp,1.4350,-3.2231e-4,6.1633e-6,-4.4099e-8,2.0968e-10,-3.040e-13)*1e3 # Ezra
        cp_v = f(T_hpfp,6.3198e-1,6.7903e-4,-2.5923e-6,4.4936e-8,2.2606e-10,-9.0694e-13)*1e3 # Ezra
    else:
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
    # print("T= ",T_hpfp)
    # print(P_v,h_fg,rho_l,rho_v,mu_l*1e7,mu_v*1e7,k_l,k_v,sigma_l*1e3,cp_l/1e3,cp_v/1e3)

    ######################################## Axial Thermal Resistances ########################################################
    k_wk=(1-epsilon)*k_w+epsilon*k_l  # Dustin
    R_aw=L_eff/(A_w*k_w) # Dustin
    R_awk=L_eff/(A_wk*k_wk) # Dustin
    ######################################## Condenser Section Thermal Resistances ########################################################
    alpha=1           # Look into this, need better way to determine this rather than referencing papers.
    h_interc=2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg)) # Sydney
    R_wc=np.log((D_od/2)/(r_i))/(2*np.pi*k_w*L_cond) # Sydney
    R_wkc=np.log((r_i)/(D_v/2))/(2*np.pi*k_wk*L_cond) # Sydney
    R_interc=1/(h_interc*A_interc) # Sydney

    print(R_wc)
    ######################################## Evaporator Section Thermal Resistances ########################################################
    h_intere=2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg)) # Karsten
    R_we=np.log((D_od/2)/(r_i))/(2*np.pi*k_w*L_evap) # Karsten
    R_wke=np.log((r_i)/(D_v/2))/(2*np.pi*k_wk*L_evap) # Karsten
    R_intere=1/(h_intere*A_intere) # Karsten
    ######################################## Vapor Region Thermal Resistance ########################################################
    r_h=D_v/2 # Karsten
    R_v=8*R_g*mu_v*T_hp**2/(np.pi*h_fg**2*P_v*rho_v)*(L_eff/(r_h**4)) # Karsten
    ######################################## Total Thermal Resistance ########################################################
    R_1=(R_wke+R_intere+R_v+R_interc+R_wkc)*R_awk/(R_wke+R_intere+R_v+R_interc+R_wkc+R_awk) # Jeff
    R_hp=(R_we+R_wc+R_1)*R_aw/(R_we+R_wc+R_1+R_aw)  # Jeff

    alpha_array[i]= alpha
    h_fg_array[i]= h_fg
    T_hp_array[i]= T_hp
    v_fg_array[i]= v_fg
    R_g_array[i]= R_g
    P_v_array[i]= P_v
    D_od_array[i]= D_od
    r_i_array[i]= r_i
    k_w_array[i]= k_w
    L_cond_array[i]= L_cond
    D_v_array[i]= D_v
    k_wk_array[i]= k_wk
    A_interc_array[i]= A_interc

    h_interc_array[i]= h_interc
    R_wc_array[i]= R_wc
    R_wkc_array[i]= R_wkc
    R_interc_array[i]= R_interc

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

    i = i + 1

print('alpha = ', alpha_array)
print('h_fg = ', h_fg_array)
print('T_hp = ', T_hp_array)
print('v_fg = ', v_fg_array)
print('R_g = ', R_g_array)
print('P_v = ', P_v_array)
print('D_od = ', D_od_array)
print('r_i = ', r_i_array)
print('k_w = ', k_w_array)
print('L_cond = ', L_cond_array)
print('D_v = ', D_v_array)
print('k_wk = ', k_wk_array)
print('A_interc = ', A_interc_array)

print('h_interc = ', h_interc_array)
print('R_wc = ', R_wc_array)
print('R_wkc = ', R_wkc_array)
print('R_interc = ', R_interc_array)

plt.show()
print("Rwe", R_we)
print("Rwke", R_wke)
print("Rv", R_v)
print("Rintere", R_intere)
print("Rinterc", R_interc)
print("Rwkc", R_wkc)
print("R_awk", R_awk)
print("Raw", R_aw)
print("Rwc", R_wc)

print("R_hp", R_hp)

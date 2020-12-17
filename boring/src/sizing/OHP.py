"""
Oscillating Heat Pipe heat flux calculalation 

Author: Jeff Chin

# References
# http://web.missouri.edu/~zhangyu/Pubs/76_081501_1.pdf
# http://egr.uri.edu/wp-uploads/mcise/ms_nonthesis_example2.pdf

# Definitions
#--------------------
# Bo # Bond Number (Ratio of buoyancy force to surface tension force)
# Fr # Froude Number (Ratio of dynamic viscosity to weight)
# Ja # Jacob Number (Ratio of sensible heat to latent heat in the working fluid)
# Pr # Pradntl Number (Ratio of momentum diffusivity to thermal diffusivity)
# rho_v # vapor density
# rho_l # liquid density
# R_cv # Check Valve Ratio
# We # Weber Number (Ratio of dynamic force to surface tension)
# Le_di # Aspect Ratio
# h_fg # Latent Heat of Vaporization
# Ku_90 # Kutateladze Number (Ratio of heat flux to critical heat flux)
# n # number of capillary tubes
# -------------------
#      Typical values
#    0.5  < D < 2.92       (mm)
# 25,000  < q < 700,000    (W/m^2)
#  1,000  < h < 60,000      (W/m^2*C)
# 0.001/A < R < 1E-5/A (C/W)
"""
import numpy as np

rho_l = 763.3  # kg/m^3 @ 50C
rho_v = 3.2  # kg/m^3 @ 100C
d = 0.00165  # m
sigma = 0.02155  # N/m @ 30C (working fluid surface tension)
g = 9.81  # gravity m/s^2
h_fg = 846000  # J/kg
D_max = 2 * (sigma / (g * (rho_l - rho_v))) ** 0.5
Beta = 0.  # radians
Cp_l = 2570  # J/kg*K
mu = 694.  # N*s/m^2 @ 50C
k = 0.171  # W/m*K
delT = 50  # K
delP = 2000  # N/m^2
Le = .038  # m
La = .102  # m
Lc = .06  # m
n = 16

# delP = sigma*(2/((d/2)-delt) - 1/((d/2)-delt))


# Closed End Oscillating Heat Pipe
# Experimental
# Ku_0 = 0.0052*((D**4.3*Lt**0.1*Le**-4.4)*n**0.5*(rho_v/rho_l)**-0.2*Pr**-25)**0.116


# p1 = D/L
# p2 = Cp*delT/h  # 1/Ja
# p3 = D*(g*((rho_l-rho_v)/sigma)**0.5) #Bo^0.5

# # Analytical correlation
# Ku_0 = 53860*p1**1.127*p2**1.417*p3**-1.32 #18% std dev error
# Ku_90 = 0.002*p1**0.92*p2**-0.212*p3**-0.59*(1+(rho_v/rho_l)**0.25)**13.06 #29% std dev error
#                                             #flooding term (Wallis Number)

# Closed Loop Oscillating Heat Pipe
Bo = d ** 2 * g * (rho_l - rho_v) / sigma
Ja = h_fg / (Cp_l * delT)
Pr = Cp_l * mu / k
Ka = rho_l * delP * d ** 2 / (mu ** 2 * 0.5 * (Le + Lc) * La)  # evap, condenser, adiabatic

# heat flux (W/m^2)
q_dot = 0.54 * (np.exp(Beta)) ** 0.48 * Ka ** 0.47 * Pr ** 0.27 * Ja ** 1.43 * n ** -0.27

# Ku_90 = 0.0004*(Bo**2.2*Fr**1.42*Ja**1.2*Pr**1.02*(rho_v/rho_l)**0.98*R_cv**1.4*We**0.8*Le_di**0.5)**0.107
# 30% std dev error

print(q_dot)

# Startup Transient
# Greater surface roughness means easier to start oscillation
# but requires more driving force to maintain oscillation
# Tn #nucleation point
# Tv #vapor point

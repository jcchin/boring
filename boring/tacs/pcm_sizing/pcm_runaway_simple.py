import numpy as np
import matplotlib.pyplot as plt


# Constants and assumptions
# -------------------------

# Geometry
w = 54.0e-3  # total width
l = 61.0e-3  # total length
bat_t = 5.60e-3   # thickness of battery
pcm_t = 7.1e-3    # thickness of pcm
dw = 10.0e-3  # width of heat-pipe centered on pouch-cell

# Material selection
foam_material = 'cu'
pcm_material = 'croda'

# Material properties
if foam_material == 'al':
    porosity = 0.90
    rho_foam = 8960.0
    c_foam = 0.91e3
    k_foam = 237.0

else:  # 'cu'
    porosity = 0.94
    rho_foam = 8960.0
    c_foam = 0.39e3
    k_foam = 401.0

if pcm_material == 'croda':
    lh = 197e3  # latent heat, melting, J/kg
    mt = 44.0  # melting temperature, deg. C

    # solid
    rho_pcm_s = 940.0
    c_pcm_s = 1.9e3
    k_pcm_s = 0.25

    # liquid
    rho_pcm_l = 829.0
    c_pcm_l = 2.0e3
    k_pcm_l = 0.16
else:  # 'puretemp'
    lh = 0.0  # latent heat, melting, J/kg
    mt = 0.0  # melting temperature, deg. C

    # solid
    rho_pcm_s = 0.0
    c_pcm_s = 0.0
    k_pcm_s = 0.0

    # liquid
    rho_pcm_l = 0.0
    c_pcm_l = 0.0
    k_pcm_l = 0.0

# solid pcm bulk properties
bulk_rho_s = 1. / (porosity / rho_pcm_s + (1 - porosity) / rho_foam)
bulk_k_s = 1. / (porosity / k_pcm_s + (1 - porosity) / k_foam)
bulk_c_s = 1. / (porosity / c_pcm_s + (1 - porosity) / c_foam)

# liquid pcm bulk properties
bulk_rho_l = 1. / (porosity / rho_pcm_l + (1 - porosity) / rho_foam)
bulk_k_l = 1. / (porosity / k_pcm_l + (1 - porosity) / k_foam)
bulk_c_l = 1. / (porosity / c_pcm_l + (1 - porosity) / c_foam)

# battery properties
rho_battery = (44e-3)/((61e-3)*(54e-3)*(5.6e-3))  # spec 44g weight/ volume
k_battery = 200.0  # ???? https://tfaws.nasa.gov/wp-content/uploads/TFAWS18-PT-11.pdf
c_battery = 1400.0  # ^ same

# Thermal runaway heat generation
duration = 10.0
total_energy = 16.5*3600.0  # 16.5 W-h -> J
thermal_pct = 0.75  # percent of total battery energy that can convert to heat during runaway
Qdot = thermal_pct*total_energy/duration  # total heat released per second
qdot = Qdot/(bat_t*l*w)  # heat released per second per unit of battery volume

# Heat-pipe cooling to get steady-state initial condition
discharge_rate = 10.4  # 2C
resistance = 15e-3
Q = resistance*discharge_rate**2
T0 = 25.0  # Ambient temperature (deg C)
Tref = 0.0  # reference temperature of heat pipe (above ambient)
h = 300.0  # convective heat transfer coefficient to from PCM to heat pipe
A = (10e-3)*(61e-3)  # area of heat pipe face, m^2
dT = Q/(h*A) + Tref + T0 # fixed temperature boundary condition at heat pipe interface

# Discretization in time
tfinal = 40.0
nsteps = 81
dt = tfinal/(nsteps-1)
t = np.linspace(0.0, tfinal, nsteps)

# Discretization in space
nelems_battery = 10
nelems_pcm = 10
nelems = nelems_battery + nelems_pcm
x_battery = np.linspace(0.0, bat_t, nelems_battery)
x_pcm = np.linspace(bat_t, bat_t+pcm_t, nelems_pcm)
dx_battery = bat_t/(nelems_battery-1)
dx_pcm = pcm_t/(nelems_pcm-1)
x = np.concatenate((x_battery, x_pcm))
T = dT*np.ones((nsteps, nelems))
Tmax = np.zeros(nsteps)

# Define some integration constants
a_battery = k_battery*dt/(c_battery*rho_battery*dx_battery**2)
b_battery = dt*dx_battery/(c_battery*rho_battery*bat_t)
a_pcm_s = bulk_k_s*dt/(bulk_c_s*bulk_rho_s*dx_pcm**2)
a_pcm_l = bulk_k_l*dt/(bulk_c_l*bulk_rho_l*dx_pcm**2)
a_conv_s = h*A/(bulk_rho_s*dx_pcm*l*w)
a_conv_l = h*A/(bulk_rho_l*dx_pcm*l*w)

for i in range(nsteps-1):
    for j in range(nelems):

        # Define the stepwise heat generation
        if t[i] <= duration:
            q = b_battery*qdot
        else:
            q = 0.0

        if j == 0:
            T[i+1, j] = T[i, j] + a_battery*(T[i, 1] - T[i, 0]) + q

        elif j < nelems_battery: # battery
            T[i+1, j] = T[i, j] + a_battery*(T[i, j-1] - 2.0*T[i, j] + T[i, j+1]) + q

        elif j < nelems-1: # pcm
            T[i+1, j] = T[i, j] + a_pcm_s*(T[i, j-1] - 2.0*T[i, j] + T[i, j+1])

        else: # x=L
            T[i+1, j] = T[i, j] + a_pcm_s*(T[i, j-1] - T[i, j]) - a_conv_s*(T[i, j] - Tref)
    Tmax[i] = np.amax(T[i, :])

print(np.amax(Tmax))
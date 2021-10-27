import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# Constants and assumptions
# -------------------------

# Geometry
w = 54.0e-3  # total width
l = 61.0e-3  # total length
A = l*w
t_battery = 5.60e-3   # thickness of battery
t_pcm = 7.1e-3    # thickness of pcm
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
    mt = 44.01  # melting temperature, deg. C

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
thermal_pct = 1.0  # percent of total battery energy that can convert to heat during runaway
Qdot = thermal_pct*total_energy/duration  # total heat released per second

# Heat-pipe cooling to get steady-state initial condition
T0 = 44.0  # Starting uniform temperature (deg C)
Tref = 0.0  # reference temperature of heat pipe (above ambient)
h = 0.0#100.0  # convective heat transfer coefficient to from PCM to heat pipe
Aconv = dw*l  # area of heat pipe face, m^2

# Discretization in time
tfinal = 40.0
nsteps = int(500*tfinal+1)
dt = tfinal/(nsteps-1)
t = np.linspace(0.0, tfinal, nsteps)

# Discretization in space
nelems_battery = 10
nelems_pcm = 10
nelems = nelems_battery + nelems_pcm
x_battery = np.linspace(0.0, t_battery, nelems_battery+1)
x_pcm = np.linspace(t_battery, t_battery+t_pcm, nelems_pcm+1)
dx_battery = t_battery/nelems_battery
dx_pcm = t_pcm/nelems_pcm
x_battery = dx_battery + x_battery[0:-1]
x_pcm = dx_pcm + x_pcm[0:-1]
x = np.concatenate((x_battery, x_pcm))
T = T0*np.ones((nsteps, nelems))
Tmax = np.zeros(nsteps)
Tmax[0] = T[0, 0]
pcm_lh = (lh*A*t_pcm*rho_pcm_s/nelems_pcm)*np.ones(nelems_pcm)
lh_hist = np.zeros((nsteps, nelems_pcm))
lh_hist[0, :] = np.ones(nelems_pcm)
lh_pct_hist = np.zeros(nsteps)
lh_pct_hist[0] = 1.0
dTdt = np.zeros(nsteps)

# Define some integration constants
a_battery = dt/(c_battery*rho_battery*dx_battery*A)
R_battery = dx_battery/(A*k_battery)
a_pcm_s = dt/(bulk_c_s*bulk_rho_s*dx_pcm*A)
a_pcm_l = dt/(bulk_c_l*bulk_rho_l*dx_pcm*A)
R_pcm_s = dx_pcm/(A*bulk_k_s)
R_pcm_l = dx_pcm/(A*bulk_k_l)

for i in range(nsteps-1):

    # Define the stepwise heat generation
    if t[i] <= duration:
        q = Qdot*(dx_battery/t_battery)
    else:
        q = 0.0

    for j in range(nelems):
        dTdt_j = np.zeros(nelems)

        # bottom boundary
        if j == 0:
            #print(a_battery*(T[i, 1] - T[i, 0])/R_battery) # should be negative
            T[i+1, j] = T[i, j] + a_battery*((T[i, 1] - T[i, 0])/R_battery + q)

        # battery
        elif j < nelems_battery:
            T[i+1, j] = T[i, j] + a_battery*((T[i, j-1] - 2.0*T[i, j] + T[i, j+1])/R_battery + q)

        # pcm
        elif j < nelems-1:

            # Check if the element is melted yet:
            if pcm_lh[j-nelems_battery] > 0.0:  # not melted yet

                # Check if the element will get melted this iteration:
                Tkp1 = T[i, j] + a_pcm_s*(T[i, j-1] - 2.0*T[i, j] + T[i, j+1])/R_pcm_s
                if Tkp1 >= mt:
                    T[i+1, j] = mt  # element temp = melting temp
                    Qres = bulk_c_s*bulk_rho_s*dx_pcm*(Tkp1 - mt) # subtract the energy it took to get to melting temperature before reducing the latent heat
                    pcm_lh[j-nelems_battery] -= Qres
                    # if pcm_lh goes negative, start heating up the liquid
                    if pcm_lh[j-nelems_battery] < 0.0:
                        T[i+1, j] += a_pcm_l*(-pcm_lh[j-nelems_battery])
                        pcm_lh[j-nelems_battery] = 0.0
                else:  # still solid
                    T[i+1, j] = Tkp1

            else:  # melted
                T[i+1, j] = T[i, j] + a_pcm_l*(T[i, j-1] - 2.0*T[i, j] + T[i, j+1])/R_pcm_l

        # heat-pipe boundary
        else:

            # Check if the element is melted yet:
            if pcm_lh[j-nelems_battery] > 0.0:  # not melted yet

                # Check if the element will get melted this iteration:
                Tkp1 = T[i, j] + a_pcm_s*((T[i, j-1] - T[i, j])/R_pcm_s - h*Aconv*(T[i, j] - Tref))
                if Tkp1 >= mt:
                    T[i+1, j] = mt  # element temp = melting temp
                    Qres = bulk_c_s*bulk_rho_s*dx_pcm*(Tkp1 - mt) # subtract the energy it took to get to melting temperature before reducing the latent heat
                    pcm_lh[j-nelems_battery] -= Qres
                    # if pcm_lh goes negative, start heating up the liquid
                    if pcm_lh[j-nelems_battery] < 0.0:
                        T[i+1, j] += a_pcm_l*(-pcm_lh[j-nelems_battery])
                        pcm_lh[j-nelems_battery] = 0.0
                else:  # still solid
                    T[i+1, j] = Tkp1

            else:  # melted
                T[i+1, j] = T[i, j] + a_pcm_l*((T[i, j-1] - T[i, j])/R_pcm_l - h*Aconv*(T[i, j] - Tref))
        dTdt_j[j] = (T[i+1, j]-T[i, j])/dt

    dTdt[i+1] = np.amax(dTdt_j)
    Tmax[i+1] = np.amax(T[i+1, :])
    lh_pct_hist[i+1] = np.sum(pcm_lh)/(lh*A*t_pcm*rho_pcm_s)
    lh_hist[i+1, :] = pcm_lh/(lh*A*t_pcm*rho_pcm_s/nelems_pcm)

# Plot the results
nsample = int((nsteps-1)/tfinal)
T = T[0::nsample, :]
plot_len = np.shape(T)[0]

plt.style.use('mark')

fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))

# Plot the temperature distribution over time
norm = plt.Normalize(vmin=0, vmax=plot_len)
cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.coolwarm)
cmap.set_array([])
colors = plt.cm.coolwarm(np.linspace(0.0, 1.0, plot_len))

for i in range(plot_len):
    ax.plot(T[i, :], x*1e3, color=colors[i])

ax.set_xlabel(r"$T(t)$ ($^\circ$C)")
#ax.set_ylabel(r"$x$ (mm)")
ax.axes.yaxis.set_visible(False)
# Add the text
ax.annotate('Battery', (40.0, 0.5*t_battery*1e3), (40.0, 0.5*t_battery*1e3),
            ha='center', rotation=90, annotation_clip=False)
ax.annotate('PCM', (40.0, t_battery*1e3+0.5*t_pcm*1e3), (40.0, t_battery*1e3+0.5*t_pcm*1e3),
            ha='center', rotation=90, annotation_clip=False)
# Draw the arrow
ax.annotate('', (40.5, 0.0), xytext=(40.5, t_battery*1e3),
            arrowprops=dict(arrowstyle="<->"),
            annotation_clip=False)
ax.annotate('', (40.5, t_battery*1e3), xytext=(40.5, t_battery*1e3+t_pcm*1e3),
            arrowprops=dict(arrowstyle="<->"),
            annotation_clip=False)

plt.savefig('temperature_history.pdf', bbox_inches='tight', transparent=True)

# Plot the latent heat (liquid/solid) over time
fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
ax.plot(t, 100*lh_pct_hist)

ax.set_xlabel(r"$t$ (s)")
ax.set_ylabel(r"Percent of PCM that is solid")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.savefig('latent_heat_history.pdf', bbox_inches='tight', transparent=True)

# Plot the max temperature rate of change over time
fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
ax.plot(t[1::], dTdt[1::])

ax.set_xlabel(r"$t$ (s)")
ax.set_ylabel(r"dT/dt ($^\circ$C/s)")

plt.savefig('dTdt_history.pdf', bbox_inches='tight', transparent=True)
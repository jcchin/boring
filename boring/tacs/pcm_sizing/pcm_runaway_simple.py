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
t_pcm = 4.5e-3    # thickness of pcm
dw = 10.0e-3  # width of heat-pipe centered on pouch-cell

# Material selection
foam_material = 'cu'
pcm_material = 'PureTemp151'

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
    rho_pcm_l = 830.0
    c_pcm_l = 2.0e3
    k_pcm_l = 0.16

elif pcm_material == 'PureTemp108':
    lh = 180e3  # latent heat, melting, J/kg
    mt = 108.0  # melting temperature, deg. C

    # solid
    rho_pcm_s = 870.0
    c_pcm_s = 2060.0  # not on data sheet - using same value as PureTemp151
    k_pcm_s = 0.25

    # liquid
    rho_pcm_l = 800.0
    c_pcm_l = 2170.0  # not on data sheet - using same value as PureTemp151
    k_pcm_l = 0.15

elif pcm_material == 'PureTemp151':
    lh = 217e3  # latent heat, melting, J/kg
    mt = 151.0  # melting temperature, deg. C

    # solid
    rho_pcm_s = 1490.0
    c_pcm_s = 2060.0
    k_pcm_s = 0.25

    # liquid
    rho_pcm_l = 1360.0
    c_pcm_l = 2170.0
    k_pcm_l = 0.15

# solid pcm bulk properties
bulk_rho_s = 1. / (porosity / rho_pcm_s + (1 - porosity) / rho_foam)
bulk_k_s = 1. / (porosity / k_pcm_s + (1 - porosity) / k_foam)
bulk_c_s = 1. / (porosity / c_pcm_s + (1 - porosity) / c_foam)

# liquid pcm bulk properties
bulk_rho_l = 1. / (porosity / rho_pcm_l + (1 - porosity) / rho_foam)
bulk_k_l = 1. / (porosity / k_pcm_l + (1 - porosity) / k_foam)
bulk_c_l = 1. / (porosity / c_pcm_l + (1 - porosity) / c_foam)

# battery properties
rho_battery = 2385.0 #(44e-3)/((61e-3)*(54e-3)*(5.6e-3))  # spec 44g weight/ volume
k_battery = 0.80  # ???? https://tfaws.nasa.gov/wp-content/uploads/TFAWS18-PT-11.pdf
c_battery = 800.0  # ^ same

# Thermal runaway heat generation
duration = 10.0
total_energy = 16.5*3600.0  # 16.5 W-h -> J
thermal_pct = 0.50  # percent of total battery energy that can convert to heat during runaway
Qdot = thermal_pct*total_energy/duration  # total heat released per second

# Heat-pipe cooling to get steady-state initial condition
T0 = 44.0  # Starting uniform temperature (deg C)
Tref = 0.0  # reference temperature of heat pipe (above ambient)
h = 300.0  # convective heat transfer coefficient to from PCM to heat pipe
Aconv = dw*l  # area of heat pipe face, m^2

# Discretization in time
tfinal = 180.0
nsteps = int(5000*tfinal+1)
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
dTdt = np.zeros(nsteps)

# Mass of each component
m_battery = rho_battery*t_battery*A
m_pcm = bulk_rho_s*t_pcm*A

# Allocate the latent heat to the pcm elements
pcm_lh = (lh*m_pcm/nelems_pcm)*np.ones(nelems_pcm)
lh_pct_hist = np.zeros(nsteps)
lh_pct_hist[0] = 1.0

# Define some integration constants
a_battery = dt/(c_battery*m_battery/nelems_battery)
a_pcm_s = dt/(bulk_c_s*m_pcm/nelems_pcm)
a_pcm_l = dt/(bulk_c_l*m_pcm/nelems_pcm)

# Thermal resistance of each element type
R_battery = dx_battery/(A*k_battery)
R_interface_s = (dx_battery + dx_pcm)/(A*(k_battery + bulk_k_s))
R_interface_l = (dx_battery + dx_pcm)/(A*(k_battery + bulk_k_l))
R_pcm_s = dx_pcm/(A*bulk_k_s)
R_pcm_l = dx_pcm/(A*bulk_k_l)

for i in range(nsteps-1):

    # Define the stepwise heat generation
    if t[i] <= duration:
        q = Qdot*(dx_battery/t_battery)
    else:
        q = 0.0

    dTdt_j = np.zeros(nelems)

    for j in range(nelems):

        # bottom boundary
        if j == 0:
            T[i+1, j] = T[i, j] + a_battery*((T[i, 1] - T[i, 0])/R_battery + q)

        # battery
        elif j < nelems_battery-1:
            T[i+1, j] = T[i, j] + a_battery*((T[i, j-1] - 2.0*T[i, j] + T[i, j+1])/R_battery + q)

        # battery top surface
        elif j == nelems_battery-1:

            # Check if the element is melted yet:
            if pcm_lh[0] > 0.0:  # not melted yet
                T[i+1, j] = T[i, j] + a_battery*((T[i, j-1] - T[i, j])/R_battery - (T[i, j] - T[i, j+1])/R_interface_s + q)
            else:
                T[i+1, j] = T[i, j] + a_battery*((T[i, j-1] - T[i, j])/R_battery - (T[i, j] - T[i, j+1])/R_interface_l + q)

        # pcm bottom surface
        elif j == nelems_battery:

            # Check if the element is melted yet:
            if pcm_lh[0] > 0.0:  # not melted yet

                # Check if the element will get melted this iteration:
                Tkp1 = T[i, j] + a_pcm_s*((T[i, j-1] - T[i, j])/R_interface_s - (T[i, j] - T[i, j+1])/R_pcm_s)
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
                T[i+1, j] = T[i, j] + a_pcm_l*((T[i, j-1] - T[i, j])/R_interface_l - (T[i, j] - T[i, j+1])/R_pcm_l)

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
    lh_pct_hist[i+1] = np.sum(pcm_lh)/(lh*m_pcm)

print(f"max(dT) = {np.amax(Tmax)}")
print(f"Remaining solid material = {100.0*lh_pct_hist[-1]}")
if lh_pct_hist[-1] == 0.0:
    i_melt = np.where(lh_pct_hist==0.0)[0][0]
    print(f"Time when pcm is fully melted = {t[i_melt]}")

# Plot the results
nsample = int((nsteps-1)/tfinal)
T = T[0::nsample, :]
plot_len = np.shape(T)[0]

plt.style.use('https://raw.githubusercontent.com/markleader/mpl_styles/master/mark.mplstyle')

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
ax.annotate('Battery', (22.0, 0.5*t_battery*1e3), (22.0, 0.5*t_battery*1e3),
            ha='center', rotation=90, annotation_clip=False)
ax.annotate('PCM', (22.0, t_battery*1e3+0.5*t_pcm*1e3), (22.0, t_battery*1e3+0.5*t_pcm*1e3),
            ha='center', rotation=90, annotation_clip=False)
# Draw the arrow
ax.annotate('', (30.0, 0.0), xytext=(30.0, t_battery*1e3),
            arrowprops=dict(arrowstyle="<->"),
            annotation_clip=False)
ax.annotate('', (30.0, t_battery*1e3), xytext=(30.0, t_battery*1e3+t_pcm*1e3),
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
ax.plot(t[1::], 60.0*dTdt[1::])

ax.set_xlabel(r"$t$ (s)")
ax.set_ylabel(r"dT/dt ($^\circ$C/min)")
ax.set_title('Rate of change of battery temperature')

plt.savefig('dTdt_history.pdf', bbox_inches='tight', transparent=True)


# Plot the maximum temperature over time
fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
ax.plot(t, Tmax)

ax.set_xlabel(r"$t$ (s)")
ax.set_ylabel(r"max(T) ($^\circ$C)")

plt.savefig('dTmax_history.pdf', bbox_inches='tight', transparent=True)

# Plot the temperature on both boundaries on the PCM over time
fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
ax.plot(t[0::nsample], T[:, -1], label='PCM-heat-pipe boundary')
ax.plot(t[0::nsample], T[:, nelems_battery], color='tab:red', label='PCM-battery boundary')

ax.set_xlabel(r"$t$ (s)")
ax.set_ylabel(r"T ($^\circ$C)")
ax.set_title("Temperature at PCM boundaries")
ax.legend()

plt.savefig('PCM_boundary_temperature.pdf', bbox_inches='tight', transparent=True)

# Make a combined plot with PCM boundary temperature and cell dT/dt

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, constrained_layout=True, figsize=(10, 8))

(plot1,) = ax0.plot(t[0::nsample], T[:, -1], color='tab:blue', label='PCM-heat-pipe boundary')
(plot2,) = ax0.plot(t[0::nsample], T[:, nelems_battery], color='tab:red', label='PCM-battery boundary')

ax1.plot(t[1::], 60.0*dTdt[1::], color='tab:blue')

ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.tick_params(axis='x', direction='out', length=0.0, width=0.0)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax0.grid(True)
ax1.grid(True)

ax1.set_xticks([0.0, 60.0, 120.0, 180.0])
ax1.set_xticklabels([0, 1, 2, 3])

# Label the maximum PCM temperature
yticks = ax0.get_yticks()
yticks = np.append(yticks[0:-1], [np.amax(T[:, nelems_battery])])
yticklabels = []
for yt in yticks:
    yticklabels.append('{0:.0f}'.format(yt))
ax0.set_yticks(yticks)
ax0.set_yticklabels(yticklabels)

for label in ax0.get_xticklabels():
    label.set_visible(False)

# ax0.text(
#     t[-1],
#     T[-1, -1],
#     " — " + plot1.get_label(),
#     size="medium",
#     color=plot1.get_color(),
#     ha="left",
#     va="center",
# )

# ax0.text(
#     t[-1],
#     T[-1, nelems_battery],
#     " — " + plot2.get_label(),
#     size="medium",
#     color=plot2.get_color(),
#     ha="left",
#     va="center",
# )

t_idx = np.where(t >= 120.0)[0][0]
T_idx = int(t_idx/nsample)
ax0.text(
    t[t_idx],
    T[T_idx, -1]+10.0,
    " " + plot1.get_label(),
    size="medium",
    color=plot1.get_color(),
    ha="left",
    va="bottom",
    rotation=2.0,
)

ax0.text(
    t[t_idx],
    T[T_idx, nelems_battery]-10.0,
    " " + plot2.get_label(),
    size="medium",
    color=plot2.get_color(),
    ha="left",
    va="top",
    rotation=-5.0,
)

# Mark where thermal runaway occurs
ax0.fill_between([0.0, 10.0], 0.0, 1.0, color='tab:gray', alpha=0.3, transform=ax0.get_xaxis_transform())
ax1.fill_between([0.0, 10.0], 0.0, 1.0, color='tab:gray', alpha=0.3, transform=ax1.get_xaxis_transform())
ax1.annotate("", (0.0, 5400.0), xytext=(10.0, 5400.0), arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0),
             annotation_clip=False)
ax1.text(
    0.0,
    5500.0,
    "Thermal runaway",
    size="small",
    color="black",
    ha="left",
    va="bottom"
)

ax0.set_ylabel(r'Temperature at PCM boundaries ($^\circ$C)', fontsize=12)
ax1.set_xlabel('Time (min.)')
ax1.set_ylabel(r'Rate of change of battery temperature ($^\circ$C/min.)', fontsize=12)

plt.savefig("PCM_runaway.pdf", transparent=True)
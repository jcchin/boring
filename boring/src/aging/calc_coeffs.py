'''

Calculate empirical coefficients based on aging tests.
This only needs to get run once to generate the best fit for total_aging.py
a_cap_c1/2/3, etc.


References:

A holistic aging model for Li(NiMnCo)O2 based 18650 lithium-ion batteries
https://www.sciencedirect.com/science/article/pii/S0378775314001876?via%3Dihub
'''
from scipy.optimize import curve_fit
import numpy as np

# Experimental data in Fig 8, 
# Voltage vs alpha dependency (linear)
# extracted via https://markummitchell.github.io/engauge-digitizer/
a_cap_data = np.array([0.0006626, 0.0009519, 0.0011814, 0.0017308, 0.0017146, 0.0021363, 0.002006, 0.0026501, 0.0027362, 0.003, 0.003101])  # Y-axis blue dots
a_res_data = np.array([0.0020048, 0.0017382, 0.0015433, 0.0030417, 0.0027981, 0.0035337, 0.0032885, 0.0043269, 0.0044712, 0.0053475, 0.0056406])  # Y-axis red dots
V_data = np.array([3.33259, 3.49631, 3.56276, 3.62714, 3.70007, 3.7498, 3.86989, 3.96506, 4.01612, 4.07222, 4.11774])      # X-axis
# define linear function to be fit 
def fig8func(V,a1,a2):
    return a1*V + a2

popt, pcov = curve_fit(fig8func,V_data,a_cap_data)
a1 = popt[0]
a2 = popt[1]


E_cap = 58000  # J/mol*K  Activation Energy for capacity
E_res = 49800  # J/mol*K  Activation Energy for resistance
R =  8.3144621  # J/mol*K
T0 = 323.15 # degK
V0 = 3.699  # Volts
a3 = 5000000

# Experimental data in Fig 9,
# Temperature vs alpha dependency (logarithmic)
# absolute values rather than axis labels
aT_cap_data = np.array([0.001661557273, 0.0009118819656, 0.0005814416122])
aT_res_data = np.array([0.002739444819, 0.001360368038, 0.001170879621])
T_data = np.array([323.15,313.15,308.15])
# define logarithmic function to be fit 
def fig9func(T,a3,E):
    return a3*np.exp(-E/(R*T))

popt2, pcov2 = curve_fit(fig9func,T_data,aT_cap_data)
a3 = popt2[0]
E = popt2[1]

# print(a3,E, a3*np.exp(-E/(R*323.15)))

def a_V(V):  # Eq. 5
    return a1*V+a2

def a_T(E,T):  # Eq. 6
    return a3*np.exp(-E/(R*T))

def a_O(T,V):  # Eq. 9
    return a_V(V)+a_T(E_cap,T)/2.

def a_total(a,T,V): # Eq. 13
    return (a*a_O(T0,V0))/(a_V(V0)*a_T(E_cap,T0))


a_cap_c1 = a_total(a1,T0,V0)
a_cap_c2 = a_total(a2,T0,V0)
a_cap_c3 = -E/R

print("alpha = (",a_cap_c1,"V", a_cap_c2,") *10^6 * np.exp(",a_cap_c3,"/T)")
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

# Numpy Array of experimental data in Fig 8
a_cap_data = np.array([])  # Y-axis blue dots
a_res_data = np.array([])  # Y-axis red dots
V_data = np.array([])      # X-axis

# define function to be fit 
def func1(V,a1,a2):
    return a1*V + a2

popt, pcov = curve_fit(func1,a,V)
a1 = popt[0]
a2 = popt[1]


E_cap = 58.0  # kJ/mol*K  Activation Energy for capacity
E_res = 49.8  # kJ/mol*K  Activation Energy for resistance
R =  8.3144621  # J/mol*K
T0 = 323.15 # degK
V0 = 3.699  # Volts

def a_V(V):
    return a1*V+a2

def a_T(T):
    return a1*np.exp(-E/RT)

def a_O(T,V):
    return a_V(V)+a_T(T)/2.

def a_total(T,V):
    return a_V(V)*a_T(T)*a_O/(a_V(V0)*a_T(T0))
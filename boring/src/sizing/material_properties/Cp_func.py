"""
These helper functions are a placeholder until a derivative friendly method replaces it (akima, convolution)


Apparent Heat Capacity Method
# https://link.springer.com/article/10.1007/s10973-019-08541-w

Author: Jeff Chin
"""

def cp_func(T, T1 = 60, T2 = 65, Cp_low = 1.5, Cp_high=50):  #kJ/kgK
    if T > T1 and T < T2:
        Cp = Cp_high
    else:
        Cp = Cp_low
    return Cp


def cp_dT_deriv_func(T):
    return 0
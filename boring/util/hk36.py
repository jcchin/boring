"""
2520 cells
NCR18650GA
0.048g
3.6V
3450mAh (@25C)


7p - unit
12s * u - module
2p * 15s * m - pack


Subsystem     |  Percent |  Mass
=====================================
Battery cells |   58%    |
Housing       |   6%     |
Cooling       |   3.5%   |
PCC           |   10.2%  |
BMS           |   13%    |
Wires         |   9.4%   |
______________________________________

"""
W_cell = 0.048
n_modules = 30
n_cell = 7
n_units = 12

W_batt = n_cell*n_units*W_cell # 4.032kg
W_ultem = 0.4  # kg per module
W_screws = 0.0167  # kg per module (screws+straps)
W_house = (W_ultem + W_screws)*n_modules
W_PCC = 0.7  # kg Phase Change Composite per module
W_fins = 0.25 # kg  cooling fins, air ducts
W_bms = 0.9 # kg PCBs + components + leaf springs

W_6awg = 0.014 * 63.5 # 0.014 kft, 63.5kg/kft  (2p modules)
W_4awg = 0.2 * 93.0 #0.2 kft, 93kg/kft (15s modules)

W_wire = (W_6awg+W_4awg)/30 # 0.667 kg per module

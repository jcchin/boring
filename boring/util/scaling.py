"""
This plot compares naive battery weight scaling vs energy-based scaling


The individual cells on X-57 have a "nominal" energy density of 225 Wh/kg and the pack
has a nominal energy density of 149 Wh/kg (55.3 kWh total, 370.86 kg total).
Since approximately 125kg is dedicated to packaging, 
halving the packaging would bring the effective pack energy density to 180 Wh/kg.


Pack Mass (in kg) = Pack Size (in kWh) * (1000/energy density +2.26)   <--- X-57
370kg =55.3*((1000/225)+2.26)


y = x/((0.00226)x+1)

vs.
y = 0.6666667x


Author(s): Jeff Chin

"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

size=30
cell_dens = np.zeros((size))
pack_dens = np.zeros((size))


slope = 0
intercept = 0
r_value = 0
s_label = 'Wh/kg cell'
popt = np.zeros((4))


df = pd.read_csv('../metamodel/scaling.csv')
cell_dens = df['cell_dens'].to_numpy()
pack_dens = df['pack_dens'].to_numpy()

x = cell_dens #np.arange(-20., 600, 20)
y1 = x / ((0.00226) * x + 1.)
y2 = 0.6666667 * x

slope, intercept, r_value, p_value, std_err = stats.linregress(cell_dens,pack_dens)
def func(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

popt,pcov = curve_fit(func, cell_dens,pack_dens)
print(popt)

y3 = func(cell_dens, *popt)-10
#plt.plot(cell_dens,pack_dens)
# plt.plot(cell_dens,cell_dens*slope+intercept) # *0.412+80
# plt.plot(cell_dens, (170/233)*cell_dens,'g--')
plt.plot(cell_dens, y3, 'b-', label='thermal based scaling')#, label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(popt))
# plt.set_ylabel('Wh/kg pack')
# s_label = s_label + '\n pack = {:.2f}*cell + {:2.1f} (R = {:.2f})'.format(slope,intercept,r_value)
# plt.set_xlabel(s_label)


#plt.plot(x, y1, 'b-', label='thermal based scaling')
plt.plot(x, y2, 'g--', label='naive linear scaling')
plt.plot(225, 149, 'ro', label='X-57 Battery')
# plt.plot(259, 150, 'r*', label='ER HK-36') #NCR18650GA, 2520
#plt.plot(135, 'r^', label='Alpha Electro')

plt.fill_between(x, y3, y2, where=(y3 < y2), color='C0', alpha=0.3)  # shade region between

SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 28
plt.annotate("",
             xy=(600, 305), xycoords='data',
             xytext=(600, 400), textcoords='data',
             arrowprops=dict(arrowstyle="<->",
                             connectionstyle="arc3"),
             )
plt.annotate("Over-Estimated \n Performance",
             xy=(0.8, 0.74), xycoords='axes fraction',
             textcoords='axes fraction',
             )
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.xlabel(r'cell energy density $\frac{Wh}{kg}$')
plt.ylabel(r'pack energy density $\frac{Wh}{kg}$')
plt.legend()
plt.show()

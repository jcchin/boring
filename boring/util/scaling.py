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
import matplotlib.pyplot as plt

x = np.arange(0., 600, 5)
y1 = x / ((0.00226) * x + 1.)
y2 = 0.6666667 * x

SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 28

plt.plot(x, y1, 'b-', label='thermal based scaling')
plt.plot(x, y2, 'g--', label='naive weight scaling')
plt.fill_between(x, y1, y2, where=(y1 < y2), color='C0', alpha=0.3)  # shade region between
plt.annotate("",
             xy=(600, 250), xycoords='data',
             xytext=(600, 400), textcoords='data',
             arrowprops=dict(arrowstyle="<->",
                             connectionstyle="arc3"),
             )
plt.annotate("Over-Estimated \n Performance",
             xy=(0.75, 0.5), xycoords='axes fraction',
             xytext=(0.75, 0.65), textcoords='axes fraction',
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

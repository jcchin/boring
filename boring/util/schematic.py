# pip install schemdraw
# (^ requires python 3.8)

# conda install -c conda-forge pyspice

# draw the schematic

import schemdraw
import schemdraw.elements as elm

d = schemdraw.Drawing()
Reex = d.add(elm.Resistor().down().label('Re,ex'))
Rew = d.add(elm.Resistor().down().label('Re,w'))
d.add(elm.Resistor().down().label('Re,wk'))
d.add(elm.Resistor().down().label('Re,inter'))
d.add(elm.Resistor().right().label('Rv'))
d.add(elm.Resistor().up().label('Rc,inter',loc='bottom'))
d.add(elm.Resistor().up().label('Rc,wk',loc='bottom'))
Rcw = d.add(elm.Resistor().up().label('Rc,w',loc='bottom'))
Rcex = d.add(elm.Resistor().up().label('Rc,ex',loc='bottom'))
d.add(elm.Resistor().endpoints(Reex.end,Rcex.start).label('Ra,w'))
d.add(elm.Resistor().endpoints(Rew.end,Rcw.start).label('Ra,wk'))


d.draw()

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

circuit = Circuit('HeatPipe')

circuit.V('input', 1, circuit.gnd, 10@u_V)
circuit.R('Reex', 1, 2, 2@u_kΩ)
circuit.R('Rew', 2, 3, 2@u_kΩ)
circuit.R('Rewk', 3, 4, 2@u_kΩ)
circuit.R('Rintere', 4, 5, 2@u_kΩ)
circuit.R('Raw', 2, 9, 1@u_kΩ)
circuit.R('Rawk', 3, 8, 1@u_kΩ)
circuit.R('Rv', 5, 6, 1@u_kΩ)
circuit.R('Rinterc', 6, 7, 2@u_kΩ)
circuit.R('Rcwk', 7, 8, 2@u_kΩ)
circuit.R('Rcw', 8, 9, 1@u_kΩ)
circuit.R('Rcex', 9, circuit.gnd, 1@u_kΩ)


for resistance in (circuit.RReex, circuit.RRcex):
    resistance.minus.add_current_probe(circuit) # to get positive value

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.operating_point()

for node in analysis.nodes.values():
    print('Node {}: {:5.2f} V'.format(str(node), float(node))) # Fixme: format value + unit

# print(circuit)
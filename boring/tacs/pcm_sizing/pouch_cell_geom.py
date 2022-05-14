import numpy as np

from egads4py import egads

def make_geometry():
    '''
    Make pouch-cell battery geometry with a face for heat-pipe-cooling
    '''

    w = 54.0e-3   # total width
    l = 61.0e-3  # total length
    bat_t = 5.60e-3   # thickness of battery
    pcm_t = 5.0e-3    # thickness of pcm
    dw = 10.0e-3  # width of heat-pipe centered on pouch-cell

    # Make the battery
    body = ctx.makeSolidBody(egads.BOX, rdata=[[0.0, 0.0, 0.0], [0.5*(w-dw), l, bat_t]])
    m1 = ctx.makeTopology(egads.MODEL, children=[body])

    body = ctx.makeSolidBody(egads.BOX, rdata=[[0.5*(w-dw), 0.0, 0.0], [dw, l, bat_t]])
    m2 = ctx.makeTopology(egads.MODEL, children=[body])

    body = ctx.makeSolidBody(egads.BOX, rdata=[[0.5*(w+dw), 0.0, 0.0], [0.5*(w-dw), l, bat_t]])
    m3 = ctx.makeTopology(egads.MODEL, children=[body])

    # Add the pcm on top
    body = ctx.makeSolidBody(egads.BOX, rdata=[[0.0, 0.0, bat_t], [0.5*(w-dw), l, pcm_t]])
    m4 = ctx.makeTopology(egads.MODEL, children=[body])

    body = ctx.makeSolidBody(egads.BOX, rdata=[[0.5*(w-dw), 0.0, bat_t], [dw, l, pcm_t]])
    m5 = ctx.makeTopology(egads.MODEL, children=[body])

    body = ctx.makeSolidBody(egads.BOX, rdata=[[0.5*(w+dw), 0.0, bat_t], [0.5*(w-dw), l, pcm_t]])
    m6 = ctx.makeTopology(egads.MODEL, children=[body])

    return [m1, m2, m3, m4, m5, m6]

# Create the egads context
ctx = egads.context()

# Create the egads battery model and save it as a step file
models = make_geometry()
for i, m in enumerate(models):
    m.saveModel('model{0}.step'.format(i), overwrite=True)
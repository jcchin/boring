"""
These helper functions are a placeholder until a derivative friendly method replaces it (akima, convolution)


Apparent Heat Capacity Method
# https://link.springer.com/article/10.1007/s10973-019-08541-w

Author: Jeff Chin
"""
from __future__ import absolute_import

import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt

class PCM_Cp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)  # parallel execution

    def setup(self):
        nn = self.options['num_nodes']

        # pad geometry
        self.add_input('T', 334 * np.ones(nn), units='K', desc='PCM temp')
        # self.add_input('T_lo', 333 * np.ones(nn), units='K', desc='PCM lower temp transition point')
        # self.add_input('T_hi', 338 * np.ones(nn), units='K', desc='PCM upper temp transition point')
        # self.add_input('Cp_lo', 333 * np.ones(nn), units='K', desc='PCM lower specific heat')
        # self.add_input('Cp_hi', 338 * np.ones(nn), units='K', desc='PCM upper specific heat')

        # outputs
        self.add_output('cp_pcm', 1.54 * np.ones(nn), units='kJ/(kg*K)', desc='specific heat of the pcm')

    def setup_partials(self):
        # self.declare_partials('*', '*', method='cs')
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.declare_partials('cp_pcm', ['T'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        T = inputs['T']
        T_lo = 333.
        T_hi = 338.
        Cp_lo = 1.5
        Cp_hi = 6.5
        a = 3  # sigma smoothing function

        outputs['cp_pcm'] = cp_sigmoid(T, T_lo, T_hi, Cp_lo, Cp_hi, a)

    def compute_partials(self, inputs, partials):
        T = inputs['T']
        T_lo = 333.
        T_hi = 338.
        Cp_lo = 1.5
        Cp_hi = 6.5
        a = 3  # sigma smoothing function

        partials['cp_pcm', 'T'] = cp_sigmoid_deriv(T, T_lo, T_hi, Cp_lo, Cp_hi, a)

def cp_sigmoid(x, start=0, stop=1, low=0, high=1, a = 10.0):
    y = (high-low)*(0.25 - 0.25 * np.tanh(a*(-stop + x)))*(np.tanh(a*(-start + x)) + 1)+low
    return y

def cp_sigmoid_deriv(x, start=0, stop=1, low=0, high=1, a = 10.0):
    dy = (high-low)*0.25*a*((np.tanh(a*(start - x)) - 1)/np.cosh(a*(stop - x))**2 + (np.tanh(a*(stop - x)) + 1)/np.cosh(a*(start - x))**2)
    return dy


if __name__ == '__main__':
    
    T1 = 333 # start of filter impulse
    T2 = 338 # stop of filter impulse
    sigma = 1 # smoothness constant (helps keep derivatives tractable)
    high = 50
    low = 1.5
    # make example
    x = np.linspace(T1 - 10, T2 + 10, 100)
    y = cp_sigmoid(x, T1, T2, low, high, sigma)
    dy = cp_sigmoid_deriv(x, T1, T2, low, high, sigma)
    plt.figure()
    plt.plot(x,y, label='y')
    plt.plot(x,dy, label='dy')
    plt.legend()

    plt.show()

    from openmdao.api import Problem

    nn = 1

    prob = Problem()
    prob.model.add_subsystem('comp1', PCM_Cp(num_nodes=nn), promotes_outputs=['*'], promotes_inputs=['*'])
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)

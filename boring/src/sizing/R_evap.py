# -*- coding: utf-8 -*-
"""
Evaporator secion thermal resistances

Created on Fri Oct 23 11:06:38 2020

@author: Karsten Look
"""

from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om


class R_evapComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn=self.options['num_nodes']

        #TODO Add units
        self.add_input('alpha')
        self.add_input('h_fg')
        self.add_input('T_hp')
        self.add_input('v_fg')
        self.add_input('R_g')
        self.add_input('P_v')
        self.add_input('D_od')
        self.add_input('r_i')
        self.add_input('k_w')
        self.add_input('L_evap', 0.01, units='m',  desc='')
        self.add_input('D_v')
        self.add_input('k_wk')
        self.add_input('h_intere')
        self.add_input('A_intere')
         
        self.add_output('h_intere')
        self.add_output('R_we')
        self.add_output('R_wke')
        self.add_output('R_intere')


    def setup_partials(self):        
        self.declare_partials('h_intere', ['alpha', 'h_fg', 'T_hp', 'v_fg', 'R_g', 'T_hp', 'P_v', 'v_fg', 'h_fg'])
        self.declare_partials('R_we', ['D_od', 'r_i', 'k_w', 'L_evap'])
        self.declare_partials('R_wke', ['r_i', 'D_v', 'k_wk', 'L_evap'])
        self.declare_partials('R_intere', ['h_intere', 'A_intere'])
    
    def compute(self,inputs, outputs):
        alpha = inputs['alpha']
        h_fg = inputs['h_fg']
        T_hp = inputs['T_hp']
        v_fg = inputs['v_fg']
        R_g = inputs['R_g']
        P_v = inputs['P_v']
        D_od = inputs['D_od']
        r_i = inputs['r_i']
        k_w = inputs['k_w']
        L_evap = inputs['L_ev]p']
        D_v = inputs['D_v']
        k_wk = inputs['k_wk']
        A_intere = inputs['A_intere']
        h_intere = inputs['h_intere']      
        
        outputs['h_intere'] = 2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg)) # Karsten
        outputs['R_we'] = np.log((D_od/2)/(r_i))/(2*np.pi*k_w*L_evap) # Karsten
        outputs['R_wke'] = np.log((r_i)/(D_v/2))/(2*np.pi*k_wk*L_evap) # Karsten
        outputs['R_intere'] = 1/(h_intere*A_intere) # Karsten                
        
        '''
        Formatted into wolfram alpha queries

        # outputs['h_intere'] = 2*alpha/(2-alpha)*(h_fg**2/(T_hp*v_fg))*np.sqrt(1/(2*np.pi*R_g*T_hp))*(1-P_v*v_fg/(2*h_fg)) # Karsten        
            h_intere = h
            alpha = a
            h_fg = f
            T_hp = t
            v_fg = v
            R_g = r
            P_v = p
            
            partial wrt a of 2*a/(2-a)*(f^2/(t*v))*sqrt(1/(2*pi*r*t))*(1-p*v/(2*f))              
               a'
               (f^2 Sqrt[2/Pi] Sqrt[1/(r t)] (1 - (p v)/(2 f)))/((2 - a) t v) + (Sqrt[2] a f^2 Sqrt[1/(r t)] (1 - (p v)/(2 f)))/((2 - a)^2 Sqrt[Pi] t v)
                
               f'
               (a p Sqrt[1/(r t)])/((2 - a) Sqrt[2 Pi] t) + (2 a f Sqrt[2/Pi] Sqrt[1/(r t)] (1 - (p v)/(2 f)))/((2 - a) t v)
               
               t'
               -((a f^2 (1 - (p v)/(2 f)))/((2 - a) Sqrt[2 Pi] r Sqrt[1/(r t)] t^3 v)) - (a f^2 Sqrt[2/Pi] Sqrt[1/(r t)] (1 - (p v)/(2 f)))/((2 - a) t^2 v)
               
               v'
               -((a f p Sqrt[1/(r t)])/((2 - a) Sqrt[2 Pi] t v)) - (a f^2 Sqrt[2/Pi] Sqrt[1/(r t)] (1 - (p v)/(2 f)))/((2 - a) t v^2)
               
               r'
               -((a f p Sqrt[1/(r t)])/((2 - a) Sqrt[2 Pi] t v)) - (a f^2 Sqrt[2/Pi] Sqrt[1/(r t)] (1 - (p v)/(2 f)))/((2 - a) t v^2)
               
               p'
                -((a f Sqrt[1/(r t)])/((2 - a) Sqrt[2 Pi] t))
                
                
                
                
            
               a'
               (f^2 Sqrt[2/Pi] Sqrt[1/(r t)] (1 - (p v)/(2 f)))/((2 - a) t v) + (Sqrt[2] a f^2 Sqrt[1/(r t)] (1 - (p v)/(2 f)))/((2 - a)^2 Sqrt[Pi] t v)
                
               f'
               (a p Sqrt[1/(r t)])/((2 - a) Sqrt[2 Pi] t) + (2 a f Sqrt[2/Pi] Sqrt[1/(r t)] (1 - (p v)/(2 f)))/((2 - a) t v)
               
               t'
               -((a f^2 (1 - (p v)/(2 f)))/((2 - a) Sqrt[2 Pi] r Sqrt[1/(r t)] t^3 v)) - (a f^2 Sqrt[2/Pi] Sqrt[1/(r t)] (1 - (p v)/(2 f)))/((2 - a) t^2 v)
               
               v'
               -((a f p Sqrt[1/(r t)])/((2 - a) Sqrt[2 Pi] t v)) - (a f^2 Sqrt[2/Pi] Sqrt[1/(r t)] (1 - (p v)/(2 f)))/((2 - a) t v^2)
               
               r'
               -((a f p Sqrt[1/(r t)])/((2 - a) Sqrt[2 Pi] t v)) - (a f^2 Sqrt[2/Pi] Sqrt[1/(r t)] (1 - (p v)/(2 f)))/((2 - a) t v^2)
               
               p'
                -((a f Sqrt[1/(r t)])/((2 - a) Sqrt[2 Pi] t))
            
            
            
        # outputs['R_we'] = np.log((D_od/2)/(r_i))/(2*np.pi*k_w*L_evap) # Karsten
        # outputs['R_wke'] = np.log((r_i)/(D_v/2))/(2*np.pi*k_wk*L_evap) # Karsten
        # outputs['R_intere'] = 1/(h_intere*A_intere) # Karsten 
        '''

    def compute_partials(self, inputs, partials):
        partials['h_intere', 'alpha'] = 
        partials['h_intere','h_fg'] = 
        partials['h_intere','T_hp'] = 
        partials['h_intere','v_fg'] = 
        partials['h_intere','R_g'] = 
        partials['h_intere','T_hp'] = 
        partials['h_intere','P_v'] = 
        partials['h_intere','v_fg'] = 
        partials['h_intere','h_fg'] = 

        partials['R_we', 'D_od'] = 
        partials['R_we', 'r_i'] = 
        partials['R_we', 'k_w'] = 
        partials['R_we', 'L_evap'] = 


        partials['R_wke', 'r_i'] = 
        partials['R_wke', 'D_v'] = 
        partials['R_wke', 'k_wk'] = 
        partials['R_wke', 'L_evap'] = 

        partials['R_intere', 'h_intere'] = 
        partials['R_intere', 'A_intere'] =        
        
        
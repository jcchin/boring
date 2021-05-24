"""
This code is for estimating the performance limitations of a heat pipe.

Equations are from:
    "Heat Pipe Science and Technology", 2nd Edition, 2016
    Amir Faghri

Results are validated against:
    "Mathematical Model for Heat Transfer Limitations of Heat Pipe"
    Patrik Nemec, et al.
    Mathematical and Computer Modelling, 2013
    
Author: Ezra McNichols
        NASA Glenn Research Center
        Turbomachinery and Turboelectric Systems

Modified for OpenMDAO compatibility by Karsten Look
        NASA Glenn Research Center
        Turbomachinery and Turboelectric Systems Branch
"""

#################################################################################

'''
#TODO where should these go in a test? Where are the results to check against?
# Validation Parameters (Nemec et al, 2013, "Mathemtatical model for heat transfer limitations of heat pipe")
r_i=0.0065
r_hv=0.005
L_e=0.15
L_a=0.2
L_c=0.15
L_eff=0.5*L_e+L_a+0.5*L_c
A_w=0.000054
k_s=393
r_p=0.0001/2
epsilon=0.65
t=0.0015
L_t=0.5
T_w=298
g=9.81
phi=180
r_ce=r_p
r_n=25e-6
K=(r_p*2)**2*epsilon**3/(150*(1-epsilon)**2)
A_v=np.pi*r_hv**2
'''

#################################################################################

from __future__ import absolute_import
import numpy as np
from math import pi
import openmdao.api as om
import matplotlib.pyplot as plt

class HeatPipeLimitsComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('geom', values=['round', 'flat'], default='round')

    def setup(self):
        nn = self.options['num_nodes']
        geom = self.options['geom']


        if geom == 'round':
            #inputs specific to round  heat pipes
            #TODO

        elif geom == 'flat':
            #inputs specific to flat heat pipes
            #TODO
            
        """
        #TODO
        Nonexistant variables:
                #TODO
                self.add_input('A_v', val=1.0*np.ones(nn), units='m**2', desc='Cross sectional area of vapor core')  # doesn't yet exist, but should be calculated in hp_geom.py as (XS:A_v)
                self.add_input('K', val=1.0*np.ones(nn), units='m**2', desc='Wick permeability')                     # doesn't exist
                self.add_input('phi', val=1.0*np.ones(nn), units='deg', desc='Angle of heat pipe wrt vertical 0 degrees being condenser on bottom')  # doesn't exist
                self.add_input('r_ce', val=1.0*np.ones(nn), units='m', desc='Wick capillary radius in the evaporator') #called r_eff in paper        # doesn't exist
                self.add_input('r_n', val=1.0*np.ones(nn), units='m', desc='nucleation radius') # ???
        """

        #TODO
        #Untabbed need to verify exist in model
            #Tabbed known to exist in model

        #Geometric variables
                self.add_input('XS:A_v', val=1.0*np.ones(nn), units='m**2', desc='Cross sectional area of vapor core')
                self.add_input('XS:A_w', val=1.0*np.ones(nn), units='m**2', desc='Cross sectional area of wick')
                self.add_input('LW:L_eff', val=1.0*np.ones(nn), units='m', desc='effective length of the heat pipe')
                self.add_input('length_hp', val=1.0*np.ones(nn), units='m', desc='total length of heat pipe') #L_t in paper
                self.add_input('XS:D_v', val=1.0*np.ones(nn), units='m', desc='vapor core radius') #XS:D_v/2 = r_hv = r_v in paper
                self.add_input('XS:r_i', val=1.0*np.ones(nn), units='m', desc='inner container radius')
        self.add_input('phi', val=1.0*np.ones(nn), units='deg', desc='Angle of heat pipe wrt vertical 0 degrees being condenser on bottom')

        #Heat pipe material props
                self.add_input('epsilon', val=1.0*np.ones(nn), units='ul', desc='Porosity (unitless) of sintered capillary structure')
        self.add_input('K', val=1.0*np.ones(nn), units='m**2', desc='Wick permeability')
        self.add_input('r_ce', val=1.0*np.ones(nn), units='m', desc='Wick capillary radius in the evaporator') #called r_eff in paper
        self.add_input('r_p', val=1.0*np.ones(nn), units='m', desc='Average capillary radius of the wick')  #called r_c,ave in paper "can often be approximated by r_eff"

        #Fluid Properties
                self.add_input('cp_v', val=1.0 * np.ones(nn), desc='vapor specific heat')
                self.add_input('h_fg', val=1.0 * np.ones(nn), units='J/kg', desc='latent heat') #called l_v in paper
                self.add_input('k_l', val=1.0 * np.ones(nn), units='W/(m*K)', desc='liquid conductivity') #Is this equivalent to lambda_m in the paper (thermal conductivity of the heat pipe material) #TODO
                self.add_input('mu_l', val=1.0 * np.ones(nn), units='N*s/m**2', desc='liquid viscosity')
                self.add_input('mu_v', val=1.0 * np.ones(nn), units='N*s/m**2', desc='vapor viscosity')
                self.add_input('P_v', val=1.0 * np.ones(nn), units='Pa', desc='pressure')
                self.add_input('R_g', val=1.0 * np.ones(nn), units='J/kg/K', desc='gas constant of the vapor')
                self.add_input('rho_l', val=1.0 * np.ones(nn), units='kg/m**3', desc='density of liquid')
                self.add_input('rho_v', val=1.0 * np.ones(nn), units='kg/m**3', desc='density of vapor')
                self.add_input('sigma_l', val=1.0 * np.ones(nn), units='N/m', desc='surface tension')
                self.add_input('T_hp', val=1.0 * np.ones(nn), units='degC', desc='vapor temperature')

                g = 9.80665 #Acceleration due to gravity, m/s**2
        self.add_input('k_s', val=1.0*np.ones(nn), units='<enter units here>', desc='TBD') #Is this equivalent to lambda_l in the paper (thermal conductivity of liquid)? #TODO
        self.add_input('r_n', val=1.0*np.ones(nn), units='m', desc='nucleation radius')


        self.add_output('q_boiling', val=1.0*np.ones(nn), units='W', desc='heat pipe boiling limit')
        self.add_output('q_sonic', val=1.0*np.ones(nn), units='W', desc='heat pipe sonic limit')
        self.add_output('q_ent', val=1.0*np.ones(nn), units='W', desc='heat pipe entrainment limit')
        self.add_output('q_vis', val=1.0*np.ones(nn), units='W', desc='heat pipe viscous limit')
        self.add_output('q_cap', val=1.0*np.ones(nn), units='W', desc='heat pipe capillary limit')


    # Add outputs for all properties
    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.declare_partials('q_boiling', ['LW:L_eff', 'T_hp', 'sigma_l', 'h_fg', 'rho_v', 'XS:r_i', 'r_n', 'r_ce'], rows=ar, cols=ar, method='cs')
        self.declare_partials('q_sonic', ['XS:A_v', 'rho_v', 'h_fg', 'T_hp'], rows=ar, cols=ar, method='cs')
        self.declare_partials('q_ent', ['XS:A_v','h_fg','sigma_l','rho_v','r_p'], rows=ar, cols=ar, method='cs')
        self.declare_partials('q_vis', ['h_fg','rho_v','P_v','mu_v','LW:L_eff','XS:A_v'], rows=ar, cols=ar, method='cs')
        self.declare_partials('q_cap', ['sigma_l','rho_l','h_fg','K','XS:A_w','mu_l','LW:L_eff','r_ce','length_hp','phi'], rows=ar, cols=ar, method='cs')

    def compute(self, inputs, outputs):
        geom = self.options['geom']

        if geom == 'round':
            r_hv = inputs['XS:D_v']/2

            # v_fg = 1/rho_v-1/rho_l
            R_g=inputs['P_v']/(inputs['T_hp']*inputs['rho_v']) #TODO what does this represent?
            cv_v=inputs['cp_v']-R_g #TODO what does this represent?
            gamma=inputs['cp_v']/cv_v #TODO what does this represent?
            #Is k_eff equivalent to lambda_eff in paper? Was not able to algabraically reduce to match paper's equation #TODO
            k_eff = inputs['k_s']*(2+inputs['k_l']/inputs['k_s']-2*inputs['epsilon']*(1-inputs['k_l']/inputs['k_s']))/(2+inputs['k_l']/inputs['k_s']+inputs['epsilon']*(1-inputs['k_l']/inputs['k_s']))

            outputs['q_boiling'] = 4*np.pi*inputs['LW:L_eff']*k_eff*inputs['T_hp']*inputs['sigma_l']/(inputs['h_fg']*inputs['rho_v']*np.log(inputs['XS:r_i']/r_hv))*(1/inputs['r_n']-1/inputs['r_ce'])
            outputs['q_sonic'] = inputs['XS:A_v']*inputs['rho_v']*inputs['h_fg']*np.sqrt(gamma*R_g*inputs['T_hp'])*np.sqrt(1+gamma)/(2+gamma)
            outputs['q_ent'] = inputs['XS:A_v']*inputs['h_fg']*np.sqrt(inputs['sigma_l']*inputs['rho_v']/(2*inputs['r_p']))
            outputs['q_vis'] = (r_hv*2)**2*inputs['h_fg']*inputs['rho_v']*inputs['P_v']/(64*inputs['mu_v']*inputs['LW:L_eff'])*inputs['XS:A_v']
            outputs['q_cap'] = inputs['sigma_l']*inputs['rho_l']*inputs['h_fg']*inputs['K']*inputs['XS:A_w']/(inputs['mu_l']*inputs['LW:L_eff'])*(2/inputs['r_ce']-inputs['rho_l']*inputs['g']*inputs['length_hp']*np.cos(inputs['phi']*np.pi/180)/inputs['sigma_l'])

        if geom == 'flat': #Rectangular
            #TODO

        ################################ Heat pipe Calculations ################################ 
        # Original equations
        # k_eff=k_s*(2+k_l/k_s-2*epsilon*(1-k_l/k_s))/(2+k_l/k_s+epsilon*(1-k_l/k_s))
        # q_boiling=4*np.pi*L_eff*k_eff*T_hp*sigma_l/(h_fg*rho_v*np.log(r_i/r_hv))*(1/r_n-1/r_ce)
        # q_sonic=A_v*rho_v*h_fg*np.sqrt(gamma*R_g*T_hp)*np.sqrt(1+gamma)/(2+gamma)
        # q_ent=A_v*h_fg*np.sqrt(sigma_l*rho_v/(2*r_p))
        # q_vis=(r_hv*2)**2*h_fg*rho_v*P_v/(64*mu_v*L_eff)*A_v
        # q_cap=sigma_l*rho_l*h_fg*K*A_w/(mu_l*L_eff)*(2/r_ce-rho_l*g*L_t*np.cos(phi*np.pi/180)/sigma_l)



if __name__ == '__main__':
    from openmdao.api import Problem
    nn = 1
    # geom='flat' #TODO Do I need to change this?
    prob = Problem()
    prob.model.add_subsystem(name = '???', #TODO
                            subsys = HeatPipeLimitsComp(num_nodes=nn),
                            promotes_inputs = ['XS:A_v','XS:A_w','cp_v','epsilon','g','gamma','h_fg','K','k_l','k_s','LW:L_eff','length_hp','mu_l','mu_v','P_v','phi','r_ce','R_g','r_hv','XS:r_i','r_n','r_p','rho_l','rho_v','sigma_l','T_hp'],
                            promotes_outputs = ['q_boiling','q_sonic','q_ent','q_vis','q_cap'])
                            prob.setup(force_alloc_complex=True)
    prob.run_model()
    ##############################################################################
    # Plots limitations
    # plt.rc('font', family='serif')
    # ax=plt.figure(1,figsize=(9,6)).add_subplot(1,1,1)
    # plt.semilogy(T_hpfp,q_cap,color='red')
    # plt.semilogy(T_hpfp,q_boiling,color='orange')
    # plt.semilogy(T_hpfp,q_sonic,color='pink')
    # plt.semilogy(T_hpfp,q_ent,color='green')
    # plt.semilogy(T_hpfp,q_vis,color='blue')
    # plt.grid(True,which="both")
    # plt.xlabel('Temperature [C]')
    # plt.ylabel('Heat [W]')

    # ax.legend(['Capillary Limit','Boiling Limit','Sonic Limit','Entrainment Limit','Viscous Inertia Limit'])
    # plt.show()

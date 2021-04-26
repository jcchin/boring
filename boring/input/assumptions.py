# Dictionary must be called 'inputs', to be used with boring.src.utils.load_inputs
# first item is the value (which will be automatically vectorized), the second is an optionally provided unit


"""
Revision Reason: First attempt
Revised by: Jeff Chin
Revised on: 12/3/20
"""

inputs = {

    # ----------------------------------------Heatpipe Sizing Vars-----------------------------------------#
    #                   """ Currently no check to ensure feasability of dimensions """
    'epsilon': [0.46],  # Porosity of the wick (0=void, 1=solid)
    'L_eff': [0.045, 'm'],  # Effective Length
    'XS:D_od': [0.006],  # Outer diameter of overall heatpipe
    'XS:D_v': [0.00362],  # Outer diameter of HP vapor chamber
    'XS:t_wk': [0.00069],  # Thickness of the wick in the interior of the HP
    'XS:t_w': [0.0005],  # Thickness of the HP wall
    # 'L_heatpipe': [0.30],  # Overall length of the HP (including adiabatic portion)
    # 'liq_density':[1000]    # Density of the liquid in HP
    # 'fill_liq':[0.70]       # Fill perentage of liquid in HP (1=full, 0=empty)
    'LW:L_adiabatic': [0.03],  # Length of each adiabatic section
    'k_w': [11.4],  # heat pipe wall thermal conductivity

    # ----------------------------------------Battery Sizing Vars-----------------------------------------#
    #                   """ Currently no check to ensure feasability of dimensions """

    'cond.L_flux': [0.02, 'm'],  # Length of Condensor exposed to heat pipe
    'evap.L_flux': [0.01, 'm'],
    'T_rate_cond.c_p': [1500],  # Specific Heat
    'T_rate_cond.mass': [.06],
    'T_rate_cond2.c_p': [1500],
    'T_rate_cond2.mass': [.06],
    'evap.Rex.R': [0.0001],  # Thermal resistance external to heat pipe wall
    'cond.Rex.R': [0.0001],
    'evap.h_c': [1200], #  Heat Transfer Coef
    'cond.h_c': [1200], #  Heat Transfer Coef
    'evap.T_coolant': [285],
    'cond.T_coolant': [285],

}

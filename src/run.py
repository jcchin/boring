






# ODE I/O
# self.promotes('interp_group', inputs=[('T_batt', 'T_{batt}')])
# self.promotes('interp_group', inputs=['SOC'])
# self.promotes('cell', inputs=[('U_Th', 'V_{thev}'), 
#                              ('I_pack', 'I_{batt}'),
#                              ('n_series','n_{series}'), 
#                              ('n_parallel', 'n_{parallel}'),
#                              ('Q_max','Q_{max}')])

# #promote outputs to match XDSM markdown spec
# self.promotes('cell', outputs=[('U_pack', 'V_{batt,actual}'),
#                                ('dXdt:U_Th', 'dXdt:V_{thev}'),
#                                ('dXdt:SOC', 'dXdt:SOC'),
#                                ('Q_pack', 'Q_{batt}')])





# static I/O
# {
#   "inputs": [
#     "weightFrac_{case}",
#     "energy_{required}",
#     "voltage_{nom,cell}",
#     "mass_{cell}",
#     "Q_{max}",
#     "I_{batt}",
#     "dischargeRate_{cell}",
#     "V_{batt}",
#     "voltage_{low,cell}",
#     "eta_{batt}"
#   ],
#   "outputs": [
#     "C_{p,batt}",
#     "n_{parallel}",
#     "n_{series}",
#     "mass_{battery}"
#   ]
# }
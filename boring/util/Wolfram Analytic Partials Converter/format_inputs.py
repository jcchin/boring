# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:16:05 2020

@author: klook
"""

#Autogenerate code for openMDAO inputs
#translate back from wolfram format to python format


import os
import pandas as pd
cwd = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import re
import glob
import sys

pd.set_option('display.max_colwidth',100) #increase max col width in terminal

f = cwd+'//testfile.txt'

df = pd.read_csv(f, header=None,squeeze=True)

df = df.str.replace(r'^\s*', '') #trim comments and whitespace
df = df.str.replace(r'#.*$', '')

print(df)

out_vars = df.str.extract(r'^(\S*)=',expand=False) #everything before '='

in_vars = df.str.extract(r'^.*=(.*)',expand=False) #everything after '='
in_vars = in_vars.str.replace(r'np.[a-zA-Z_]+', '') #remove np.somethings
in_vars = in_vars.str.findall(r'[a-zA-Z_]+') #find all input variables without numbers in their names

print(in_vars)

in_var_list = []
for row in in_vars:
    for var in row:
      in_var_list.append(var)  

in_var_list = list(dict.fromkeys(in_var_list)) # remove duplicates


in_var_list = ['self.add_input(\'{}\')'.format(in_var) for in_var in in_var_list]

# formula = "P = V*I*cos(2*pi*f)"
# variables = re.findall('[a-zA-Z_]+' , formula)
# special = ['cos', 'pi']
# variables = [v for v in variables if v not in special]  

out_vars.rename("output",inplace=True)
in_vars.rename("inputs",inplace=True)
allvars = pd.concat([out_vars,in_vars],axis=1)

x = []
for eqn in allvars.iterrows():    
     x.append('self.declare_partials(\'{}\', {})'.format(eqn[1]['output'], eqn[1]['inputs']))
    
    
out = pd.Series(in_var_list)
out = out.append(pd.Series(' '))
out = out.append('self.add_output(\''+out_vars+'\')',ignore_index=True)
out = out.append(pd.Series([' ',' ']))
#partials
out = out.append(pd.Series(x))

#convert to output lines instead of pd series to make things easeir
'''partials -> to wolfram alpha
    repalce all variables with single letters
    replace np.something with the equivalent (sqrt, pi)
    ** = ^
    
    wolfram -> python format
    Ex:
    (2 a h^2 Sqrt[2/Pi] Sqrt[1/(R T)] (1 - (P v)/(2 h)))/((2 - a) T v) + (a^2 h^2 Sqrt[2/Pi] Sqrt[1/(R T)] (1 - (P v)/(2 h)))/((2 - a)^2 T v)
    
    Sqrt[] = np.sqrt()
    spaces with no [+ -] symbols around mean *
    letters -> back into variables
    pi = np.pi
    
'''

out.to_csv(cwd+'\\outputfile.txt',index=False,header=False,)
print(out)



# self.declare_partials('A_w', ['XS:D_od', 'XS:t_w'])
#         self.declare_partials('A_wk', ['XS:D_od', 'XS:t_w', 'XS:D_v'])
#         self.declare_partials('A_interc', ['XS:D_v', 'L_cond'])
#         self.declare_partials('A_intere', ['XS:D_v', 'L_evap'])


# #Find the newest version of the response file based on the file name
# input_file = glob.glob(cwd+"/NASA GRC ML*.xlsx")

# newest = 0

# for f in input_file:
#     match = re.search(r'\(.*\-(\d+)\)', f)
#     ver = int(match.group(1))
#     if ver > newest:
#         newest = ver
#         file = f

# print("Found newest file: {} total responses\n{}".format(newest,file))   
# df = pd.read_excel (file)
# df = df.loc[:,['Name','Contact information (NASA email)','Do you have a PIV card?','Please select your NASA center']]
# df = df.rename(columns={'Contact information (NASA email)':'Email','Do you have a PIV card?':'PIV','Please select your NASA center':'Center'})
# cntr_dict ={"Glenn Research Center": "GRC"} 
# df = df.replace({'Center':cntr_dict})

# df = df[~df.isin(existing)].dropna()
# print("Processing Input: {}".format(os.path.basename(file)))

# if df.empty:
#     print('No new entries')
    
# else:
#     print('\n## {} New entries! :'.format(df.shape[0]))
#     print(df.to_string(index=False))



# df.to_csv('new_participants_{}.csv'.format(str(td)),index=False)
print('End')
# -*- coding: utf-8 -*-
"""
10/26/20

@author: klook
"""

#translate back from wolfram format to python format


import os
import pandas as pd
cwd = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import re
import glob
import sys

pd.set_option('display.max_colwidth',100) #increase max col width in terminal

f = cwd+'//wolfram_file.txt'

df = pd.read_csv(f, header=None,squeeze=True)

df = df.str.replace(r'^\s*', '') #trim comments and whitespace
df = df.str.replace(r'#.*$', '')



dictionary = pd.DataFrame()

dictionary['from'] = df.str.extract(r'^(\S*)\s*=',expand=False) #everything before '='
dictionary['to'] = df.str.extract(r'^.*=\s(\S*)',expand=False) #everything after '='

dictionary.dropna(inplace=True)
dictionary['to'].replace('^',' (',regex=True)

print(dictionary)

dic = dictionary.set_index('to').to_dict()['from']

dic["Sqrt\["] = "np.sqrt("

dic2 = {']':')',' - ':'-',' + ':'+','Pi':'np.pi'}


df = df[~df.str.contains('=')] #remove all dictionary rows

print(df)

for x in dic: #replace variables
    df = df.str.replace(' '+x,' '+dic[x])
    df = df.str.replace(x+' ',dic[x]+' ')    
    

for x in dic2: #replace symbols
    df = df.str.replace(x,dic2[x]) 
    
# after everything, all remaining spaces are multiplications
df = df.str.replace(' ','*')

print(df)

out = pd.Series(df)

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

out.to_csv(cwd+'\\wolfram_translated.txt',index=False,header=False,)
print(out)







print('End')
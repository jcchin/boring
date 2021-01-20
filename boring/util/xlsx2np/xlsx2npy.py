# -*- coding: utf-8 -*-
"""
# Overview
Script: xlsx2npy.py
Function: Converts files from xlsx to numpy array (.npy)

Created: 1/15/21

Author: Karsten Look (GRC-LTE0)

# Usage
1. Set input_file variable to xlsx file in ../inputs
2. Run script
3. Output file is saved with same name into ../outputs
Indexing: (Layer, Row, Column)

!NOTICE::
For some very strange reason, pandas will sometimes not read the xlsx file unless you open it, do not change anything, and save it again

# Assumptions
- All layers of the numpy array are of the same dimensions
- The elements are floats or ints
- The first column differentiates layers, but is not saved in output
- The second column is ignored (and also not saved in output)

# Compatibility
Anaconda environment file is provided in this folder as _env-spec-file.txt_ a human-friendly version is saved as _env_ref-spec-file.txt_
"""



import os
import pandas as pd
import numpy as np
from test2 import t_data # Verification data

cwd = os.path.dirname(os.path.realpath(__file__))

input_file = os.path.join(cwd,'inputs','test3.xlsx')

print("# xlsx2npy.py #\nConverting file \"{}\" to .npy\n".format(input_file))

df = pd.read_excel(input_file, engine='openpyxl',header=None)
# For backwards compatibility in case you have an old pandas version

first = True
for layer in df.iloc[:,0].unique():
    print("Adding layer: {}".format(layer))
    
    df_layer = df[df.iloc[:,0] == layer].iloc[:,2:]
    np_arr_layer = np.array(df_layer)
    
    if first:
        first = False #First layer gets added on to, this makes sure its the same shape
        np_arr = np_arr_layer
        np_arr = np.expand_dims(np_arr,2)
    else:
        np_arr = np.dstack((np_arr,np_arr_layer))

np_arr = np.moveaxis(np_arr,2,0) #Make the axes match verification data

#Verification test
diff = np_arr - t_data
maxdiff_loc = np.unravel_index(np.argmax(abs(diff)),(6,7,61))
maxdiff = np.max(abs(diff))
print("\n### Verification test (for test2.xlsx only) ### : The maximum difference is {}, at location {}".format(maxdiff,maxdiff_loc))


output_file = os.path.join(cwd,'outputs',os.path.splitext(os.path.basename(input_file))[0]+".npy")

np.save(output_file,np_arr) #Saves a binary numpy format
#open the file in another script with: arr = np.load("test3.npy")

print("\n.npy file saved to {}".format(output_file))


'''
References

"Python How to Import xlsx file using numpy"
https://stackoverflow.com/questions/29438631/python-how-to-import-xlsx-file-using-numpy

xlrd.biffh.XLRDError: Excel xlsx file; not supported [duplicate]
https://stackoverflow.com/questions/65254535/xlrd-biffh-xlrderror-excel-xlsx-file-not-supported


'''


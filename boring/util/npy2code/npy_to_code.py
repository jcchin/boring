# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:59:18 2021

@author: klook

Converts a 1D numpy array into a line of code with the equivalent array

Written by Karsten Look 
6/1/21

Usage:
1) run the validation script in spyder
2) in variable explorer, export the desired quantity to npy
3) run this script on the file (change "files" to filename.npy -> file = ["filename"])
4) copy and paste the resulting txt file into wherever you want the hardcoded value

tested on a 1500 entry 1D float64 numpy array

"""
import numpy as np
roundto = 13 #Round the values to this decimal when converting (for cases where values are off by eps)
files = ["q_sonic","q_vis"]


for filein in files:
    filein = filein+".npy"
    arr = np.load(filein)
    varname = filein.split(".")[0]
    with open("{}.txt".format(varname),"w") as file:
        file.write("{} = np.array([".format(varname))
        for idx in np.arange(0,arr.size-1):
            file.write("{}, ".format(round(arr[idx],roundto)))
        file.write("{}])\n".format(round(arr[arr.size-1],roundto))) #last value




# print(np.round(arr,roundto))

# compare = q_boiling
# a = np.array([1, 44, 2])
# print(np.array_equal(compare,np.round(arr,roundto)))
# print(max(abs(compare-arr)))
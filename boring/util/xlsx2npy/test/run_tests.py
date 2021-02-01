# -*- coding: utf-8 -*-
"""
xlsx2npy test file

Created on Mon Feb  1 11:24:11 2021

@author: klook
"""

def two_col_test():
    np_arr = np_arr_    
    #Verification test
    diff = np_arr - t_data
    maxdiff_loc = np.unravel_index(np.argmax(abs(diff)),(6,7,61))
    maxdiff = np.max(abs(diff))
    print("\n### Verification test (for test2.xlsx only) ### : The maximum difference is {}, at location {}".format(maxdiff,maxdiff_loc))

def 
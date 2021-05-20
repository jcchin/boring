# -*- coding: utf-8 -*-
"""
xlsx2npy test file

Created on Mon Feb  1 11:24:11 2021

@author: klook

    # test if missing parameter values save as NaN
    # test against 4 parameters
    # test against two separate parameter files


"""

import unittest
import numpy as np

class TestXlsx2npy(unittest.TestCase):
    def setUp(self):
        # Nothing to add
        print("Setup")
        
    def test_two_col(self): # test against 2 parameters
        np_arr = np_arr_    
        #Verification test
        diff = np_arr - t_data
        maxdiff_loc = np.unravel_index(np.argmax(abs(diff)),(6,7,61))
        maxdiff = np.max(abs(diff))
        print("\n### Verification test (for test2.xlsx only) ### : The maximum difference is {}, at location {}".format(maxdiff,maxdiff_loc))

if __name__ == '__main__':
    unittest.main()
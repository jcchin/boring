# -*- coding: utf-8 -*-
"""
# Overview
Script: xlsx2npy.py
Function: Converts files from xlsx to numpy array (.npy)

Author: Karsten Look (GRC-LTE0)

# Usage
    Will search for all .xlsx files in the ./inputs folder (nonrecursive), ignoring everything else 
    Assumes all files in the input folder are of the same search grid
    Assumes files contain only numerical data, and there is no header or blank rows at top
    
    Load order and file name scheme does not matter, all content will be sorted by the first column first, 
      then the second, and so on for all given parameters
      
    output file is put into outputs folder, named like <name of first input file>_<number of files>-combined.npy"


    Example mapping file:
        index value : entry value
        
        human readable:
        # index 1
        0 : 1
        1 : 1.1
        2 : 1.2
        ...

        # index 2
        0 : 0.5
        1 : 0.75
        ...
        
        # index 3
        0 : 5
        1 : 6
        
        saved a a pickled list
        [dict for idx 1, dict for idx 2, ....]
                
        ...

# Example Output Usage

arr = np.load("output_filename.npy")

!NOTICE::
For some very strange reason, pandas will sometimes not read the xlsx file unless you open it, do not change anything, and save it again

# Assumptions
- All layers of the numpy array are of the same dimensions
- The elements are floats or ints
- The first column differentiates layers, but is not saved in output
- The second column is ignored (and also not saved in output)

# Compatibility
Anaconda environment file is provided in this folder as _env-spec-file.txt_ a human-friendly version is saved as _env_ref-spec-file.txt_

    
===
Further Development
Backlog:
    + Move verification testing into a git automated panel of tests

Working:
    

Done:
    + Add feature extensibility for more parameter columns
    + Condense time series to only max value
    + Print column statistics (how many geometric parameters)
    + Add batching for many files -> common output file

"""

import os, sys
import pandas as pd
import numpy as np
import glob
import pickle
# from test2 import t_data # Verification data

### Settings ###

num_param = 3 #Required because there is no way to programatically differentiate 
# between parameters and timeseries portions

# If batch processing, export each to its own output file or merge all into one file?
#merge = False - To be added later

################

cwd = os.path.dirname(os.path.realpath(__file__))

def getFiles():
    
    files = glob.glob(os.path.join(cwd,'inputs','mass_hny_hole_h100.xlsx'))
    
    
    #Check there are >0 files, and that they are all xlsx
    if not len(files):
        sys.exit("xlsx2npy: Import error; number of input files cannot be zero")
    
    for file in files: 
        file_extention = os.path.splitext(file)[1]
        if  file_extention != '.xlsx':
            sys.exit("xlsx2npy: Import error; input file type {} not supported".format(file_extention)) #Should never happen, but just in case
    
    return files

def loadFiles(files_):
    files = files_
    
    if len(files) == 1:
        print("# xlsx2npy.py #\nFound one file, converting to .npy:\n{}\n".format(files[0]))
    
    else:   
        print("Found {} files".format(len(files)))
        for file in files:
            print(file)
            
    df = pd.DataFrame()
    
    for file in files:
        df = df.append(pd.read_excel(file, engine='openpyxl',header=None))
    
    
    df = df.sort_values([_ for _ in range(num_param)],'index') # Sorts by first col first, second second, etc.
    df = df.reset_index(drop=True)
    
    
    print("\nMerged files\nNumber of param given: {}\nTimeseries entries: {}\nNumber of cases: {}".format(num_param,df.shape[1]-num_param,df.shape[0]))
    
    df_raw = df.copy() #For debugging to inspect sorted values
    
    df.iloc[:,num_param] = df.iloc[:,num_param:].max(axis=1)
    df.drop(columns=[_ for _ in range(num_param+1,df.shape[1])],inplace=True)
    
    return [df, df_raw]


def indexRows(df_):
    df = df_
    
    mapping = []
    invmapping = []
    idx_df = df.copy()
    
    for col in range(num_param):
        
        values = df.iloc[:,col].unique()
        keys = range(len(values))
        
        mapping.append(dict(zip(keys, values))) 
        invmapping.append(dict(zip(values,keys)))
        
        idx_df.iloc[:,col] = df.iloc[:,col].transform(lambda x: invmapping[col][x])
        
    return [mapping, idx_df]

def df2npy(idx_df_,mapping_):
    idx_df = idx_df_
    mapping = mapping_

    npsize = []
    for dim in mapping:
        npsize.append(len(dim))
    
    np_arr = np.empty(npsize)
    
    for row in idx_df.index:
        coordinates = tuple(idx_df.iloc[row,:num_param].astype('int').values)
        # print(coordinates)
        np_arr[coordinates] = idx_df.iloc[row,num_param]    
    
    return np_arr

def export(name_, np_arr_):
    np.save(name_,np_arr_) #Save to numpy binary 
    
    #open the file in another script with: arr = np.load("test3.npy")
    print("\n.npy file saved to {}".format(name_))


# move to git test
# def veriftest2(np_arr_):
#     np_arr = np_arr_    
#     #Verification test
#     diff = np_arr - t_data
#     maxdiff_loc = np.unravel_index(np.argmax(abs(diff)),(6,7,61))
#     maxdiff = np.max(abs(diff))
#     print("\n### Verification test (for test2.xlsx only) ### : The maximum difference is {}, at location {}".format(maxdiff,maxdiff_loc))


    
### Main ###
files = getFiles()
[df, df_raw] = loadFiles(files)
[mapping, idx_df] = indexRows(df)
np_arr = df2npy(idx_df,mapping)

first_file = os.path.split(files[0])[1]
outfile = os.path.join(cwd,'outputs',os.path.splitext(first_file)[0]+'_{}.npy'.format(len(files)))
export(outfile,np_arr)

mapfilename = os.path.join(cwd,'outputs',os.path.splitext(first_file)[0]+'_{}.pickle'.format(len(files)))

with open(mapfilename, "wb") as mapfile:
    pickle.dump(mapping, mapfile)

print("Saved mapping file to {}".format(mapfilename))
    # load the mapping file into another script with:
    # with open("test.txt", "rb") as fp:   # Unpickling
    # ...   b = pickle.load(fp)      
    


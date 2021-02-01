# -*- coding: utf-8 -*-
"""
# Overview
Script: xlsx2npy.py
Function: Converts files from xlsx to numpy array (.npy)

Created: 1/15/21
Author: Karsten Look (GRC-LTE0)

# Usage
    + FIRST enter the number of parameter columns into the variable num_param in this script
    + Place desired files in input folder
    + Run script
    + Check output folder

### !NOTICE ###::
    There is a known bug in openpyxl that in some cases pandas fails to import xlsx files due to a change in the xlsx file format in 2019
    Workaround: open the file, do not change anything, and save it again. Script should then finish successfully

    More info:
        https://stackoverflow.com/questions/64126185/openpyxl-cant-read-xlsx-file-but-if-i-save-the-file-it-opens
        https://foss.heptapod.net/openpyxl/openpyxl/-/issues/1071

# Details
    Will search for all .xlsx files in the ./inputs folder (nonrecursive), ignoring everything else 
    
    Load order and file name scheme does not matter, all content will be sorted by the first column first, 
      then the second, and so on for all given parameters
      
    output file is put into ./outputs folder, named like <name of first input file>_<number of files>-combined.npy"
    
    outputs are:
        1. A .npy file (the pickled numpy array)
        2. A .pickle file (a python list of dict vars) that translates indices to the given input parameter values


    Example mapping file:
        index value : parameter value
        
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
        
        saved as a pickled list
        [dict for idx 1, dict for idx 2, ....]
                
        ...

# Example Output Usage
    import numpy as np
    arr = np.load("output_filename.npy")

    with open("output_filename_mapping.pickle", "rb") as file:   # Unpickling
        mapping = pickle.load(file) 


# Assumptions
    - If one file of a set fails to import, all files will similarly fail
    
    - All files in the input folder cover the same parameter sweep set
        Missing parameter values are saved as NaN
    
    - Files contain only numerical data, and there is no header or blank rows at top
    
    - The user correctly identifies the number of parametersr in the input file

# Compatibility
    Anaconda environment file is provided in ./envr folder. A human-friendly version is also provided

    
===
Further Development

Backlog (May or may not implement):
    + Add check - Does the number of params seems right? (How close are values
  last parameter to the values in the first time series relative to the time series columns)
    + Find fix to openpyxl excel read errors -> Other library, auto open and save, etc.
    + add verification that imported files contain data (if multiple sets and only some fail)
    + Move verification testing into a git automated panel of tests

Planned:    
 
In Progress:
    test if missing parameter values save as NaN
    test against 2 parameters
    test against 4 parameters
    test against two separate parameter files
    
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

num_param = 4 # How many parameter columns are there?
# Required because there is no way for script to differentiate between parameters and timeseries portions

################


cwd = os.path.dirname(os.path.realpath(__file__))

def getFiles():
    
    files = glob.glob(os.path.join(cwd,'inputs','*.xlsx'))
    
    
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
    
    if(df.shape == (1,1)):
        sys.exit("xlsx2npy: Import error; no data returned from file")
        
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
    print("\n.npy file saved to {}".format(name_))


### Main ###
files = getFiles()
[df, df_raw] = loadFiles(files)
[mapping, idx_df] = indexRows(df)
np_arr = df2npy(idx_df,mapping)

first_file = os.path.split(files[0])[1]
outfile = os.path.join(cwd,'outputs',os.path.splitext(first_file)[0]+'_{}-combined.npy'.format(len(files)))
export(outfile,np_arr)


mapfilename = os.path.join(cwd,'outputs',os.path.splitext(first_file)[0]+'_{}-combined_mapping.pickle'.format(len(files)))

with open(mapfilename, "wb") as mapfile:
    pickle.dump(mapping, mapfile)

print("Saved mapping file to {}".format(mapfilename))
         
    


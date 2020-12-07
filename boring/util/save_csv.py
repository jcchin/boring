import pandas as pd
import numpy as np
import datetime

"""
Author: Jeff Chin
"""


def save_csv(prob, sim, file_name='output.csv',
             traj='traj', phase='phase',
             x_name='time', x_units='s', y_name='', y_units=''):
    ''' Save Time History to CSV file for convenient portabilitiy

    prob is the problem
    sim is the explicit simulation 

    optionally take in case recorder prob/sim

    y_name should be an array of variables
    y_units should be an array of corresponding units (not yet implemented)


    Usage:

    from boring.util.save_csv import save_csv
    save_csv(prob, sim, 'output.csv', 
             y_name = ['h', 'r', 'CAS', 'SOC', 'T_motor'], 
             y_units= ['m', 'km','m/s',  None, 'degC'   ])

    '''
    print(f'Saving CSV {file_name}...')

    varDict = {}

    try:
        t = prob.get_val(f'{traj}.{phase}.timeseries.time')
        print('Implicit Save')
    except:
        print('Fail Implicit Save')
        print(f'{traj}.{phase}.timeseries.time')

    try:
        t_s = sim.get_val(f'{traj}.{phase}.timeseries.time')
        print('Explicit Save')
    except:
        print('Fail Explicit Save')

    varDict.update({'t': np.concatenate(t).ravel().tolist()})

    for name in y_name:
        d = {}
        d2 = {}  # TMS system has a different size
        try:
            y = prob.get_val(f'{traj}.{phase}.timeseries.{name}')
            d[f"{name}"] = np.concatenate(y).ravel().tolist()
            print(f'Saving problem: {name} ...')
            varDict.update(d)
        except:
            print(f'Unable to save: {name} ...')

    df = pd.DataFrame(varDict)
    df = df.set_index(['t'])
    df.index = pd.to_datetime(df.index, unit='s')
    df = df[~df.index.duplicated()]  # remove duplicate timestamps
    df = df.resample('1s').bfill(limit=1).interpolate()  # resample every 5 sec
    deltaT = df.index.to_series() - datetime.datetime(1970, 1, 1)  # calculate timedelta
    df.index = deltaT.dt.total_seconds()  # convert index back to elapsed seconds
    df['t'] = df.index  # make explicit column with index

    df.to_pickle('./output.pkl')
    df.to_csv(file_name, index=False)

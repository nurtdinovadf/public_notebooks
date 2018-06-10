import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

ex21 = pd.read_csv('DS_ex2_04.18.csv', 
                   parse_dates = ['DATE_OF_BIRTH'], 
                   header = None, 
                   quoting=3, 
                   names = ['SUBS_ID', 'STATUS', 'DATE_OF_BIRTH', 'GNDR_ID', 'COUNT(PT.PAY_ID)', 'PAYMENT', 'CHARGE', 'UNKNOWN1', 'UNKNOWN2'], 
                   skiprows = [0])
ex22 = pd.read_csv('DS_ex2_callcenter_04.18.csv')
ex23 = pd.read_csv('DS_ex2_os_04.18.csv')

ex23 = pd.concat([ex23, pd.get_dummies(ex23.OS)], axis = 1)[['SUBS_ID', 'android', 'ios', 'web']].groupby(by = ['SUBS_ID']).sum().reset_index(drop = False)

ex21 = ex21.merge(ex22, how = 'left', on = ['SUBS_ID'])
ex21 = ex21.merge(ex23, how = 'left', on = ['SUBS_ID'])
ex21['oses'] = ex21[['android', 'ios', 'web']].sum(axis = 1)
ex21['year'] = ex21.DATE_OF_BIRTH.dt.year
ex21['month'] = ex21.DATE_OF_BIRTH.dt.month
ex21['day'] = ex21.DATE_OF_BIRTH.dt.day
ex21['woy'] = ex21.DATE_OF_BIRTH.dt.week
ex21['dow'] = ex21.DATE_OF_BIRTH.dt.weekday

encoder = LabelEncoder()
ex21['status_encoded'] = encoder.fit_transform(ex21.STATUS)

columns = list(ex21.columns)
for col in ['STATUS', 'DATE_OF_BIRTH']:
    columns.remove(col)
    
ex21[columns] = ex21[columns].fillna(-1)

ex21.to_feather('ex21.feather')
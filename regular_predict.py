import pickle
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)

with open('data/Metro_train/Metro_train/record_all.pkl', 'rb') as f:
    record_all = pickle.load(f)

df_29 = pd.read_csv('data/Metro_testA/Metro_testA/testA_submit_2019-01-29.csv')

df_29['hms'] = [x.strftime('%H-%M-%S') for x in df_29['startTime'].astype('datetime64[ns]')]
df_29 = pd.merge(left=df_29, right=record_all, left_on=['stationID', 'hms'], right_on=['station', 'hms'],
                 how='left').fillna(0)

# base data
df_29['inNums'], df_29['outNums'] = df_29['2019-01-28_inNums'], df_29['2019-01-28_outNums']

# rule
df_29[df_29['startTime'] < '2019-01-29 00:10:00']

print(df_29[df_29['startTime'] < '2019-01-29 00:10:00'])

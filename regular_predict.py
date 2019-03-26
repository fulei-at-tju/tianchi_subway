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

"""rule 周一周二之差最小的两天差距均值"""
Monday = ['2019-01-07', '2019-01-14', '2019-01-21']
Tuesday = ['2019-01-08', '2019-01-15', '2019-01-22']

for m, t in zip(Monday, Tuesday):
    df_29[m + '_diff_in'] = df_29[m + '_inNums'] - df_29[t + '_inNums']
    df_29[m + '_diff_out'] = df_29[m + '_outNums'] - df_29[t + '_outNums']


def diff_mean(row, in_out):
    diff_min = 999
    re = 0
    for m1 in Monday:
        for m2 in Monday:
            if m1 == m2:
                continue
            diff = abs(row[m1 + in_out] - row[m2 + in_out])
            if diff < diff_min:
                diff_min = diff
                re = (row[m1 + in_out] + row[m2 + in_out]) / 2
    return re


df_29['inNums'] = df_29['inNums'] - df_29.apply(diff_mean, axis=1, args=('_diff_in',))
df_29['outNums'] = df_29['outNums'] - df_29.apply(diff_mean, axis=1, args=('_diff_out',))

"""rule 5:20前及22:50后无人进站"""
flag = (df_29['startTime'] <= '2019-01-29 05:20:00') | (df_29['startTime'] >= '2019-01-29 22:50:00')
df_29.loc[flag, 'inNums'] = 0

"""答案"""
ans = df_29.loc[:, ['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].fillna(0)
ans.to_csv('data/result/ans_rule_0326.csv', index=False)

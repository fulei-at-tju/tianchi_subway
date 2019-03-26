import pickle
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVR

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
time_str = time.strftime('%m.%d-%H-%M-%S', time.localtime(time.time()))

with open('data/Metro_train/Metro_train/record_all.pkl', 'rb') as f:
    record_all = pickle.load(f)

df_29 = pd.read_csv('data/Metro_testA/Metro_testA/testA_submit_2019-01-29.csv')

df_29['hms'] = [x.strftime('%H-%M-%S') for x in df_29['startTime'].astype('datetime64[ns]')]
df_29 = pd.merge(left=df_29, right=record_all, left_on=['stationID', 'hms'], right_on=['station', 'hms'],
                 how='left').fillna(0)

# base data
df_29['inNums'], df_29['outNums'] = df_29['2019-01-28_inNums'], df_29['2019-01-28_outNums']

"""rule 利用2019-01-28日数据预测2019-01-29日数据。所以，计算历史上周一数据与周二数据的差值
   mean(d_monday - d_tuesday) = d_2019-01-28 - d_2019-01-29
"""
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


def diff_trend(row, in_out):
    """
    计算周二与周一之间的流量差距，并预测差距
    """
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

    X = [[i] for i in range(len(Monday))]
    Y = [row[Monday[i] + in_out] for i in range(len(Monday))]
    regr = LinearRegression()
    regr.fit(X, Y)
    r2 = r2_score(regr.predict(X), Y)

    if r2 > 0.8:
        svr = SVR(gamma=.1, C=100)
        svr.fit(X, Y)
        re = svr.predict([[len(Monday)]])[0]
        if abs(re) > 20:
            re *= .8
        if abs(re) > 50:
            re *= .6
        if abs(re) > 100:
            re *= .5

    return re


# df_29['inNums'] = df_29['inNums'] - df_29.apply(diff_mean, axis=1, args=('_diff_in',))
# df_29['outNums'] = df_29['outNums'] - df_29.apply(diff_mean, axis=1, args=('_diff_out',))


df_29['inNums'] = df_29['inNums'] - df_29.apply(diff_trend, axis=1, args=('_diff_in',))
df_29['outNums'] = df_29['outNums'] - df_29.apply(diff_trend, axis=1, args=('_diff_out',))

"""rule 5:20前及22:50后无人进站"""
flag = (df_29['startTime'] <= '2019-01-29 05:20:00') | (df_29['startTime'] >= '2019-01-29 22:50:00')
df_29.loc[flag, 'inNums'] = 0

"""rule 小于0数据置为0"""
flag = df_29['inNums'] < 0
df_29.loc[flag, 'inNums'] = 0

flag = df_29['outNums'] < 0
df_29.loc[flag, 'outNums'] = 0

"""rule 如果某时间段内出入站人数为0，判断其前后进出站人数是否为0,不为0则插值"""
df_29['inNums_-1'], df_29['inNums_1'] = df_29.shift(-1)['inNums'], df_29.shift(1)['inNums']
df_29['outNums_-1'], df_29['outNums_1'] = df_29.shift(-1)['outNums'], df_29.shift(1)['outNums']

flag = df_29['inNums'] == 0
df_29.loc[flag, 'inNums'] = (df_29.loc[flag, 'inNums_-1'] + df_29.loc[flag, 'inNums_1']) / 2

flag = df_29['outNums'] == 0
df_29.loc[flag, 'outNums'] = (df_29.loc[flag, 'outNums_-1'] + df_29.loc[flag, 'outNums_1']) / 2

"""答案"""
ans = df_29.loc[:, ['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].fillna(0)
ans['inNums'] = ans['inNums'].astype('int')
ans['outNums'] = ans['outNums'].astype('int')
ans.to_csv('data/result/ans_rule_{}.csv'.format(time_str), index=False)

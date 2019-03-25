import pickle

import pandas as pd
import numpy as np

# 日期时间
datetime = pd.date_range('2019-01-01', '2019-01-25')
dt = [dt.strftime("%Y-%m-%d") for dt in datetime]
record = []

# 从csv读取
for dt in [dt.strftime("%Y-%m-%d") for dt in datetime]:
    print(dt)
    df = pd.read_csv('data/Metro_train/Metro_train/record_{}.csv'.format(dt))
    record.append(df)
    with open('data/Metro_train/Metro_train/record_df_{}.pkl'.format(dt), 'wb') as f:
        pickle.dump(df, f)

# 从持久化读
for dt in [dt.strftime("%Y-%m-%d") for dt in datetime]:
    print(dt)
    with open('data/Metro_train/Metro_train/record_df_{}.pkl'.format(dt), 'rb') as f:
        record.append(pickle.load(f))

with open('data/Metro_train/Metro_train/record_df_{}.pkl'.format(dt[1]), 'rb') as f:
    df0102 = pickle.load(f)

# 读测试数据
df = pd.read_csv('D:\\workspace\\tianchi_subway\\data\\Metro_testA\\Metro_testA\\testA_record_2019-01-28.csv')




# 十分钟标签
# for tuples, group in dfg:
#     print(tuples)
#     station, status = tuples
#     tmp = group.set_index('time')
#     tmp.index = tmp.index.astype('datetime64[ns]')
#     tmp = tmp.resample('10Min').count()
#     tmp2 = pd.DataFrame({'station': station, 'status': status, 'num': list(tmp.payType), 'time': list(tmp.index)})
#     dfl.append(tmp2)
#
# # 统计
# sdf = pd.concat(dfl)
# sdfg = sdf.groupby(['station', 'time', 'status']).sum().reset_index()
# sdfg0 = sdfg[sdfg.status == 0]
# sdfg1 = sdfg[sdfg.status == 1]
# sdfg2 = pd.merge(left=sdfg0, right=sdfg1, on=['station', 'time'])
# sdfg2['stationID'] = sdfg2['station']
# sdfg2['hm'] = [x.strftime('%H-%M-%S') for x in sdfg2['time']]

# 形成结果
submit = df = pd.read_csv('data/Metro_testA/Metro_testA/testA_submit_2019-01-29.csv')
submit['startTime'] = submit['startTime'].astype('datetime64[ns]')
submit['hm'] = [x.strftime('%H-%M-%S') for x in submit['startTime']]
s2 = pd.merge(left=submit, right=sdfg2, on=['stationID', 'hm'], how='outer')
s2['inNums'] = s2['num_y']
s2['outNums'] = s2['num_x']
s2 = s2.loc[:, ['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].fillna(0)
s2 = s2.sort_values(by=['stationID', 'startTime'])
s2['inNums'] = s2['inNums'].astype('int')
s2['outNums'] = s2['outNums'].astype('int')
s2['startTime'] = [x.strftime('%Y-%m-%d %H:%M:%S') for x in s2['startTime']]
s2.to_csv('data/Metro_testA/Metro_testA/ans.csv', index=False)

# 生成测试数据结果
df_test = pd.read_csv('data/Metro_testA/Metro_testA/testA_record_2019-01-28.csv')
dfg = df0102.groupby(['stationID', 'status'])
dfl = []
for tuples, group in dfg:
    print(tuples)
    station, status = tuples
    tmp = group.set_index('time')
    tmp.index = tmp.index.astype('datetime64[ns]')
    tmp = tmp.resample('10Min').count()
    tmp2 = pd.DataFrame({'station': station, 'status': status, 'num': list(tmp.payType), 'time': list(tmp.index)})
    dfl.append(tmp2)
sdf = pd.concat(dfl)
sdfg = sdf.groupby(['station', 'time', 'status']).sum().reset_index()
sdfg0 = sdfg[sdfg.status == 0]
sdfg1 = sdfg[sdfg.status == 1]
sdfg2 = pd.merge(left=sdfg0, right=sdfg1, on=['station', 'time'], how='outer')
sdfg2['hm'] = [x.strftime('%H-%M-%S') for x in sdfg2['time']]
sdfg2 = sdfg2.fillna(0).sort_values(by=['station', 'hm'])
sdfg2['inNums'] = sdfg2['num_y'].astype('int')
sdfg2['outNums'] = sdfg2['num_x'].astype('int')

sdfg2 = sdfg2.loc[:, ['station', 'time', 'hm', 'inNums', 'outNums']]

with open('data/Metro_testA/Metro_testA/testA.pkl', 'wb') as f:
    pickle.dump(sdfg2, f)

with open('data/Metro_testA/Metro_testA/testA.pkl', 'rb') as f:
    testA = pickle.load(f)

pre = pd.read_csv('data/Metro_testA/Metro_testA/ans.csv')


def mae(y_predict, y_test):
    return np.sum(np.absolute(y_predict - y_test)) / len(y_test)


in_mae = mae(np.array(pre.inNums), np.array(testA.inNums))
out_mae = mae(np.array(pre.outNums), np.array(testA.outNums))


def sample_10min(_df):
    dfg = _df.groupby(['stationID', 'status'])
    dfl = []

    # 十分钟标签
    for tuples, group in dfg:
        print(tuples)
        station, status = tuples
        tmp = group.set_index('time')
        tmp.index = tmp.index.astype('datetime64[ns]')
        tmp = tmp.resample('10Min').count()
        tmp2 = pd.DataFrame({'station': station, 'status': status, 'num': list(tmp.payType), 'time': list(tmp.index)})
        dfl.append(tmp2)

    # 统计
    sdf = pd.concat(dfl)
    sdfg = sdf.groupby(['station', 'time', 'status']).sum().reset_index()
    sdfg0 = sdfg[sdfg.status == 0]
    sdfg1 = sdfg[sdfg.status == 1]
    sdfg2 = pd.merge(left=sdfg0, right=sdfg1, on=['station', 'time'])
    sdfg2['stationID'] = sdfg2['station']
    sdfg2['hm'] = [x.strftime('%H-%M-%S') for x in sdfg2['time']]
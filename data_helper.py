import pickle
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# 日期时间
datetime = pd.date_range('2019-01-01', '2019-01-25')
dt = [dt.strftime("%Y-%m-%d") for dt in datetime]
record = []


def model_saver(model, name):
    with open('data/{}.pkl'.format(name), 'wb') as f:
        pickle.dump(model, f)


def model_loader(name):
    with open('data/{}.pkl'.format(name), 'rb') as f:
        morel = pickle.load(f)
    return morel


def test_data_loader():
    """
    加载29日数据，用于预测
    :return:
    """
    with open('data/Metro_train/Metro_train/test_29.pkl', 'rb') as f:
        df_29 = pickle.load(f)
    test_x = [[row.station, row.weekday, row.hour, row.minute, row.holiday] for index, row in df_29.iterrows()]

    return test_x


def make_ans(in_nums, out_nums, file_name):
    """
    将结果写入csv
    :param in_nums: 列表 或 ndarray
    :param out_nums: 列表 或 ndarray
    :param file_name:
    :return:
    """
    df_29 = pd.read_csv('data/Metro_testA/Metro_testA/testA_submit_2019-01-29.csv')
    df_29['inNums'] = in_nums
    df_29['outNums'] = out_nums
    df_29.to_csv('data/result/{file_name}.csv'.format(file_name=file_name), index=False)


def data_loader():
    """
    加载训练数据
    :return:
    """
    with open('data/Metro_train/Metro_train/train_all.pkl', 'rb') as f:
        train = pickle.load(f)
    with open('data/Metro_train/Metro_train/test_28.pkl', 'rb') as f:
        test = pickle.load(f)

    train_x = [[row.station, row.weekday, row.hour, row.minute, row.holiday] for index, row in train.iterrows()]
    train_y = [[row.inNums, row.outNums] for index, row in train.iterrows()]

    test_x = [[row.station, row.weekday, row.hour, row.minute, row.holiday] for index, row in test.iterrows()]
    test_y = [[row.inNums, row.outNums] for index, row in test.iterrows()]
    return train_x, train_y, test_x, test_y


def data_loader2():
    with open('data/Metro_train/Metro_train/baseline_train.csv', 'rb') as f:
        data = pickle.load(f)
    all_columns = [f for f in data.columns if f not in ['weekend', 'inNums', 'outNums']]

    train = data[(data.day != 29) & (data.day != 28)]
    train_x = train[all_columns].values
    train_y = train[['inNums', 'outNums']].values

    test = data[data.day == 29]
    test_x = test[all_columns].values
    test_y = test[['inNums', 'outNums']].values

    return train_x, train_y, test_x, test_y


def csv2df():
    """
    [ 预处理 ]原始csv文件存储为dataframe
    :return:
    """
    for _dt in dt:
        print(_dt)
        df = pd.read_csv('data/Metro_train/Metro_train/record_{}.csv'.format(_dt))
        record.append(df)
        with open('data/Metro_train/Metro_train/record_df_{}.pkl'.format(_dt), 'wb') as f:
            pickle.dump(df, f)


def mae(y_predict, y_test):
    """
    计算mae
    :param y_predict:
    :param y_test:
    :return:
    """
    return np.sum(np.absolute(y_predict - y_test)) / len(y_test)


def sample_10min(_df):
    """
    [ 预处理 ]读取原始df 形成10分钟统计结果，数据预处理
    :param _df:
    :return:
    """
    dfg = _df.groupby(['stationID', 'status'])
    dfl = []

    # 十分钟标签
    for tuples, group in dfg:
        station, status = tuples
        tmp = group.set_index('time')
        tmp.index = tmp.index.astype('datetime64[ns]')
        tmp = tmp.resample('10Min').count()
        tmp2 = pd.DataFrame({'station': station, 'status': status, 'num': list(tmp.payType), 'time': list(tmp.index)})
        dfl.append(tmp2)

    sdf = pd.concat(dfl)
    sdfg = sdf.groupby(['station', 'time', 'status']).sum().reset_index()
    sdfg0, sdfg1 = sdfg[sdfg.status == 0], sdfg[sdfg.status == 1]
    sdfg2 = pd.merge(left=sdfg0, right=sdfg1, on=['station', 'time'])
    sdfg2['inNums'], sdfg2['outNums'], sdfg2['hms'], sdfg2['stationID'] = sdfg2['num_y'].astype('int'), sdfg2[
        'num_x'].astype('int'), [x.strftime('%H-%M-%S') for x in sdfg2['time']], sdfg2['station']
    return sdfg2.loc[:, ['station', 'time', 'hms', 'inNums', 'outNums']]


def all_sample_10min():
    """
    [ 预处理 ]所有日期数据，转化成10分钟统计数据
    :return:
    """
    for _dt in dt:
        print(_dt)
        with open('data/Metro_train/Metro_train/record_df_{}.pkl'.format(_dt), 'rb') as f:
            df = pickle.load(f)
        df2 = sample_10min(df)
        with open('data/Metro_train/Metro_train/record_df_10min_{}.pkl'.format(_dt), 'wb') as f:
            pickle.dump(df2, f)


def test_29_data():
    """
    29日csv数据形成训练数据
    :return:
    """

    def holiday(time):
        if time.weekday() in [5, 6]:
            return 1
        return -1

    df_29 = pd.read_csv('data/Metro_testA/Metro_testA/testA_submit_2019-01-29.csv')
    df_29['time'] = df_29['startTime'].astype('datetime64[ns]')
    df_29['weekday'] = [x.weekday() for x in df_29['time']]
    df_29['hour'] = [x.hour for x in df_29['time']]
    df_29['minute'] = [x.hour * 60 + x.minute for x in df_29['time']]
    df_29['holiday'] = [holiday(x) for x in df_29['time']]
    df_29['station'] = df_29['stationID']

    with open('data/Metro_train/Metro_train/test_29.pkl', 'wb') as f:
        pickle.dump(df_29, f)


def concat_10min():
    """
    [ 数据预处理 ]将10min聚合数据，聚合成单一文件，行
    :return:
    """
    dt_list = []
    for _dt in dt:
        with open('data/Metro_train/Metro_train/record_df_10min_{}.pkl'.format(_dt), 'rb') as f:
            df = pickle.load(f)
            df[_dt + '_inNums'], df[_dt + '_outNums'] = df['inNums'], df['outNums']
            tmp = df.loc[:, ['station', 'hms', _dt + '_inNums', _dt + '_outNums']]
            dt_list.append(tmp)

    df_28 = pd.read_csv('data/Metro_testA/Metro_testA/testA_record_2019-01-28.csv')
    df_28 = sample_10min(df_28)
    _dt = '2019-01-28'
    df_28[_dt + '_inNums'], df_28[_dt + '_outNums'] = df_28['inNums'], df_28['outNums']
    tmp = df_28.loc[:, ['station', 'hms', _dt + '_inNums', _dt + '_outNums']]
    dt_list.append(tmp)
    df = dt_list[0]
    for index in range(1, len(dt_list)):
        df = pd.merge(left=df, right=dt_list[index], on=['station', 'hms'], how='outer')

    df = df.sort_values(by=['station', 'hms']).fillna(0)
    df.to_excel('data/Metro_train/Metro_train/record_all.xls', index=False)
    with open('data/Metro_train/Metro_train/record_all.pkl', 'wb') as f:
        pickle.dump(df, f)
    return df


def concat_10min_train_test_data():
    """
        [ 数据预处理 ]将10min聚合数据，聚合成单一文件，列
        特征：站点编号，星期几，小时，当天分钟数，是否是节假日
        去除 2019-01-01
        :return:
    """

    def holiday(time):
        if time.weekday() in [5, 6]:
            return 1
        return -1

    # 训练数据01-25日
    dt_list = []
    for _dt in dt:
        if _dt == '2019-01-01':
            continue
        print(_dt)
        with open('data/Metro_train/Metro_train/record_df_10min_{}.pkl'.format(_dt), 'rb') as f:
            df = pickle.load(f)
            df['weekday'] = [x.weekday() for x in df['time']]
            df['hour'] = [x.hour for x in df['time']]
            df['minute'] = [x.hour * 60 + x.minute for x in df['time']]
            df['holiday'] = [holiday(x) for x in df['time']]

            tmp = df.loc[:, ['station', 'weekday', 'hour', 'minute', 'holiday', 'inNums', 'outNums']]
            dt_list.append(tmp)

    df_train = pd.concat(dt_list)
    df_train.to_csv('data/Metro_train/Metro_train/train_all.csv', index=False)
    with open('data/Metro_train/Metro_train/train_all.pkl', 'wb') as f:
        pickle.dump(df_train, f)

    # 测试A 28日
    df = pd.read_csv('data/Metro_testA/Metro_testA/testA_record_2019-01-28.csv')
    df = sample_10min(df)
    df['weekday'] = [x.weekday() for x in df['time']]
    df['hour'] = [x.hour for x in df['time']]
    df['minute'] = [x.hour * 60 + x.minute for x in df['time']]
    df['holiday'] = [holiday(x) for x in df['time']]

    df_28 = df.loc[:, ['station', 'weekday', 'hour', 'minute', 'holiday', 'inNums', 'outNums']]

    df_28.to_csv('data/Metro_train/Metro_train/test_28.csv', index=False)
    with open('data/Metro_train/Metro_train/test_28.pkl', 'wb') as f:
        pickle.dump(df_28, f)


def form_ans(df_pre):
    df_29 = pd.read_csv('data/Metro_testA/Metro_testA/testA_submit_2019-01-29.csv')
    df_29['startTime'] = df_29['startTime'].astype('datetime64[ns]')
    df_29['hms'] = [x.strftime('%H-%M-%S') for x in df_29['startTime']]
    s2 = pd.merge(left=df_29, right=df_pre, on=['stationID', 'hms'], how='left')
    s2['inNums'] = s2['inNums_y']
    s2['outNums'] = s2['outNums_x']
    s2 = s2.loc[:, ['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].fillna(0)
    s2 = s2.sort_values(by=['stationID', 'startTime'])
    s2['inNums'] = s2['inNums'].astype('int')
    s2['outNums'] = s2['outNums'].astype('int')
    s2['startTime'] = [x.strftime('%Y-%m-%d %H:%M:%S') for x in s2['startTime']]
    s2.to_csv('data/Metro_testA/Metro_testA/ans.csv', index=False)


def test_data_2_feature():
    test = pd.read_csv('data/Metro_testA/Metro_testA/testA_submit_2019-01-29.csv')


def original_2_feature(path):
    """
    原始数据->10分钟聚合
    :param path:
    :return:
    """
    df = pd.read_csv(path)
    df['day'] = df['time'].apply(lambda x: int(x[8:10]))
    df['week'] = pd.to_datetime(df['time']).dt.dayofweek + 1
    df['weekend'] = (pd.to_datetime(df.time).dt.weekday >= 5).astype(int)
    df['hour'] = df['time'].apply(lambda x: int(x[11:13]))
    df['minute'] = df['time'].apply(lambda x: int(x[14:15] + '0'))
    result = df.groupby(['stationID', 'week', 'weekend', 'day', 'hour', 'minute']).status.agg(
        ['count', 'sum']).reset_index()

    tmp = df.groupby(['stationID'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID')
    result = result.merge(tmp, on=['stationID'], how='left')
    tmp = df.groupby(['stationID', 'hour'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID_hour')
    result = result.merge(tmp, on=['stationID', 'hour'], how='left')
    tmp = df.groupby(['stationID', 'hour', 'minute'])['deviceID'].nunique().reset_index(
        name='nuni_deviceID_of_stationID_hour_minute')
    result = result.merge(tmp, on=['stationID', 'hour', 'minute'], how='left')

    result['inNums'] = result['sum']
    result['outNums'] = result['count'] - result['sum']

    result['day_since_first'] = result['day'] - 1
    result.fillna(0, inplace=True)
    del result['sum'], result['count']

    return result


def fix_day(d):
    if d in [1, 2, 3, 4]:
        return d
    elif d in [7, 8, 9, 10, 11]:
        return d - 2
    elif d in [14, 15, 16, 17, 18]:
        return d - 4
    elif d in [21, 22, 23, 24, 25]:
        return d - 6
    elif d in [28, 29]:
        return d - 8


def test_29_data_process(df):
    """
    29日测试数据转换为特征
    :param df:
    :return:
    """
    df['week'] = pd.to_datetime(df['startTime']).dt.dayofweek + 1
    df['weekend'] = (pd.to_datetime(df.startTime).dt.weekday >= 5).astype(int)
    df['day'] = df['startTime'].apply(lambda x: int(x[8:10]))
    df['hour'] = df['startTime'].apply(lambda x: int(x[11:13]))
    df['minute'] = df['startTime'].apply(lambda x: int(x[14:15] + '0'))
    df['day_since_first'] = df['day'] - 1
    df = df.drop(['startTime', 'endTime'], axis=1)
    return df


def in_out_feature(df):
    """
    构造进站出站的特征
    :param df:
    :return:
    """
    tmp = df.groupby(['stationID', 'week', 'hour', 'minute'], as_index=False)['inNums'].agg({
        'inNums_whm_max': 'max',
        'inNums_whm_min': 'min',
        'inNums_whm_mean': 'mean'
    })
    df = df.merge(tmp, on=['stationID', 'week', 'hour', 'minute'], how='left')

    tmp = df.groupby(['stationID', 'week', 'hour', 'minute'], as_index=False)['outNums'].agg({
        'outNums_whm_max': 'max',
        'outNums_whm_min': 'min',
        'outNums_whm_mean': 'mean'
    })
    df = df.merge(tmp, on=['stationID', 'week', 'hour', 'minute'], how='left')

    tmp = df.groupby(['stationID', 'week', 'hour'], as_index=False)['inNums'].agg({
        'inNums_wh_max': 'max',
        'inNums_wh_min': 'min',
        'inNums_wh_mean': 'mean'
    })
    df = df.merge(tmp, on=['stationID', 'week', 'hour'], how='left')

    tmp = df.groupby(['stationID', 'week', 'hour'], as_index=False)['outNums'].agg({
        'outNums_wh_max': 'max',
        'outNums_wh_min': 'min',
        'outNums_wh_mean': 'mean'
    })
    df = df.merge(tmp, on=['stationID', 'week', 'hour'], how='left')
    return df


def last_day_feature(df):
    """
    增加前一日特征[29日前一日数据用的是28日,需要更换成前几日的]
    :param df:
    :return:
    """

    def get_refer_day(d):
        if d == 20:
            return 29
        else:
            return d + 1

    stat_columns = ['inNums', 'outNums']

    df_00 = df[df.day == 1]
    df_00['day'] = df_00['day'] - 1

    df = pd.concat([df, df_00], axis=0, ignore_index=True)
    df['day'] = df['day'].apply(get_refer_day)
    for f in stat_columns:
        df.rename(columns={f: f + '_last'}, inplace=True)

    df = df[['stationID', 'day', 'hour', 'minute', 'inNums_last', 'outNums_last']]
    return df


def recover_day(d):
    """
    将日期还原
    :param d:
    :return:
    """
    if d in [1, 2, 3, 4]:
        return d
    elif d in [5, 6, 7, 8, 9]:
        return d + 2
    elif d in [10, 11, 12, 13, 14]:
        return d + 4
    elif d in [15, 16, 17, 18, 19]:
        return d + 6
    elif d == 20:
        return d + 8
    else:
        return d


def feature_engineering():
    test_29 = pd.read_csv('data/Metro_testA/Metro_testA/testA_submit_2019-01-29.csv')

    data = original_2_feature('data/Metro_testA/Metro_testA/testA_record_2019-01-28.csv')

    for _dt in dt:
        print(_dt)
        df = original_2_feature('data/Metro_train/Metro_train/record_{}.csv'.format(_dt))
        data = pd.concat([data, df], axis=0, ignore_index=True)
        print(data.day.unique())

    # data
    # with open('data/Metro_train/Metro_train/baseline_train_data_b.csv', 'wb') as f:
    #     pickle.dump(data, f)

    with open('data/Metro_train/Metro_train/baseline_train_data_b.csv', 'rb') as f:
        data = pickle.load(f)

    data = data[(data.day != 5) & (data.day != 6)]
    data = data[(data.day != 12) & (data.day != 13)]
    data = data[(data.day != 19) & (data.day != 20)]
    data = data[(data.day != 26) & (data.day != 27)]

    data['day'] = data['day'].apply(fix_day)
    data = pd.concat([data, test_29_data_process(test_29.copy())], axis=0, ignore_index=True)

    data = data.merge(last_day_feature(data.copy()), on=['stationID', 'day', 'hour', 'minute'], how='left').fillna(0)

    data = in_out_feature(data)
    data.day = data.day.apply(recover_day)

    with open('data/Metro_train/Metro_train/baseline_train.csv', 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    # csv2df()
    # all_sample_10min()
    # concat_10min()
    # concat_10min_train_test_data()
    # test_29_data()
    feature_engineering()

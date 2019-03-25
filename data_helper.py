import pickle

import numpy as np
import pandas as pd

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


def test_29():
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


if __name__ == '__main__':
    csv2df()
    all_sample_10min()
    concat_10min()
    concat_10min_train_test_data()
    test_29()

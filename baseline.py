import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

plt.style.use('seaborn')
sns.set(font_scale=1)
pd.set_option('display.max_columns', 500)

test = pd.read_csv('data/Metro_testA/Metro_testA/testA_submit_2019-01-29.csv')
test_28 = pd.read_csv('data/Metro_testA/Metro_testA/testA_record_2019-01-28.csv')


def get_base_features(df_):
    df = df_.copy()

    # base time
    df['day'] = df['time'].apply(lambda x: int(x[8:10]))
    df['week'] = pd.to_datetime(df['time']).dt.dayofweek + 1
    df['weekend'] = (pd.to_datetime(df.time).dt.weekday >= 5).astype(int)
    df['hour'] = df['time'].apply(lambda x: int(x[11:13]))
    df['minute'] = df['time'].apply(lambda x: int(x[14:15] + '0'))

    # count,sum
    result = df.groupby(['stationID', 'week', 'weekend', 'day', 'hour', 'minute']).status.agg(
        ['count', 'sum']).reset_index()

    # nunique
    tmp = df.groupby(['stationID'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID')
    result = result.merge(tmp, on=['stationID'], how='left')
    tmp = df.groupby(['stationID', 'hour'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID_hour')
    result = result.merge(tmp, on=['stationID', 'hour'], how='left')
    tmp = df.groupby(['stationID', 'hour', 'minute'])['deviceID'].nunique(). \
        reset_index(name='nuni_deviceID_of_stationID_hour_minute')
    result = result.merge(tmp, on=['stationID', 'hour', 'minute'], how='left')

    # in,out
    result['inNums'] = result['sum']
    result['outNums'] = result['count'] - result['sum']

    #
    result['day_since_first'] = result['day'] - 1
    result.fillna(0, inplace=True)
    del result['sum'], result['count']

    return result


data = get_base_features(test_28)

datetime = pd.date_range('2019-01-01', '2019-01-25')
dt = [dt.strftime("%Y-%m-%d") for dt in datetime]

data_list = ['data/Metro_train/Metro_train/record_{}.csv'.format(d) for d in dt]

for p in data_list:
    df = pd.read_csv(p)
    df = get_base_features(df)
    data = pd.concat([data, df], axis=0, ignore_index=True)

data = data[(data.day != 5) & (data.day != 6)]
data = data[(data.day != 12) & (data.day != 13)]
data = data[(data.day != 19) & (data.day != 20)]
data = data[(data.day != 26) & (data.day != 27)]


def fix_day(d):
    if d in [1, 2, 3, 4]:
        return d
    elif d in [7, 8, 9, 10, 11]:
        return d - 2
    elif d in [14, 15, 16, 17, 18]:
        return d - 4
    elif d in [21, 22, 23, 24, 25]:
        return d - 6
    elif d in [28]:
        return d - 8


data['day'] = data['day'].apply(fix_day)

test['week'] = pd.to_datetime(test['startTime']).dt.dayofweek + 1
test['weekend'] = (pd.to_datetime(test.startTime).dt.weekday >= 5).astype(int)
test['day'] = test['startTime'].apply(lambda x: int(x[8:10]))
test['hour'] = test['startTime'].apply(lambda x: int(x[11:13]))
test['minute'] = test['startTime'].apply(lambda x: int(x[14:15] + '0'))
test['day_since_first'] = test['day'] - 1
test = test.drop(['startTime', 'endTime'], axis=1)
data = pd.concat([data, test], axis=0, ignore_index=True)

stat_columns = ['inNums', 'outNums']


def get_refer_day(d):
    if d == 20:
        return 29
    else:
        return d + 1


tmp = data.copy()
tmp_df = tmp[tmp.day == 1]
tmp_df['day'] = tmp_df['day'] - 1
tmp = pd.concat([tmp, tmp_df], axis=0, ignore_index=True)
tmp['day'] = tmp['day'].apply(get_refer_day)

for f in stat_columns:
    tmp.rename(columns={f: f + '_last'}, inplace=True)

tmp = tmp[['stationID', 'day', 'hour', 'minute', 'inNums_last', 'outNums_last']]

data = data.merge(tmp, on=['stationID', 'day', 'hour', 'minute'], how='left')
data.fillna(0, inplace=True)

tmp = data.groupby(['stationID', 'week', 'hour', 'minute'], as_index=False)['inNums'].agg({
    'inNums_whm_max': 'max',
    'inNums_whm_min': 'min',
    'inNums_whm_mean': 'mean'
})
data = data.merge(tmp, on=['stationID', 'week', 'hour', 'minute'], how='left')

tmp = data.groupby(['stationID', 'week', 'hour', 'minute'], as_index=False)['outNums'].agg({
    'outNums_whm_max': 'max',
    'outNums_whm_min': 'min',
    'outNums_whm_mean': 'mean'
})
data = data.merge(tmp, on=['stationID', 'week', 'hour', 'minute'], how='left')

tmp = data.groupby(['stationID', 'week', 'hour'], as_index=False)['inNums'].agg({
    'inNums_wh_max': 'max',
    'inNums_wh_min': 'min',
    'inNums_wh_mean': 'mean'
})
data = data.merge(tmp, on=['stationID', 'week', 'hour'], how='left')

tmp = data.groupby(['stationID', 'week', 'hour'], as_index=False)['outNums'].agg({
    # 'outNums_wh_max'    : 'max',
    # 'outNums_wh_min'    : 'min',
    'outNums_wh_mean': 'mean'
})
data = data.merge(tmp, on=['stationID', 'week', 'hour'], how='left')


def recover_day(d):
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


all_columns = [f for f in data.columns if f not in ['weekend', 'inNums', 'outNums']]
### all data
all_data = data[(data.day != 29)]
all_data['day'] = all_data['day'].apply(recover_day)
X_data = all_data[all_columns].values

train = data[data.day < 20]
train['day'] = train['day'].apply(recover_day)
X_train = train[all_columns].values

valid = data[data.day == 20]
valid['day'] = valid['day'].apply(recover_day)
X_valid = valid[all_columns].values

test = data[data.day == 29]
X_test = test[all_columns].values

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 127,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_seed': 0,
    'bagging_freq': 1,
    'verbose': 1,
    'reg_alpha': 20,
    'reg_lambda': 20
}

######################################################inNums
y_train = train['inNums']
y_valid = valid['inNums']
y_data = all_data['inNums']
lgb_train = lgb.Dataset(X_train, y_train)
lgb_evals = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=[lgb_train, lgb_evals],
                valid_names=['train', 'valid'],
                early_stopping_rounds=200,
                verbose_eval=1000,
                )

### all_data
lgb_train = lgb.Dataset(X_data, y_data)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=gbm.best_iteration,
                valid_sets=[lgb_train],
                valid_names=['train'],
                verbose_eval=1000,
                )
test['inNums'] = gbm.predict(X_test)

######################################################outNums
y_train = train['outNums']
y_valid = valid['outNums']
y_data = all_data['outNums']
lgb_train = lgb.Dataset(X_train, y_train)
lgb_evals = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=[lgb_train, lgb_evals],
                valid_names=['train', 'valid'],
                early_stopping_rounds=200,
                verbose_eval=1000,
                )

### all_data
lgb_train = lgb.Dataset(X_data, y_data)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=gbm.best_iteration,
                valid_sets=[lgb_train],
                valid_names=['train'],
                verbose_eval=1000,
                )
test['outNums'] = gbm.predict(X_test)

sub = pd.read_csv('data/Metro_testA/Metro_testA/testA_submit_2019-01-29.csv')
sub['inNums'] = test['inNums'].values
sub['outNums'] = test['outNums'].values
# 结果修正
sub.loc[sub.inNums < 0, 'inNums'] = 0
sub.loc[sub.outNums < 0, 'outNums'] = 0

flag = (sub['startTime'] <= '2019-01-29 05:20:00') | (sub['startTime'] >= '2019-01-29 22:50:00')
sub.loc[flag, 'inNums'] = 0

flag = (sub['startTime'] <= '2019-01-29 05:40:00') | (sub['startTime'] >= '2019-01-29 23:00:00')
sub.loc[flag, 'outNums'] = 0
sub[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv('data/result/sub_model.csv', index=False)

# data = feature_engineering()

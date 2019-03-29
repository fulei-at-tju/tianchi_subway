import pickle
import warnings
import lightgbm as lgb
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

with open('data/Metro_train/Metro_train/baseline_train.csv', 'rb') as f:
    data = pickle.load(f)

all_columns = [f for f in data.columns if f not in ['weekend', 'inNums', 'outNums']]

all_data = data[(data.day != 29) & (data.day != 28)]
X_data = all_data[all_columns].values

train = data[data.day < 25]
X_train = train[all_columns].values

valid = data[data.day == 25]
X_valid = valid[all_columns].values

test = data[data.day == 29]
X_test = test[all_columns].values


def regular(df):
    df.loc[df.inNums < 0, 'inNums'] = 0
    df.loc[df.outNums < 0, 'outNums'] = 0

    flag = (df['startTime'] <= '2019-01-29 05:20:00') | (df['startTime'] >= '2019-01-29 22:50:00')
    df.loc[flag, 'inNums'] = 0

    flag = (df['startTime'] <= '2019-01-29 05:40:00') | (df['startTime'] >= '2019-01-29 23:00:00')
    df.loc[flag, 'outNums'] = 0
    return df


def lgb_model():
    """
    lgb 模型
    :return:
    """
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 127,
        'learning_rate': 0.005,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_seed': 0,
        'bagging_freq': 1,
        'verbose': 1,
        'reg_alpha': 15,
        'reg_lambda': 15
    }

    y_train = train['inNums']
    y_valid = valid['inNums']
    y_data = all_data['inNums']
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_evals = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=50000,
                    valid_sets=[lgb_train, lgb_evals],
                    valid_names=['train', 'valid'],
                    early_stopping_rounds=500,
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
                    num_boost_round=50000,
                    valid_sets=[lgb_train, lgb_evals],
                    valid_names=['train', 'valid'],
                    early_stopping_rounds=500,
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

    ans = pd.read_csv('data/Metro_testA/Metro_testA/testA_submit_2019-01-29.csv')
    ans['inNums'] = test['inNums'].values
    ans['outNums'] = test['outNums'].values
    # 结果修正
    ans = regular(ans.copy())
    ans[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv('data/result/sub_model.csv', index=False)


if __name__ == '__main__':
    lgb_model()